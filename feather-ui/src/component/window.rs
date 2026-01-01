// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::component::{ChildOf, DynComponent};
use crate::input::{ModifierKeys, MouseState, RawEvent};
use crate::layout::root;
use crate::reactive::{DynSignal, SignalMap};
use crate::reactive::{MutableSignal, sample, sample_val};
use crate::render::compositor::Compositor;
use crate::rtree::Node;
use crate::{PxPoint, PxVector, RelDim, RelVector, graphics, layout, rtree};
use alloc::sync::Arc;
use core::f32;
use eyre::{OptionExt, Result};
use guillotiere::euclid::default::Rotation3D;
use guillotiere::euclid::{Point3D, Size2D};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{DeviceId, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{CursorIcon, WindowAttributes};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum WindowNodeTrack {
    Focus = 0,
    Hover = 1,
    Capture = 2,
}

/// Holds our internal mutable state for this window
pub struct WindowState {
    pub surface: wgpu::Surface<'static>, // Ensure surface get dropped before window
    pub window: Arc<winit::window::Window>,
    pub dpi: MutableSignal<crate::RelDim>,
    pub config: MutableSignal<wgpu::SurfaceConfiguration>,
    pub surface_dim: DynSignal<Size2D<u32, crate::Pixel>>,
    all_buttons: u16,
    modifiers: u8,
    last_mouse: PhysicalPosition<f32>,
    pub driver: Arc<graphics::Driver>,
    trackers: [HashMap<DeviceId, Weak<Node>>; 3],
    pub compositor: Compositor,
    pub clipstack: Vec<crate::PxRect>, /* Current clipping rectangle stack. These only get added
                                        * to the GPU clip list if something is rotated */
    pub layers: Vec<std::rc::Weak<crate::render::compositor::Layer>>, /* All layers that render directly to the final
                                                                       * compositor */
}

const BACKCOLOR: wgpu::Color = wgpu::Color {
    r: 0.1,
    g: 0.2,
    b: 0.3,
    a: 1.0,
};

impl WindowState {
    pub(crate) fn new(
        attributes: &WindowAttributes,
        driver_ref: &mut std::sync::Weak<graphics::Driver>,
        instance: &wgpu::Instance,
        event_loop: &ActiveEventLoop,
        on_driver: &mut Option<Box<dyn FnOnce(std::sync::Weak<graphics::Driver>) + 'static>>,
    ) -> Result<Self> {
        let window = Arc::new(event_loop.create_window(attributes.clone())?);

        let surface: wgpu::Surface<'static> = instance.create_surface(window.clone())?;

        let driver = futures_lite::future::block_on(crate::graphics::Driver::new(
            driver_ref, instance, &surface, on_driver,
        ))?;

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&driver.adapter, size.width, size.height)
            .ok_or_eyre("Failed to find a default configuration")?;
        let view_format = config.format.add_srgb_suffix();
        //let view_format = config.format.remove_srgb_suffix();
        config.format = view_format;
        config.view_formats.push(view_format);
        surface.configure(&driver.device, &config);

        let compositor = Compositor::new(
            &driver.device,
            &driver.shared,
            &driver.atlas.read().view,
            &driver.layer_atlas[0].read().view,
            config.view_formats[0],
            false,
        );

        let config = MutableSignal::new(config);
        let mut windowstate = Self {
            modifiers: 0,
            all_buttons: 0,
            last_mouse: PhysicalPosition::new(f32::NAN, f32::NAN),
            config: config.clone(),
            dpi: MutableSignal::new(RelDim::splat(window.scale_factor() as f32)),
            surface_dim: config
                .map(|c| Size2D::<u32, crate::Pixel>::new(c.width, c.height))
                .into(),
            surface,
            window,
            driver: driver.clone(),
            trackers: Default::default(),
            compositor,
            clipstack: Vec::new(),
            layers: Vec::new(),
        };

        windowstate.resize(size);

        // This causes an unwanted flash, but makes it easier to capture the initial
        // frame for debugging, so it's left here to be uncommented for
        // debugging purposes let frame =
        // windowstate.surface.get_current_texture()?; frame.present();

        Ok(windowstate)
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.config.set_with(|config| {
            config.width = size.width;
            config.height = size.height;
        });
        self.surface
            .configure(&self.driver.device, &sample(&self.config.clone()));
    }

    pub(crate) fn nodes(&self, tracker: WindowNodeTrack) -> SmallVec<[Weak<Node>; 4]> {
        self.trackers[tracker as usize]
            .values()
            .map(|node| node.clone())
            .collect()
    }

    pub(crate) fn drain(&mut self, tracker: WindowNodeTrack) -> SmallVec<[Weak<Node>; 4]> {
        self.trackers[tracker as usize]
            .drain()
            .map(|(_, v)| v)
            .collect()
    }

    pub(crate) fn set(
        &mut self,
        tracker: WindowNodeTrack,
        device_id: DeviceId,
        node: Weak<Node>,
    ) -> Option<Weak<Node>> {
        self.trackers[tracker as usize].insert(device_id, node)
    }

    pub(crate) fn remove(
        &mut self,
        tracker: WindowNodeTrack,
        device_id: &DeviceId,
    ) -> Option<Weak<Node>> {
        self.trackers[tracker as usize].remove(device_id)
    }

    pub(crate) fn get(&self, tracker: WindowNodeTrack, device_id: &DeviceId) -> Option<Weak<Node>> {
        self.trackers[tracker as usize]
            .get(device_id)
            .map(|v| v.clone())
    }

    pub(crate) fn draw(&mut self, mut encoder: wgpu::CommandEncoder) {
        let frame = self.surface.get_current_texture().unwrap();
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let surface_dim = sample(&self.surface_dim);
        self.compositor
            .prepare(&self.driver, &mut encoder, surface_dim.to_f32());

        {
            let mut backcolor = BACKCOLOR;
            if frame.texture.format().is_srgb() {
                backcolor.r = crate::color::srgb_to_linear(backcolor.r);
                backcolor.g = crate::color::srgb_to_linear(backcolor.g);
                backcolor.b = crate::color::srgb_to_linear(backcolor.b);
            }

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Window Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(backcolor),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let viewport_dim = surface_dim.to_f32();
            pass.set_viewport(0.0, 0.0, viewport_dim.width, viewport_dim.height, 0.0, 1.0);

            self.compositor.draw(&self.driver, &mut pass, 0, 0);
            self.compositor.cleanup();
        }

        self.driver.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

/// Represents an OS window. All outline functions must return a set of windows
/// as a result of their evaluation, which represents all the windows that are
/// currently open as part of the application. The ID of the window that
/// a particular component belongs to is propagated down the outline evaluation
/// phase, because this is needed to acquire window-specific information that
/// depends on which monitor the OS thinks the window belongs to, like DPI
/// or orientation.
pub struct Window {
    pub attributes: MutableSignal<WindowAttributes>,
    pub child: MutableSignal<Arc<ChildOf<dyn root::Prop>>>,
}

impl std::fmt::Debug for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Window")
            .field("attributes", &self.attributes)
            .finish()
    }
}

impl Window {
    pub(crate) fn layout(
        &self,
        state: &WindowState,
    ) -> Box<dyn crate::layout::Layout<Props = DynSignal<Size2D<u32, crate::Pixel>>>> {
        let driver = state.driver.clone();
        let size = state.surface_dim.clone();
        let dpi = state.dpi.clone();

        let children = self
            .child
            .clone()
            .map_ex(move |child| child.layout(driver.clone(), dpi.clone()));

        Box::new(
            layout::Node::<DynSignal<Size2D<u32, crate::Pixel>>, dyn root::Prop, ()> {
                props: Rc::new(size),
                children: children.into(),
                renderable: None,
            },
        )
    }
}

impl Window {
    pub fn new(
        attributes: WindowAttributes,
        child: Box<dyn DynComponent<dyn crate::layout::base::Empty>>,
    ) -> Self {
        Self {
            attributes: MutableSignal::new(attributes),
            child: MutableSignal::new(Arc::from(child)),
        }
    }

    /// Returns true if the event is consumed, false if it should be forwarded
    #[allow(clippy::result_unit_err)]
    pub fn on_window_event(
        window: &mut WindowState,
        root: Rc<rtree::Node>,
        event: WindowEvent, // EventStream<WindowEvent>
        driver: std::sync::Weak<graphics::Driver>,
    ) -> bool {
        use crate::Convert;

        let dpi = window.dpi.clone();
        let inner = window.window.clone();
        match event {
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                window.dpi.replace(RelDim::splat(scale_factor as f32));
                window.window.request_redraw();
                true
            }
            WindowEvent::ModifiersChanged(m) => {
                window.modifiers = if m.state().control_key() {
                    ModifierKeys::Control as u8
                } else {
                    0
                } | if m.state().alt_key() {
                    ModifierKeys::Alt as u8
                } else {
                    0
                } | if m.state().shift_key() {
                    ModifierKeys::Shift as u8
                } else {
                    0
                } | if m.state().super_key() {
                    ModifierKeys::Super as u8
                } else {
                    0
                };
                true
            }
            WindowEvent::Resized(new_size) => {
                // Resize events can sometimes give empty sizes if the window is minimized
                if new_size.height > 0 && new_size.width > 0 {
                    window.resize(new_size);
                }
                // On macos the window needs to be redrawn manually after resizing
                window.window.request_redraw();
                true
            }
            WindowEvent::CloseRequested => {
                // If this returns Reject(data), the close request will be ignored
                true
            }
            WindowEvent::RedrawRequested => {
                panic!("Don't process this with on_window_event");
            }
            WindowEvent::Focused(acquired) => {
                // When the window loses or gains focus, we send a focus event to our children
                // that had focus, but we don't forget or change which children
                // have focus.
                let evt = RawEvent::Focus {
                    acquired,
                    window: window.window.clone(),
                };

                // We have to collect this map so we aren't borrowing manager twice
                let nodes: SmallVec<[Weak<Node>; 4]> = window.nodes(WindowNodeTrack::Focus);

                for node in nodes.iter().filter_map(|x| x.upgrade()) {
                    let _ = node.inject_event(&evt, evt.kind(), window, &root);
                }
                true
            }
            _ => {
                let e = match event {
                    WindowEvent::Ime(winit::event::Ime::Commit(s)) => RawEvent::Key {
                        device_id: DeviceId::dummy(), /* TODO: No way to derive originating
                                                       * keyboard from IME event???? */
                        physical_key: winit::keyboard::PhysicalKey::Unidentified(
                            winit::keyboard::NativeKeyCode::Unidentified,
                        ),
                        location: winit::keyboard::KeyLocation::Standard,
                        down: false,
                        logical_key: winit::keyboard::Key::Character(s.into()),
                        modifiers: 0,
                    },
                    WindowEvent::KeyboardInput {
                        device_id,
                        event,
                        is_synthetic: _,
                    } => RawEvent::Key {
                        device_id,
                        physical_key: event.physical_key,
                        location: event.location,
                        down: event.state.is_pressed(),
                        logical_key: event.logical_key,
                        modifiers: window.modifiers
                            | if event.repeat {
                                ModifierKeys::Held as u8
                            } else {
                                0
                            },
                    },
                    WindowEvent::CursorMoved {
                        device_id,
                        position,
                    } => {
                        window.last_mouse =
                            PhysicalPosition::new(position.x as f32, position.y as f32);
                        RawEvent::MouseMove {
                            device_id,
                            pos: window.last_mouse.to(),
                            all_buttons: window.all_buttons,
                            modifiers: window.modifiers,
                        }
                    }
                    WindowEvent::CursorEntered { device_id } => {
                        #[cfg(windows)]
                        {
                            let points = {
                                let mut p = unsafe { std::mem::zeroed() };
                                unsafe {
                                    windows_sys::Win32::UI::WindowsAndMessaging::GetCursorPos(
                                        &mut p,
                                    )
                                };
                                p
                            };

                            window.last_mouse =
                                PhysicalPosition::new(points.x as f32, points.y as f32);
                        }

                        // If the cursor enters and no buttons are pressed, ensure any captured
                        // state is reset
                        if window.all_buttons == 0 {
                            // No need to inject a mousemove event here, as MouseOn is already being
                            // sent.
                            window.remove(WindowNodeTrack::Capture, &device_id);
                        }
                        RawEvent::MouseOn {
                            device_id,
                            pos: window.last_mouse.to(),
                            all_buttons: window.all_buttons,
                            modifiers: window.modifiers,
                        }
                    }
                    WindowEvent::CursorLeft { device_id } => {
                        window.last_mouse = PhysicalPosition::new(f32::NAN, f32::NAN);
                        RawEvent::MouseOff {
                            device_id,
                            all_buttons: window.all_buttons,
                            modifiers: window.modifiers,
                        }
                    }
                    WindowEvent::MouseWheel {
                        device_id,
                        delta,
                        phase,
                    } => match delta {
                        winit::event::MouseScrollDelta::LineDelta(x, y) => RawEvent::MouseScroll {
                            device_id,
                            state: phase.into(),
                            pos: window.last_mouse.to(),
                            delta: Err(RelVector::new(x, y)),
                        },
                        winit::event::MouseScrollDelta::PixelDelta(physical_position) => {
                            RawEvent::MouseScroll {
                                device_id,
                                state: phase.into(),
                                pos: window.last_mouse.to(),
                                delta: Ok(physical_position.to().to_f32().to_vector()),
                            }
                        }
                    },
                    WindowEvent::MouseInput {
                        device_id,
                        state,
                        button,
                    } => {
                        let b = button.into();

                        if state == winit::event::ElementState::Pressed {
                            window.all_buttons |= b as u16;
                        } else {
                            window.all_buttons &= !(b as u16);
                        }

                        RawEvent::Mouse {
                            device_id,
                            state: if state == winit::event::ElementState::Pressed {
                                MouseState::Down
                            } else {
                                MouseState::Up
                            },
                            pos: window.last_mouse.to(),
                            button: b,
                            all_buttons: window.all_buttons,
                            modifiers: window.modifiers,
                        }
                    }
                    WindowEvent::AxisMotion {
                        device_id,
                        axis,
                        value,
                    } => RawEvent::JoyAxis {
                        device_id,
                        value,
                        axis,
                    },
                    WindowEvent::Touch(touch) => RawEvent::Touch {
                        device_id: touch.device_id,
                        state: touch.phase.into(),
                        pos: touch.location.to().to_f32().to_3d(),
                        index: touch.id as u32,
                        angle: Rotation3D::<f32>::identity(),
                        pressure: match touch.force {
                            Some(winit::event::Force::Normalized(x)) => x,
                            Some(winit::event::Force::Calibrated {
                                force,
                                max_possible_force: _,
                                altitude_angle: _,
                            }) => force,
                            None => 0.0,
                        },
                    },
                    _ => return false,
                };

                match e {
                    RawEvent::MouseMove { .. } | RawEvent::MouseOn { .. } => {
                        if let Some(d) = driver.upgrade() {
                            *d.cursor.write() = CursorIcon::Default;
                        }
                    }
                    _ => (),
                }
                let r = match e {
                    RawEvent::Drag => false,
                    RawEvent::Focus { .. } => false,
                    RawEvent::JoyAxis { device_id: _, .. }
                    | RawEvent::JoyButton { device_id: _, .. }
                    | RawEvent::JoyOrientation { device_id: _, .. }
                    | RawEvent::Key { device_id: _, .. } => {
                        // We have to collect this map so we aren't borrowing manager twice
                        let nodes: SmallVec<[Weak<Node>; 4]> = window.nodes(WindowNodeTrack::Focus);

                        // Currently, we always duplicate key/joystick events to all focused
                        // elements. Later, we may map specific keyboards to
                        // specific mouse input device IDs. We use a fold instead of any() to avoid
                        // short-circuiting.
                        if nodes.iter().fold(false, |ok, node| {
                            ok | node
                                .upgrade()
                                .map(|n| n.inject_event(&e, e.kind(), window, &root).0)
                                .unwrap_or(false)
                        }) {
                            true
                        } else {
                            false
                        }
                    }
                    RawEvent::MouseOff { .. } => {
                        // We have to collect this map so we aren't borrowing manager twice
                        let nodes: SmallVec<[Weak<Node>; 4]> = window.drain(WindowNodeTrack::Hover);

                        // Send a mouseoff event to all captures, but don't drain the captures so we
                        // have a chance to recover.
                        let capture_nodes: SmallVec<[Weak<Node>; 4]> =
                            window.nodes(WindowNodeTrack::Capture);

                        for node in nodes.iter().filter_map(|x| x.upgrade()) {
                            let _ = node.inject_event(&e, e.kind(), window, &root);
                        }

                        // While we could recover the offset here, we don't so we can be consistent
                        // about MouseOff not having offset.
                        for node in capture_nodes.iter().filter_map(|x| x.upgrade()) {
                            let _ = node.inject_event(&e, e.kind(), window, &root);
                        }

                        true
                    }
                    RawEvent::Mouse {
                        device_id,
                        pos: PxPoint { x, y, .. },
                        ..
                    }
                    | RawEvent::MouseOn {
                        device_id,
                        pos: PxPoint { x, y, .. },
                        ..
                    }
                    | RawEvent::MouseMove {
                        device_id,
                        pos: PxPoint { x, y, .. },
                        ..
                    }
                    | RawEvent::Drop {
                        device_id,
                        pos: PhysicalPosition { x, y },
                        ..
                    }
                    | RawEvent::MouseScroll {
                        device_id,
                        pos: PxPoint { x, y, .. },
                        ..
                    }
                    | RawEvent::Touch {
                        device_id,
                        pos: Point3D::<f32, crate::Pixel> { x, y, .. },
                        ..
                    } => {
                        if let Some(node) = window
                            .get(WindowNodeTrack::Capture, &device_id)
                            .and_then(|x| x.upgrade())
                        {
                            return node.clone().inject_event(&e, e.kind(), window, &root).0;
                        }

                        root.process(
                            &e,
                            e.kind(),
                            PxPoint::new(x, y),
                            PxVector::zero(),
                            sample_val(dpi.clone()),
                            &driver,
                            window,
                            &root,
                        )
                    }
                };

                if !r {
                    match e {
                        // If everything rejected the mousemove, remove hover from all elements
                        RawEvent::MouseMove {
                            device_id,
                            modifiers,
                            all_buttons,
                            ..
                        } => {
                            let evt = RawEvent::MouseOff {
                                device_id,
                                modifiers,
                                all_buttons,
                            };

                            // Drain() holds a reference, so we still have to collect these to avoid
                            // borrowing manager twice
                            let nodes: SmallVec<[Weak<Node>; 4]> =
                                window.drain(WindowNodeTrack::Hover);

                            for node in nodes.iter().filter_map(|x| x.upgrade()) {
                                let _ = node.inject_event(&evt, evt.kind(), window, &root);
                            }
                        }
                        // If everything rejected a mousedown, remove all focused elements
                        RawEvent::Mouse {
                            state: MouseState::Down,
                            button: crate::input::MouseButton::Left,
                            ..
                        }
                        | RawEvent::Mouse {
                            state: MouseState::Down,
                            button: crate::input::MouseButton::Middle,
                            ..
                        }
                        | RawEvent::Mouse {
                            state: MouseState::Down,
                            button: crate::input::MouseButton::Right,
                            ..
                        } => {
                            let evt = RawEvent::Focus {
                                acquired: false,
                                window: window.window.clone(),
                            };

                            // Drain() holds a reference, so we still have to collect these to avoid
                            // borrowing manager twice
                            let nodes: SmallVec<[Weak<Node>; 4]> =
                                window.drain(WindowNodeTrack::Focus);

                            for node in nodes.iter().filter_map(|x| x.upgrade()) {
                                let _ = node.inject_event(&evt, evt.kind(), window, &root);
                            }
                        }
                        _ => (),
                    }
                }

                // After finishing all processing, if we were processing a mousemove or mouseon
                // event, update our cursor
                match e {
                    RawEvent::MouseMove { .. } | RawEvent::MouseOn { .. } => {
                        if let Some(d) = driver.upgrade() {
                            inner.set_cursor(*d.cursor.read());
                        }
                    }
                    _ => (),
                }
                r
            }
        }
    }
}
