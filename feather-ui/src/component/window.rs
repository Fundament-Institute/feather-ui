// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::{Component, StateMachine};
use crate::component::ComponentWrap;
use crate::input::{ModifierKeys, MouseState, RawEvent};
use crate::layout::root;
use crate::render::compositor::Compositor;
use crate::rtree::Node;
use crate::{AbsDim, SourceID, StateMachineChild, StateManager, graphics, layout, rtree};
use alloc::sync::Arc;
use core::f32;
use eyre::{OptionExt, Result};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use ultraviolet::Vec2;
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
    pub config: wgpu::SurfaceConfiguration,
    all_buttons: u16,
    modifiers: u8,
    last_mouse: PhysicalPosition<f32>,
    pub dpi: Vec2,
    pub driver: Arc<graphics::Driver>,
    trackers: [HashMap<DeviceId, RcNode>; 3],
    lookup: HashMap<(Arc<SourceID>, u8), DeviceId>,
    pub compositor: Compositor,
    pub clipstack: Vec<crate::AbsRect>, // Current clipping rectangle stack. These only get added to the GPU clip list if something is rotated
    pub layers: Vec<std::sync::Weak<SourceID>>, // All layers that render directly to the final compositor
}

impl super::EventRouter for WindowState {
    type Input = ();
    type Output = ();
}

const BACKCOLOR: wgpu::Color = wgpu::Color {
    r: 0.1,
    g: 0.2,
    b: 0.3,
    a: 1.0,
};

impl WindowState {
    pub(crate) fn nodes(&self, tracker: WindowNodeTrack) -> SmallVec<[Weak<Node>; 4]> {
        self.trackers[tracker as usize]
            .values()
            .map(|RcNode(_, node)| node.clone())
            .collect()
    }

    pub(crate) fn drain(&mut self, tracker: WindowNodeTrack) -> SmallVec<[Weak<Node>; 4]> {
        self.trackers[tracker as usize]
            .drain()
            .map(|(_, RcNode(id, v))| {
                self.lookup.remove(&(id, tracker as u8));
                v
            })
            .collect()
    }

    pub(crate) fn set(
        &mut self,
        tracker: WindowNodeTrack,
        device_id: DeviceId,
        id: Arc<SourceID>,
        node: Weak<Node>,
    ) -> Option<Weak<Node>> {
        let n = self.trackers[tracker as usize]
            .insert(device_id, RcNode(id.clone(), node))
            .map(|RcNode(id, v)| {
                self.lookup.remove(&(id, tracker as u8));
                v
            });

        self.lookup.insert((id, tracker as u8), device_id);
        n
    }

    pub(crate) fn remove(
        &mut self,
        tracker: WindowNodeTrack,
        device_id: &DeviceId,
    ) -> Option<Weak<Node>> {
        self.trackers[tracker as usize]
            .remove(device_id)
            .map(|RcNode(id, v)| {
                self.lookup.remove(&(id, tracker as u8));
                v
            })
    }

    pub(crate) fn get(&self, tracker: WindowNodeTrack, device_id: &DeviceId) -> Option<Weak<Node>> {
        self.trackers[tracker as usize]
            .get(device_id)
            .map(|RcNode(_, v)| v.clone())
    }

    pub(crate) fn update_node(&mut self, id: Arc<SourceID>, node: Weak<Node>) {
        for i in 0..self.trackers.len() {
            if let Some(device) = self.lookup.get(&(id.clone(), i as u8)) {
                if let Some(RcNode(_, n)) = self.trackers[i].get_mut(device) {
                    *n = node.clone();
                }
            }
        }
    }

    pub(crate) fn draw(&mut self, mut encoder: wgpu::CommandEncoder) {
        let frame = self.surface.get_current_texture().unwrap();
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.compositor
            .prepare(&self.driver, &mut encoder, self.surface_dim());

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
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(backcolor),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let viewport_dim = self.surface_dim();
            pass.set_viewport(0.0, 0.0, viewport_dim.x, viewport_dim.y, 0.0, 1.0);

            self.compositor.draw(&self.driver, &mut pass, 0, 0);
            self.compositor.cleanup();
        }

        self.driver.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    pub fn surface_dim(&self) -> Vec2 {
        Vec2::new(self.config.width as f32, self.config.height as f32)
    }
}

pub(crate) struct RcNode(Arc<SourceID>, pub(crate) Weak<Node>);

impl PartialEq for RcNode {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialEq for WindowState {
    fn eq(&self, other: &Self) -> bool {
        self.surface == other.surface
            && Arc::ptr_eq(&self.window, &other.window)
            && self.config == other.config
            && self.all_buttons == other.all_buttons
            && self.modifiers == other.modifiers
            && self.last_mouse == other.last_mouse
            && self.dpi == other.dpi
            && Arc::ptr_eq(&self.driver, &other.driver)
            && self.trackers == other.trackers
    }
}

pub(crate) type WindowStateMachine = StateMachine<WindowState, 0>;

#[derive(Clone)]
pub struct Window {
    pub id: Arc<SourceID>,
    attributes: WindowAttributes,
    child: Box<dyn ComponentWrap<<dyn root::Prop as crate::component::Desc>::Child>>,
}

impl Component for Window {
    type Props = AbsDim;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        _: &graphics::Driver,
        _: &Arc<SourceID>,
    ) -> Box<dyn crate::layout::Layout<AbsDim>> {
        let inner = manager
            .get::<WindowStateMachine>(&self.id)
            .unwrap()
            .state
            .as_ref()
            .unwrap();
        let size = inner.window.inner_size();
        let driver = inner.driver.clone();
        Box::new(layout::Node::<AbsDim, dyn root::Prop> {
            props: Rc::new(crate::AbsDim(Vec2 {
                x: size.width as f32,
                y: size.height as f32,
            })),
            children: self.child.layout(manager, &driver, &self.id),
            id: Arc::downgrade(&self.id),
            renderable: None,
            layer: None,
        })
    }
}

impl StateMachineChild for Window {
    fn init(
        &self,
        _: &std::sync::Weak<graphics::Driver>,
    ) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        Err(crate::Error::UnhandledEvent)
    }

    fn apply_children(
        &self,
        f: &mut dyn FnMut(&dyn StateMachineChild) -> eyre::Result<()>,
    ) -> eyre::Result<()> {
        f(self.child.as_ref())
    }

    fn id(&self) -> Arc<SourceID> {
        self.id.clone()
    }
}

impl Window {
    pub(crate) fn init_custom(
        &self,
        manager: &mut StateManager,
        driver_ref: &mut std::sync::Weak<graphics::Driver>,
        instance: &wgpu::Instance,
        event_loop: &ActiveEventLoop,
        on_driver: &mut Option<Box<dyn FnOnce(std::sync::Weak<graphics::Driver>) + 'static>>,
    ) -> Result<()> {
        if manager.get::<WindowStateMachine>(&self.id).is_err() {
            let attributes = self.attributes.clone();

            let window = Arc::new(event_loop.create_window(attributes)?);

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

            let mut windowstate = WindowState {
                modifiers: 0,
                all_buttons: 0,
                last_mouse: PhysicalPosition::new(f32::NAN, f32::NAN),
                config,
                dpi: Vec2::broadcast(window.scale_factor() as f32),
                surface,
                window,
                driver: driver.clone(),
                trackers: Default::default(),
                lookup: Default::default(),
                compositor,
                clipstack: Vec::new(),
                layers: Vec::new(),
            };

            Window::resize(size, &mut windowstate);

            // This causes an unwanted flash, but makes it easier to capture the initial frame for debugging, so it's left here
            // to be uncommented for debugging purposes
            //let frame = windowstate.surface.get_current_texture().unwrap();
            //frame.present();

            manager.init(
                self.id.clone(),
                Box::new(StateMachine::<WindowState, 0> {
                    state: Some(windowstate),
                    output: [],
                    input_mask: 0,
                    changed: false,
                }),
            );
        }

        manager.init_child(self.child.as_ref(), driver_ref)?;
        Ok(())
    }

    pub fn new(
        id: Arc<SourceID>,
        attributes: WindowAttributes,
        child: Box<dyn ComponentWrap<dyn crate::layout::base::Empty>>,
    ) -> Self {
        super::set_children(Self {
            id,
            attributes,
            child,
        })
    }

    fn resize(size: PhysicalSize<u32>, state: &mut WindowState) {
        state.config.width = size.width;
        state.config.height = size.height;
        state.surface.configure(&state.driver.device, &state.config);
    }

    #[allow(clippy::result_unit_err)]
    pub fn on_window_event(
        id: Arc<SourceID>,
        rtree: Weak<rtree::Node>,
        event: WindowEvent,
        manager: &mut StateManager,
        driver: std::sync::Weak<graphics::Driver>,
    ) -> Result<(), ()> {
        let state: &mut WindowStateMachine = manager.get_mut(&id).map_err(|_| ())?;
        let window = state.state.as_mut().unwrap();
        let dpi = window.dpi;
        let inner = window.window.clone();
        match event {
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                window.dpi = Vec2::broadcast(scale_factor as f32);
                window.window.request_redraw();
                Ok(())
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
                Ok(())
            }
            WindowEvent::Resized(new_size) => {
                // Resize events can sometimes give empty sizes if the window is minimized
                if new_size.height > 0 && new_size.width > 0 {
                    Self::resize(new_size, window);
                }
                // On macos the window needs to be redrawn manually after resizing
                window.window.request_redraw();
                Ok(())
            }
            WindowEvent::CloseRequested => {
                // If this returns Some(data), the close request will be ignored
                Err(())
            }
            WindowEvent::RedrawRequested => {
                panic!("Don't process this with on_window_event");
            }
            WindowEvent::Focused(acquired) => {
                // When the window loses or gains focus, we send a focus event to our children that had
                // focus, but we don't forget or change which children have focus.
                let evt = RawEvent::Focus {
                    acquired,
                    window: window.window.clone(),
                };

                // We have to collect this map so we aren't borrowing manager twice
                let nodes: SmallVec<[Weak<Node>; 4]> = window.nodes(WindowNodeTrack::Focus);

                for node in nodes.iter().filter_map(|x| x.upgrade()) {
                    let _ = node.inject_event(
                        &evt,
                        evt.kind(),
                        dpi,
                        Vec2::zero(),
                        id.clone(),
                        &driver,
                        manager,
                    );
                }
                Ok(())
            }
            _ => {
                let e = match event {
                    WindowEvent::Ime(winit::event::Ime::Commit(s)) => RawEvent::Key {
                        device_id: DeviceId::dummy(), // TODO: No way to derive originating keyboard from IME event????
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
                            pos: window.last_mouse,
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

                        // If the cursor enters and no buttons are pressed, ensure any captured state is reset
                        if window.all_buttons == 0 {
                            // No need to inject a mousemove event here, as MouseOn is already being sent.
                            window.remove(WindowNodeTrack::Capture, &device_id);
                        }
                        RawEvent::MouseOn {
                            device_id,
                            pos: window.last_mouse,
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
                            pos: window.last_mouse,
                            delta: Vec2::new(x, y),
                            pixels: false,
                        },
                        winit::event::MouseScrollDelta::PixelDelta(physical_position) => {
                            RawEvent::MouseScroll {
                                device_id,
                                state: phase.into(),
                                pos: window.last_mouse,
                                delta: Vec2::new(
                                    physical_position.x as f32,
                                    physical_position.y as f32,
                                ),
                                pixels: true,
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
                            pos: window.last_mouse,
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
                        pos: ultraviolet::Vec3::new(
                            touch.location.x as f32,
                            touch.location.y as f32,
                            0.0,
                        ),
                        index: touch.id,
                        angle: Default::default(),
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
                    _ => return Err(()),
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
                    RawEvent::Drag => Err(()),
                    RawEvent::Focus { .. } => Err(()),
                    RawEvent::JoyAxis { device_id: _, .. }
                    | RawEvent::JoyButton { device_id: _, .. }
                    | RawEvent::JoyOrientation { device_id: _, .. }
                    | RawEvent::Key { device_id: _, .. } => {
                        // We have to collect this map so we aren't borrowing manager twice
                        let nodes: SmallVec<[Weak<Node>; 4]> = window.nodes(WindowNodeTrack::Focus);

                        // Currently, we always duplicate key/joystick events to all focused elements. Later, we may map specific
                        // keyboards to specific mouse input device IDs. We use a fold instead of any() to avoid short-circuiting.
                        if nodes.iter().fold(false, |ok, node| {
                            ok | node
                                .upgrade()
                                .map(|n| {
                                    n.inject_event(
                                        &e,
                                        e.kind(),
                                        dpi,
                                        Vec2::zero(),
                                        id.clone(),
                                        &driver,
                                        manager,
                                    )
                                    .is_ok()
                                })
                                .unwrap_or(false)
                        }) {
                            Ok(())
                        } else {
                            Err(())
                        }
                    }
                    RawEvent::MouseOff { .. } => {
                        // We have to collect this map so we aren't borrowing manager twice
                        let nodes: SmallVec<[Weak<Node>; 4]> = window.drain(WindowNodeTrack::Hover);

                        // Send a mouseoff event to all captures, but don't drain the captures so we have a chance to recover.
                        let capture_nodes: SmallVec<[Weak<Node>; 4]> =
                            window.nodes(WindowNodeTrack::Capture);

                        for node in nodes.iter().filter_map(|x| x.upgrade()) {
                            let _ = node.inject_event(
                                &e,
                                e.kind(),
                                dpi,
                                Vec2::zero(),
                                id.clone(),
                                &driver,
                                manager,
                            );
                        }

                        // While we could recover the offset here, we don't so we can be consistent about MouseOff not having offset.
                        for node in capture_nodes.iter().filter_map(|x| x.upgrade()) {
                            let _ = node.inject_event(
                                &e,
                                e.kind(),
                                dpi,
                                Vec2::zero(),
                                id.clone(),
                                &driver,
                                manager,
                            );
                        }

                        Ok(())
                    }
                    RawEvent::Mouse {
                        device_id,
                        pos: PhysicalPosition { x, y },
                        ..
                    }
                    | RawEvent::MouseOn {
                        device_id,
                        pos: PhysicalPosition { x, y },
                        ..
                    }
                    | RawEvent::MouseMove {
                        device_id,
                        pos: PhysicalPosition { x, y },
                        ..
                    }
                    | RawEvent::Drop {
                        device_id,
                        pos: PhysicalPosition { x, y },
                        ..
                    }
                    | RawEvent::MouseScroll {
                        device_id,
                        pos: PhysicalPosition { x, y },
                        ..
                    }
                    | RawEvent::Touch {
                        device_id,
                        pos: ultraviolet::Vec3 { x, y, z: _ },
                        ..
                    } => {
                        if let Some(node) = window
                            .get(WindowNodeTrack::Capture, &device_id)
                            .and_then(|x| x.upgrade())
                        {
                            let offset = Node::offset(node.clone());
                            return node
                                .clone()
                                .inject_event(
                                    &e,
                                    e.kind(),
                                    dpi,
                                    offset,
                                    id.clone(),
                                    &driver,
                                    manager,
                                )
                                .map(|_| ())
                                .map_err(|_| ());
                        }

                        if let Some(rt) = rtree.upgrade() {
                            rt.process(
                                &e,
                                e.kind(),
                                Vec2::new(x, y),
                                Vec2::zero(),
                                dpi,
                                &driver,
                                manager,
                                id.clone(),
                            )
                        } else {
                            Err(())
                        }
                    }
                };

                if r.is_err() {
                    match e {
                        // If everything rejected the mousemove, remove hover from all elements
                        RawEvent::MouseMove {
                            device_id,
                            modifiers,
                            all_buttons,
                            ..
                        } => {
                            // We reborrow everything here or rust gets upset
                            let state: &mut WindowStateMachine =
                                manager.get_mut(&id).map_err(|_| ())?;
                            let window = state.state.as_mut().unwrap();
                            let evt = RawEvent::MouseOff {
                                device_id,
                                modifiers,
                                all_buttons,
                            };

                            // Drain() holds a reference, so we still have to collect these to avoid borrowing manager twice
                            let nodes: SmallVec<[Weak<Node>; 4]> =
                                window.drain(WindowNodeTrack::Hover);

                            for node in nodes.iter().filter_map(|x| x.upgrade()) {
                                let _ = node.inject_event(
                                    &evt,
                                    evt.kind(),
                                    dpi,
                                    Vec2::zero(),
                                    id.clone(),
                                    &driver,
                                    manager,
                                );
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
                            // We reborrow everything here or rust gets upset
                            let state: &mut WindowStateMachine =
                                manager.get_mut(&id).map_err(|_| ())?;
                            let window = state.state.as_mut().unwrap();
                            let evt = RawEvent::Focus {
                                acquired: false,
                                window: window.window.clone(),
                            };

                            // Drain() holds a reference, so we still have to collect these to avoid borrowing manager twice
                            let nodes: SmallVec<[Weak<Node>; 4]> =
                                window.drain(WindowNodeTrack::Focus);

                            for node in nodes.iter().filter_map(|x| x.upgrade()) {
                                let _ = node.inject_event(
                                    &evt,
                                    evt.kind(),
                                    dpi,
                                    Vec2::zero(),
                                    id.clone(),
                                    &driver,
                                    manager,
                                );
                            }
                        }
                        _ => (),
                    }
                }

                // After finishing all processing, if we were processing a mousemove or mouseon event, update our cursor
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
