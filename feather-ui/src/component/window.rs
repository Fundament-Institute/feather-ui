// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::{Component, StateMachine};
use crate::component::ComponentWrap;
use crate::input::{ModifierKeys, MouseState, RawEvent};
use crate::layout::root;
use crate::rtree::Node;
use crate::{
    AbsDim, DriverState, FnPersist, RenderInstruction, SourceID, StateMachineChild, StateManager,
    layout, rtree,
};
use alloc::sync::Arc;
use core::f32;
use eyre::{OptionExt, Result};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use ultraviolet::Vec2;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{DeviceId, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{CursorIcon, WindowAttributes};

/// Holds our internal mutable state for this window
pub(crate) struct WindowState {
    pub surface: wgpu::Surface<'static>, // Ensure surface get dropped before window
    pub window: Arc<winit::window::Window>,
    pub config: wgpu::SurfaceConfiguration,
    all_buttons: u16,
    modifiers: u8,
    last_mouse: PhysicalPosition<f32>,
    pub dpi: Vec2,
    pub driver: Arc<DriverState>,
    pub draw: im::Vector<RenderInstruction>,
    pub viewport: Rc<glyphon::Viewport>,
    pub atlas: Rc<RefCell<glyphon::TextAtlas>>,
    pub focus: HashMap<DeviceId, RcNode>,
    pub hover: HashMap<DeviceId, RcNode>,
    pub capture: HashMap<DeviceId, Vec<RcNode>>,
}

pub(crate) struct RcNode(pub(crate) Rc<Node>);

impl PartialEq for RcNode {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
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
            && self.focus == other.focus
            && self.dpi == other.dpi
            && Arc::ptr_eq(&self.driver, &other.driver)
    }
}

pub(crate) type WindowStateMachine = StateMachine<(), WindowState, 0, 0>;

#[derive(Clone)]
pub struct Window {
    pub id: Rc<SourceID>,
    attributes: WindowAttributes,
    child: Box<dyn ComponentWrap<<dyn root::Prop as crate::component::Desc>::Child>>,
}

impl Component<AbsDim> for Window {
    fn layout(
        &self,
        manager: &crate::StateManager,
        _: &DriverState,
        _: &Rc<SourceID>,
        _: &wgpu::SurfaceConfiguration,
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
            children: self.child.layout(manager, &driver, &self.id, &inner.config),
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}

impl StateMachineChild for Window {
    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        Err(crate::Error::UnhandledEvent)
    }

    fn apply_children(
        &self,
        _: &mut dyn FnMut(&dyn StateMachineChild) -> eyre::Result<()>,
    ) -> eyre::Result<()> {
        Err(eyre::eyre!(
            "Cannot use normal apply_children function for top-level windows"
        ))
    }

    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }
}

impl Window {
    pub(crate) fn init_custom<
        AppData: 'static + PartialEq,
        O: FnPersist<AppData, im::HashMap<Rc<SourceID>, Option<Window>>>,
    >(
        &self,
        manager: &mut StateManager,
        driver: &mut alloc::sync::Weak<DriverState>,
        instance: &wgpu::Instance,
        event_loop: &ActiveEventLoop,
    ) -> Result<()> {
        if manager.get::<WindowStateMachine>(&self.id).is_err() {
            let attributes = self.attributes.clone();

            let window = Arc::new(event_loop.create_window(attributes)?);

            let surface: wgpu::Surface<'static> = instance.create_surface(window.clone())?;

            let driver = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(crate::App::<AppData, O>::create_driver(
                    driver, instance, &surface,
                ))?;

            let size = window.inner_size();
            let mut config = surface
                .get_default_config(&driver.adapter, size.width, size.height)
                .ok_or_eyre("Failed to find a default configuration")?;
            // let view_format = config.format.add_srgb_suffix();
            let view_format = config.format.remove_srgb_suffix();
            config.format = view_format;
            config.view_formats.push(view_format);
            surface.configure(&driver.device, &config);

            let cache = glyphon::Cache::new(&driver.device);
            let atlas = glyphon::TextAtlas::new(&driver.device, &driver.queue, &cache, view_format);
            let viewport = glyphon::Viewport::new(&driver.device, &cache);

            let mut windowstate = WindowState {
                modifiers: 0,
                all_buttons: 0,
                last_mouse: PhysicalPosition::new(f32::NAN, f32::NAN),
                config,
                dpi: Vec2::broadcast(window.scale_factor() as f32),
                focus: HashMap::new(),
                surface,
                window,
                driver: driver.clone(),
                draw: im::Vector::new(),
                atlas: Rc::new(RefCell::new(atlas)),
                viewport: Rc::new(viewport),
                hover: Default::default(),
                capture: Default::default(),
            };

            Window::resize(size, &mut windowstate);
            manager.init(
                self.id.clone(),
                Box::new(StateMachine::<(), WindowState, 0, 0> {
                    state: Some(windowstate),
                    input: [],
                    output: [],
                }),
            );
        }

        manager.init_child(self.child.as_ref())?;
        Ok(())
    }

    pub fn new(
        id: Rc<SourceID>,
        attributes: WindowAttributes,
        child: Box<dyn ComponentWrap<dyn crate::layout::base::Empty>>,
    ) -> Self {
        Self {
            id,
            attributes,
            child,
        }
    }

    fn resize(size: PhysicalSize<u32>, state: &mut WindowState) {
        state.config.width = size.width;
        state.config.height = size.height;
        state.surface.configure(&state.driver.device, &state.config);

        if let Some(viewport) = Rc::get_mut(&mut state.viewport) {
            viewport.update(
                &state.driver.queue,
                glyphon::Resolution {
                    width: state.config.width,
                    height: state.config.height,
                },
            );
        }
    }

    #[allow(clippy::result_unit_err)]
    pub fn on_window_event(
        id: Rc<SourceID>,
        rtree: Weak<rtree::Node>,
        event: WindowEvent,
        manager: &mut StateManager,
        driver: std::sync::Weak<DriverState>,
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
                let frame = window.surface.get_current_texture().unwrap();
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    window
                        .driver
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Window Component"),
                        });

                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Window Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.1,
                                    g: 0.2,
                                    b: 0.3,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    pass.set_viewport(
                        0.0,
                        0.0,
                        window.config.width as f32,
                        window.config.height as f32,
                        0.0,
                        1.0,
                    );

                    for f in window.draw.iter().flatten() {
                        f(&mut pass);
                    }
                }

                window.driver.queue.submit(Some(encoder.finish()));
                frame.present();
                Ok(())
            }
            WindowEvent::Focused(acquired) => {
                // When the window loses or gains focus, we send a focus event to our children that had
                // focus, but we don't forget or change which children have focus.
                let evt = RawEvent::Focus {
                    acquired,
                    window: window.window.clone(),
                };

                // We have to collect this map so we aren't borrowing manager twice
                let nodes: SmallVec<[RcNode; 4]> = window
                    .focus
                    .values()
                    .map(|node| RcNode(node.0.clone()))
                    .collect();

                for node in nodes {
                    let _ = node
                        .0
                        .inject_event(&evt, evt.kind(), dpi, Vec2::zero(), manager);
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
                            driver: driver.clone(),
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
                            let _ = window.capture.entry(device_id).and_modify(|x| x.clear());
                        }
                        RawEvent::MouseOn {
                            device_id,
                            pos: window.last_mouse,
                            all_buttons: window.all_buttons,
                            modifiers: window.modifiers,
                            driver: driver.clone(),
                        }
                    }
                    WindowEvent::CursorLeft { device_id } => {
                        window.last_mouse = PhysicalPosition::new(f32::NAN, f32::NAN);
                        let e = RawEvent::MouseOff {
                            device_id,
                            all_buttons: window.all_buttons,
                            modifiers: window.modifiers,
                            driver: driver.clone(),
                        };
                        e
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

                if let RawEvent::MouseMove { .. } = e {
                    if let Some(d) = driver.upgrade() {
                        *d.cursor.write() = CursorIcon::Default;
                    }
                }
                let r = match e {
                    RawEvent::Drag => Err(()),
                    RawEvent::Focus { .. } => Err(()),
                    RawEvent::JoyAxis { device_id: _, .. }
                    | RawEvent::JoyButton { device_id: _, .. }
                    | RawEvent::JoyOrientation { device_id: _, .. }
                    | RawEvent::Key { device_id: _, .. } => {
                        // We have to collect this map so we aren't borrowing manager twice
                        let nodes: SmallVec<[RcNode; 4]> = window
                            .focus
                            .values()
                            .map(|node| RcNode(node.0.clone()))
                            .collect();

                        // Currently, we always duplicate key/joystick events to all focused elements. Later, we may map specific
                        // keyboards to specific mouse input device IDs. We use a fold instead of any() to avoid short-circuiting.
                        if nodes.iter().fold(false, |ok, node| {
                            ok | node
                                .0
                                .inject_event(&e, e.kind(), dpi, Vec2::zero(), manager)
                                .is_ok()
                        }) {
                            Ok(())
                        } else {
                            Err(())
                        }
                    }
                    RawEvent::MouseOff { .. } => {
                        // We have to collect this map so we aren't borrowing manager twice
                        let nodes: SmallVec<[RcNode; 4]> =
                            window.hover.drain().map(|(_, v)| v).collect();

                        // Send a mouseoff event to all captures, but don't drain the captures so we have a chance to recover.
                        let capture_nodes: SmallVec<[RcNode; 4]> = window
                            .capture
                            .values()
                            .flat_map(|nodes| nodes.last().map(|x| RcNode(x.0.clone())))
                            .collect();

                        for node in nodes {
                            let _ = node
                                .0
                                .inject_event(&e, e.kind(), dpi, Vec2::zero(), manager);
                        }

                        // While we could recover the offset here, we don't so we can be consistent about MouseOff not having offset.
                        for node in capture_nodes {
                            let _ = node
                                .0
                                .inject_event(&e, e.kind(), dpi, Vec2::zero(), manager);
                        }

                        Ok(())
                    }
                    RawEvent::Mouse { device_id, pos, .. }
                    | RawEvent::MouseOn { device_id, pos, .. }
                    | RawEvent::MouseMove { device_id, pos, .. }
                    | RawEvent::Drop { device_id, pos, .. }
                    | RawEvent::MouseScroll { device_id, pos, .. } => {
                        if let Some((node, list)) =
                            window.capture.get(&device_id).and_then(|x| x.split_first())
                        {
                            let offset = Node::offset(list);
                            return node
                                .0
                                .clone()
                                .inject_event(&e, e.kind(), dpi, offset, manager);
                        }

                        if let Some(rt) = rtree.upgrade() {
                            rt.process(
                                &e,
                                e.kind(),
                                Vec2::new(pos.x, pos.y),
                                Vec2::zero(),
                                dpi,
                                manager,
                                id.clone(),
                                None,
                            )
                        } else {
                            Err(())
                        }
                    }
                    RawEvent::Touch { device_id, pos, .. } => {
                        if let Some((node, list)) =
                            window.capture.get(&device_id).and_then(|x| x.split_first())
                        {
                            let offset = Node::offset(list);
                            return node
                                .0
                                .clone()
                                .inject_event(&e, e.kind(), dpi, offset, manager);
                        }

                        if let Some(rt) = rtree.upgrade() {
                            rt.process(
                                &e,
                                e.kind(),
                                pos.xy(),
                                Vec2::zero(),
                                dpi,
                                manager,
                                id.clone(),
                                None,
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
                            ref driver,
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
                                driver: driver.clone(),
                            };

                            // Drain() holds a reference, so we still have to collect these to avoid borrowing manager twice
                            let nodes: SmallVec<[RcNode; 4]> =
                                window.hover.drain().map(|(_, v)| v).collect();

                            for node in nodes {
                                let _ = node.0.inject_event(
                                    &evt,
                                    evt.kind(),
                                    dpi,
                                    Vec2::zero(),
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
                            let nodes: SmallVec<[RcNode; 4]> =
                                window.focus.drain().map(|(_, v)| v).collect();

                            for node in nodes {
                                let _ = node.0.inject_event(
                                    &evt,
                                    evt.kind(),
                                    dpi,
                                    Vec2::zero(),
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
