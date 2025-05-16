// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::{Outline, StateMachine};
use crate::input::{ModifierKeys, MouseMoveState, MouseState, RawEvent};
use crate::layout::root;
use crate::outline::OutlineWrap;
use crate::{
    layout, pixel_to_vec, rtree, AbsDim, DriverState, FnPersist, RenderInstruction, SourceID,
    StateManager,
};
use alloc::sync::Arc;
use core::f32;
use eyre::{OptionExt, Result};
use std::rc::{Rc, Weak};
use ultraviolet::Vec2;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowAttributes;

/// Holds our internal mutable state for this window
pub(crate) struct WindowState {
    pub surface: wgpu::Surface<'static>, // Ensure surface get dropped before window
    pub window: Arc<winit::window::Window>,
    pub config: wgpu::SurfaceConfiguration,
    all_buttons: u8,
    modifiers: u8,
    last_mouse: PhysicalPosition<f32>,
    pub dpi: Vec2,
    pub driver: Rc<DriverState>,
    pub draw: im::Vector<RenderInstruction>,
}

pub(crate) type WindowStateMachine = StateMachine<(), WindowState, 0, 0>;

#[derive(Clone)]
pub struct Window {
    pub id: Rc<SourceID>,
    attributes: WindowAttributes,
    child: Box<dyn OutlineWrap<<dyn root::Prop as crate::outline::Desc>::Child>>,
}

impl Outline<AbsDim> for Window {
    fn layout(
        &self,
        manager: &crate::StateManager,
        _: &DriverState,
        _: crate::Vec2,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn crate::layout::Layout<AbsDim>> {
        let inner = manager
            .get::<StateMachine<(), WindowState, 0, 0>>(&self.id)
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
            children: self
                .child
                .layout(manager, &driver, inner.dpi, &inner.config),
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }

    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        Err(crate::Error::UnhandledEvent)
    }

    fn init_all(&self, _: &mut crate::StateManager) -> eyre::Result<()> {
        Err(eyre::eyre!(
            "Cannot use normal init_all function for top-level windows"
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
        driver: &mut std::rc::Weak<DriverState>,
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
            driver.format.replace(Some(view_format)).map_or_else(
                || Ok(()),
                |format| {
                    Err(eyre::eyre!(
                        "driver.format is already configured: {:?}",
                        format
                    ))
                },
            )?;
            let mut windowstate = WindowState {
                modifiers: 0,
                all_buttons: 0,
                last_mouse: PhysicalPosition::new(f32::NAN, f32::NAN),
                config,
                dpi: Vec2::broadcast(window.scale_factor() as f32),
                surface,
                window,
                driver,
                draw: im::Vector::new(),
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

        manager.init_outline(self.child.as_ref())?;
        Ok(())
    }

    pub fn new(
        id: Rc<SourceID>,
        attributes: WindowAttributes,
        child: Box<dyn OutlineWrap<dyn crate::layout::base::Empty>>,
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

        let text_system = state.driver.text().expect("driver.text not initialized");
        text_system.borrow_mut().viewport.update(
            &state.driver.queue,
            glyphon::Resolution {
                width: state.config.width,
                height: state.config.height,
            },
        );
    }

    #[allow(clippy::result_unit_err)]
    pub fn on_window_event(
        id: Rc<SourceID>,
        rtree: Weak<rtree::Node>,
        event: WindowEvent,
        manager: &mut StateManager,
    ) -> Result<(), ()> {
        let state: &mut WindowStateMachine = manager.get_mut(&id).map_err(|_| ())?;
        let window = state.state.as_mut().unwrap();
        match event {
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                inner_size_writer,
            } => {
                window.dpi = Vec2::broadcast(scale_factor as f32);
                window.window.request_redraw();
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
            }
            WindowEvent::Resized(new_size) => {
                // Resize events can sometimes give empty sizes if the window is minimized
                if new_size.height > 0 && new_size.width > 0 {
                    Self::resize(new_size, window);
                }
                // On macos the window needs to be redrawn manually after resizing
                window.window.request_redraw();
                return Ok(());
            }
            WindowEvent::CloseRequested => {
                // If this returns Some(data), the close request will be ignored
                return Err(());
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
                            label: Some("Window Outline"),
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
                return Ok(());
            }
            _ => {
                let e = match event {
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
                            state: MouseMoveState::Move,
                            pos: window.last_mouse,
                            all_buttons: window.all_buttons,
                            modifiers: window.modifiers,
                        }
                    }
                    WindowEvent::CursorEntered { device_id } => RawEvent::MouseMove {
                        device_id,
                        state: MouseMoveState::On,
                        pos: window.last_mouse,
                        all_buttons: window.all_buttons,
                        modifiers: window.modifiers,
                    },
                    WindowEvent::CursorLeft { device_id } => {
                        let e = RawEvent::MouseMove {
                            device_id,
                            state: MouseMoveState::Off,
                            pos: window.last_mouse,
                            all_buttons: window.all_buttons,
                            modifiers: window.modifiers,
                        };
                        window.last_mouse = PhysicalPosition::new(f32::NAN, f32::NAN);
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
                            pos: pixel_to_vec(window.last_mouse),
                            delta: Vec2::new(x, y),
                            pixels: false,
                        },
                        winit::event::MouseScrollDelta::PixelDelta(physical_position) => {
                            RawEvent::MouseScroll {
                                device_id,
                                state: phase.into(),
                                pos: pixel_to_vec(window.last_mouse),
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
                            window.all_buttons |= b as u8;
                        } else {
                            window.all_buttons &= !(b as u8);
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

                if let Some(rt) = rtree.upgrade() {
                    return rt.on_event(&e, window.dpi, manager);
                }
            }
        }

        Err(())
    }
}
