use super::Component;
use super::ComponentFrom;
use crate::layout::root::Root;
use crate::layout::EventList;
use crate::rtree;
use crate::DriverState;
use crate::RenderInstruction;
use crate::{layout, AbsRect};
use derive_where::derive_where;
use eyre::OptionExt;
use std::rc::{Rc, Weak};
use std::sync::Arc;
use ultraviolet::Vec2;
use wgpu::{Adapter, Device};
use winit::dpi::PhysicalSize;
use winit::window::WindowId;
use winit::{event::WindowEvent, event_loop::EventLoopWindowTarget};

#[derive_where(Clone)]
pub struct Window<AppData: 'static> {
    pub id: WindowId,
    pub window: Arc<winit::window::Window>,
    surface: Arc<wgpu::Surface<'static>>,
    config: wgpu::SurfaceConfiguration,
    child: Box<ComponentFrom<AppData, Root>>,
    driver: Arc<crate::DriverState>,
    layout_tree: Option<Box<dyn crate::layout::Layout<(), AppData>>>,
    staging: Option<Box<dyn layout::Staged<AppData>>>,
    rtree: Weak<rtree::Node<AppData>>,
    draw: im::Vector<RenderInstruction>,
}

impl<AppData> Component<AppData, ()> for Window<AppData> {
    fn layout(
        &self,
        data: &AppData,
        driver: &DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn crate::layout::Layout<(), AppData>> {
        let size = self.window.inner_size();
        Box::new(layout::Node::<AppData, Root, ()> {
            props: Root {
                area: AbsRect {
                    topleft: Default::default(),
                    bottomright: Vec2 {
                        x: size.width as f32,
                        y: size.height as f32,
                    },
                },
            },
            imposed: (),
            children: self.child.layout(data, driver, config),
            events: std::rc::Weak::<EventList<AppData>>::new(),
            renderable: None,
        })
    }
}

impl<'a, AppData> Window<AppData> {
    pub fn new<T: 'static>(
        window: Arc<winit::window::Window>,
        surface: wgpu::Surface<'static>,
        child: Box<ComponentFrom<AppData, Root>>,
        driver: Arc<crate::DriverState>,
    ) -> eyre::Result<Self> {
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&driver.adapter, size.width, size.height)
            .ok_or_eyre("Failed to find a default configuration")?;
        //let view_format = config.format.add_srgb_suffix();
        let view_format = config.format.remove_srgb_suffix();
        config.format = view_format;
        config.view_formats.push(view_format);

        surface.configure(&driver.device, &config);

        Ok(Self {
            id: window.id(),
            surface: Arc::new(surface),
            window,
            child,
            driver,
            config,
            layout_tree: None,
            staging: None,
            rtree: std::rc::Weak::<rtree::Node<AppData>>::new(),
            draw: im::Vector::new(),
        })
    }

    pub fn layout_all(&mut self, data: &AppData) {
        self.layout_tree = Some(self.layout(data, &self.driver, &self.config));
    }

    pub fn stage_all(&mut self) {
        if let Some(layout) = self.layout_tree.as_ref() {
            self.staging = Some(layout.stage(Default::default(), &self.driver.queue));
        }
    }

    pub fn render_all(&mut self) {
        if let Some(staging) = self.staging.as_ref() {
            self.draw = staging.render();
        }
    }

    pub fn set_position<P: Into<winit::dpi::Position>>(&self, position: P) {
        self.window.set_outer_position(position)
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.driver.device, &self.config);
    }

    pub fn on_event(&mut self, event: winit::event::WindowEvent) -> bool {
        match event {
            WindowEvent::Resized(new_size) => {
                // Resize events can sometimes give empty sizes if the window is minimized
                if new_size.height > 0 && new_size.width > 0 {
                    self.resize(new_size);
                }
                // On macos the window needs to be redrawn manually after resizing
                self.window.request_redraw();
            }
            WindowEvent::CloseRequested => {
                // If this returns false, the close request will be ignored
                return true;
            }
            WindowEvent::RedrawRequested => {
                let frame = self.surface.get_current_texture().unwrap();
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    self.driver
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
                        self.config.width as f32,
                        self.config.height as f32,
                        0.0,
                        1.0,
                    );

                    for f in self.draw.iter() {
                        if let Some(g) = f {
                            g(&mut pass);
                        }
                    }
                }

                self.driver.queue.submit(Some(encoder.finish()));
                frame.present();
            }
            /*WindowEvent::AxisMotion {
                device_id,
                axis,
                value,
            } => {}
            WindowEvent::CursorEntered { device_id } => {}
            WindowEvent::CursorLeft { device_id } => {}
            WindowEvent::CursorMoved {
                device_id,
                position,
            } => {}
            WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic,
            } => {}
            WindowEvent::MouseWheel {
                device_id,
                delta,
                phase,
            } => {}*/
            _ => {}
        }
        true
    }
}
