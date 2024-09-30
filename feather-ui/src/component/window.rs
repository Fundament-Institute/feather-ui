use super::Component;
use super::ComponentFrom;
use crate::layout::root::Root;
use crate::layout::EventList;
use crate::{layout, AbsRect};
use derive_where::derive_where;
use std::sync::Arc;
use ultraviolet::Vec2;
use wgpu::{Adapter, Device};
use winit::dpi::PhysicalSize;
use winit::window::WindowId;
use winit::{event::WindowEvent, event_loop::EventLoopWindowTarget};

#[derive_where(Clone)]
pub struct Window<AppData> {
    pub id: WindowId,
    window: Arc<winit::window::Window>,
    pub surface: Arc<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    child: Box<ComponentFrom<AppData, Root>>,
}

impl<AppData> Component<AppData, ()> for Window<AppData> {
    fn layout(&self, data: &AppData) -> Box<dyn crate::layout::Layout<(), AppData>> {
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
            children: self.child.layout(data),
            events: std::rc::Weak::<EventList<AppData>>::new(),
            renderable: None,
        })
    }
}
impl<AppData> Window<AppData> {
    pub fn new<T: 'static>(
        instance: &wgpu::Instance,
        builder: winit::window::WindowBuilder,
        event_loop: &EventLoopWindowTarget<T>,
        child: Box<ComponentFrom<AppData, Root>>,
    ) -> Self {
        let window = Arc::new(builder.build(&event_loop).unwrap());
        let surface = Arc::new(instance.create_surface(window.clone()).unwrap());

        Self {
            id: window.id(),
            window,
            surface,
            config: None,
            child,
        }
    }

    pub fn set_position<P: Into<winit::dpi::Position>>(&self, position: P) {
        self.window.set_outer_position(position)
    }

    pub fn configure(&mut self, adapter: &Adapter, device: &Device) -> bool {
        let size = self.window.inner_size();
        self.config = self
            .surface
            .get_default_config(adapter, size.width, size.height);
        if let Some(config) = self.config.as_ref() {
            self.surface.configure(device, config);
            true
        } else {
            false
        }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>, device: &Device) -> bool {
        if let Some(config) = self.config.as_mut() {
            config.width = size.width;
            config.height = size.height;
            self.surface.configure(device, config);
            true
        } else {
            false
        }
    }
    pub async fn get_adapter(&self, instance: &wgpu::Instance) -> Option<Adapter> {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&self.surface),
                ..Default::default()
            })
            .await
    }

    pub fn process_event<'a>(&mut self, event: winit::event::WindowEvent, device: &Device) {
        match event {
            WindowEvent::Resized(new_size) => {
                // Recreate the swap chain with the new size
                self.resize(new_size, device);
                // On macos the window needs to be redrawn manually after resizing
                self.window.request_redraw();
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
    }
}
