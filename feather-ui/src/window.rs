use std::sync::Arc;
use wgpu::{Adapter, Device};
use winit::dpi::PhysicalSize;
use winit::window::WindowId;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoopWindowTarget,
    platform::windows::WindowExtWindows,
};

pub struct Window {
    pub id: WindowId,
    window: Arc<winit::window::Window>,
    pub surface: wgpu::Surface<'static>,
    config: Option<wgpu::SurfaceConfiguration>,
}
impl Window {
    pub fn new<T: 'static>(
        instance: &wgpu::Instance,
        builder: winit::window::WindowBuilder,
        event_loop: &EventLoopWindowTarget<T>,
    ) -> Self {
        let window = Arc::new(builder.build(&event_loop).unwrap());
        let surface = instance.create_surface(window.clone()).unwrap();

        Self {
            id: window.id(),
            window,
            surface,
            config: None,
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
