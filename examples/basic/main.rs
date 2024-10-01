use feather_ui::component::window::Window;
use winit::event::Event;
use winit::event::WindowEvent;
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

struct CounterState {
    count: i32,
}

fn counter_increment(st: CounterState) -> CounterState {
    CounterState {
        count: st.count + 1,
    }
}

fn counter_decrement(st: CounterState) -> CounterState {
    CounterState {
        count: st.count - 1,
    }
}

fn main() {
    let instance = wgpu::Instance::default();

    let event_loop = EventLoop::new().unwrap();
    let mut window = Window::new(
        &instance,
        WindowBuilder::new()
            .with_title(format!("test_blank!"))
            .with_inner_size(winit::dpi::PhysicalSize::new(600, 400)),
        &event_loop,
    );

    let (adapter, device, queue) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let adapter = window
                .get_adapter(&instance)
                .await
                .expect("Failed to find adapter");

            // Create the logical device and command queue
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::MemoryUsage,
                    },
                    None,
                )
                .await
                .expect("Failed to create device");

            (adapter, device, queue)
        });

    window.configure(&adapter, &device);

    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter);

            if let Event::WindowEvent { window_id, event } = event {
                if window_id == window.id {
                    match event {
                        WindowEvent::RedrawRequested => {
                            let frame = window.surface.get_current_texture().unwrap();
                            let view = frame
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            let mut encoder =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: None,
                                });
                            {
                                let rpass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: None,
                                        color_attachments: &[Some(
                                            wgpu::RenderPassColorAttachment {
                                                view: &view,
                                                resolve_target: None,
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                                        r: 0.4,
                                                        g: 0.5,
                                                        b: 0.6,
                                                        a: 0.0,
                                                    }),
                                                    store: wgpu::StoreOp::Store,
                                                },
                                            },
                                        )],
                                        depth_stencil_attachment: None,
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                    });
                            }

                            queue.submit(Some(encoder.finish()));
                            frame.present();
                        }
                        WindowEvent::CloseRequested => target.exit(),
                        _ => window.process_event(event, &device),
                    }
                }
            }
        })
        .unwrap();
}
