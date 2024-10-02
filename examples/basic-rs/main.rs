use feather_ui::component::region::Region;
use feather_ui::component::round_rect::RoundRect;
use feather_ui::component::window::Window;
use feather_ui::layout::root::Root;
use feather_ui::layout::Desc;
use feather_ui::App;
use std::sync::Arc;
use ultraviolet::Mat4;
use ultraviolet::Vec4;
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
use ultraviolet::Vec3;

fn viewport_transform(p: Vec3, x: f32, y: f32, w: f32, h: f32, f: f32, n: f32) -> Vec3 {
    Vec3 {
        x: (w / 2.0) * p.x + x + (w / 2.0),
        y: (h / 2.0) * p.y + y + (h / 2.0),
        z: ((f - n) / 2.0) * p.z + ((f + n) / 2.0),
    }
}

fn NDC_perspective(p: Vec4) -> Vec3 {
    Vec3 {
        x: p.x / p.w,
        y: p.y / p.w,
        z: p.z / p.w,
    }
}

// Inverts a viewport transform into an orthographic projection matrix
fn viewport_ortho(x: f32, y: f32, w: f32, h: f32, n: f32, f: f32) -> Mat4 {
    let mut m = Mat4 {
        cols: [
            Vec4::new(2.0 / w, 0.0, 0.0, 0.0).into(),
            Vec4::new(0.0, 2.0 / h, 0.0, 0.0).into(),
            Vec4::new(0.0, 0.0, -2.0 / (f - n), 0.0).into(),
            Vec4::new(
                -(2.0 * x + w) / w,
                -(2.0 * y + h) / h,
                (f + n) / (f - n),
                1.0,
            )
            .into(),
        ],
    };
    m
}

fn viewport_proj(x: f32, y: f32, w: f32, h: f32, n: f32, f: f32) -> Mat4 {
    let mut m = Mat4 {
        cols: [
            Vec4::new(2.0 / w, 0.0, 0.0, 0.0).into(),
            Vec4::new(0.0, 2.0 / h, 0.0, 0.0).into(),
            Vec4::new(0.0, 0.0, -2.0 / (f - n), 0.0).into(),
            Vec4::new(
                -(2.0 * x + w) / w,
                -(2.0 * y + h) / h,
                (f + n) / (f - n),
                1.0,
            )
            .into(),
        ],
    };
    m
}

fn main() {
    let m = viewport_ortho(0.0, 600.0, 800.0, -600.0, 0.0, 1.0);
    let v = Vec4::new(200.0, 200.0, 1.0, 1.0);
    let v = m * v;
    println!("clip space: x: {}, y: {}, z: {}, w: {}", v.x, v.y, v.z, v.w);
    let p = NDC_perspective(v);
    let p = viewport_transform(p, 0.0, 600.0, 800.0, -600.0, 0.0, 1.0);

    println!("screen space: x: {}, y: {}, z: {}", p.x, p.y, p.z);

    let (mut app, event_loop) = App::<()>::new(()).unwrap();

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("basic-rs")
            .with_resizable(true)
            .build(&event_loop)
            .unwrap(),
    );

    let surface = app.instance.create_surface(window.clone()).unwrap();

    let driver = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(app.create_driver(&surface))
        .unwrap();

    let region = RoundRect::<<Root as Desc<()>>::Impose> {
        fill: Vec4::new(1.0, 1.0, 0.0, 1.0),
        ..Default::default()
    };
    let window_id = window.id();
    let window = Window::new::<()>(window, surface, Box::new(region), driver).unwrap();

    app.windows.insert(window_id, window);

    event_loop
        .run(
            move |e: winit::event::Event<()>,
                  target: &winit::event_loop::EventLoopWindowTarget<()>| {
                if let winit::event::Event::WindowEvent { window_id, event } = e.clone() {
                    if let winit::event::WindowEvent::MouseInput {
                        device_id,
                        state,
                        button,
                    } = event
                    {
                        app.windows.values().for_each(|w| w.window.request_redraw())
                    }
                }
                app.event(e, target)
            },
        )
        .unwrap();
}
