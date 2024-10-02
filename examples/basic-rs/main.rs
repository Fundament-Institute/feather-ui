use feather_ui::component::region::Region;
use feather_ui::component::round_rect::RoundRect;
use feather_ui::component::window::Window;
use feather_ui::component::ComponentFrom;
use feather_ui::layout::basic;
use feather_ui::layout::basic::Basic;
use feather_ui::layout::root;
use feather_ui::layout::root::Root;
use feather_ui::layout::Desc;
use feather_ui::App;
use feather_ui::UPoint;
use std::sync::Arc;
use ultraviolet::Mat4;
use ultraviolet::Vec2;
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

fn main() {
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

    let rect = RoundRect::<<Basic as Desc<()>>::Impose> {
        fill: Vec4::new(1.0, 1.0, 0.0, 1.0),
        corners: Vec4::broadcast(100.0),
        props: basic::Inherited {
            area: feather_ui::FILL_URECT,
            margin: Default::default(),
        },
        ..Default::default()
    };
    let mut children: im::Vector<Option<Box<ComponentFrom<(), Basic>>>> = im::Vector::new();
    children.push_back(Some(Box::new(rect)));

    let region = Region::new(
        root::Inherited {
            area: feather_ui::FILL_URECT,
        },
        Basic {
            padding: Default::default(),
            zindex: 0,
        },
        children,
    );
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
