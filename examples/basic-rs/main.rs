use feather_ui::component::button::Button;
use feather_ui::component::region::Region;
use feather_ui::component::round_rect::RoundRect;
use feather_ui::component::text::Text;
use feather_ui::component::window::Window;
use feather_ui::component::ComponentFrom;
use feather_ui::layout::basic;
use feather_ui::layout::basic::Basic;
use feather_ui::layout::root;
use feather_ui::layout::Desc;
use feather_ui::AbsRect;
use feather_ui::App;
use std::default;
use std::sync::Arc;
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
    let (mut app, event_loop) = App::<CounterState>::new(CounterState { count: 0 }, |data| {
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

        let button = {
            let rect = RoundRect::<basic::Inherited> {
                fill: Vec4::new(0.2, 0.7, 0.4, 1.0),
                corners: Vec4::broadcast(10.0),
                props: basic::Inherited {
                    area: feather_ui::FILL_URECT,
                    margin: Default::default(),
                },
                ..Default::default()
            };

            let text = Text::<basic::Inherited> {
                props: basic::Inherited {
                    area: feather_ui::FILL_URECT,
                    margin: Default::default(),
                },
                text: "Button".to_string(),
                font_size: 30.0,
                line_height: 42.0,
                ..Default::default()
            };

            let mut children: im::Vector<Option<Box<ComponentFrom<CounterState, Basic>>>> =
                im::Vector::new();
            //children.push_back(Some(Box::new(rect)));
            children.push_back(Some(Box::new(text)));

            let onclick = Box::new(|_, _, mut appdata: CounterState| {
                appdata.count += 1;
                Ok(appdata)
            });
            Button::<CounterState, basic::Inherited>::new(
                basic::Inherited {
                    area: feather_ui::FILL_URECT,
                    margin: Default::default(),
                },
                Default::default(),
                onclick,
                children,
            )
        };

        let mut children: im::Vector<Option<Box<ComponentFrom<CounterState, Basic>>>> =
            im::Vector::new();
        children.push_back(Some(Box::new(button)));

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
        let window =
            Window::new::<CounterState>(window, surface, Box::new(region), driver).unwrap();
    })
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
