use feather_ui::layout::basic;
use feather_ui::layout::basic::Basic;
use feather_ui::layout::root;
use feather_ui::outline::button::Button;
use feather_ui::outline::region::Region;
use feather_ui::outline::round_rect::RoundRect;
use feather_ui::outline::text::Text;
use feather_ui::outline::window::Window;
use feather_ui::outline::OutlineFrom;
use feather_ui::persist::FnPersist;
use feather_ui::AbsRect;
use feather_ui::App;
use feather_ui::DriverState;
use feather_ui::WindowCollection;
use std::sync::Arc;
use ultraviolet::Vec4;

#[derive(PartialEq, Clone, Debug)]
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

struct BasicApp {
    window: Arc<winit::window::Window>,
    surface: Arc<wgpu::Surface<'static>>,
}

impl FnPersist<CounterState, WindowCollection<CounterState>> for BasicApp {
    type Store = (CounterState, WindowCollection<CounterState>);

    fn init(&self) -> Self::Store {
        (CounterState { count: -1 }, WindowCollection::new())
    }
    fn call(
        &self,
        mut store: Self::Store,
        args: &CounterState,
    ) -> (Self::Store, WindowCollection<CounterState>) {
        if store.0 != *args {
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
                    text: format!("Clicks: {}", args.count),
                    font_size: 30.0,
                    line_height: 42.0,
                    ..Default::default()
                };

                let mut children: im::Vector<Option<Box<OutlineFrom<CounterState, Basic>>>> =
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

            let mut children: im::Vector<Option<Box<OutlineFrom<CounterState, Basic>>>> =
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
            let window_id = self.window.id();
            let window = Window::new::<CounterState>(
                self.window.clone(),
                self.surface.clone(),
                Box::new(region),
                self.driver.clone(),
            )
            .unwrap();

            store.1 = WindowCollection::new();
            store.1.insert(window_id, window);
            store.0 = args.clone();
        }
        let windows = store.1.clone();
        (store, windows)
    }
}

fn main() {
    let (mut app, event_loop): (
        App<CounterState, BasicApp>,
        winit::event_loop::EventLoop<()>,
    ) = App::new(
        CounterState { count: 0 },
        move |app: &mut App<CounterState, BasicApp>,
              event_loop: &winit::event_loop::ActiveEventLoop| {
            let window = Arc::new(
                event_loop
                    .create_window(
                        winit::window::Window::default_attributes()
                            .with_title("basic-rs")
                            .with_resizable(true),
                    )
                    .unwrap(),
            );

            let surface = Arc::new(app.instance.create_surface(window.clone()).unwrap());

            let driver = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(app.create_driver(&surface))
                .unwrap();

            BasicApp {
                window,
                surface,
                driver,
            }
        },
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
