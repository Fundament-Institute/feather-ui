// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use core::f32;
use feather_ui::gen_id;
use feather_ui::layout::prop;
use feather_ui::layout::root;
use feather_ui::layout::simple;
use feather_ui::outline::button::Button;
use feather_ui::outline::mouse_area;
use feather_ui::outline::text::Text;
use feather_ui::outline::window::Window;
use feather_ui::outline::OutlineFrom;
use feather_ui::persist::FnPersist;
use feather_ui::AbsRect;
use feather_ui::App;
use feather_ui::Slot;
use feather_ui::SourceID;
use feather_ui::URect;
use feather_ui::FILL_URECT;
use std::rc::Rc;
use ultraviolet::Vec4;

#[derive(PartialEq, Clone, Debug)]
struct CounterState {
    count: i32,
}

struct SimpleData {
    area: feather_ui::URect,
    margin: feather_ui::URect,
    anchor: feather_ui::UPoint,
    limits: feather_ui::URect,
    zindex: i32,
}

impl prop::Empty for SimpleData {}

impl prop::Area for SimpleData {
    fn area(&self) -> &feather_ui::URect {
        &self.area
    }
}
impl prop::Margin for SimpleData {
    fn margin(&self) -> &feather_ui::URect {
        &self.margin
    }
}
impl prop::Anchor for SimpleData {
    fn anchor(&self) -> &feather_ui::UPoint {
        &self.anchor
    }
}
impl prop::Limits for SimpleData {
    fn limits(&self) -> &feather_ui::URect {
        &self.limits
    }
}

impl prop::ZIndex for SimpleData {
    fn zindex(&self) -> i32 {
        self.zindex
    }
}
impl simple::Prop for SimpleData {}
impl root::Inherited for SimpleData {}

struct BasicApp {}

impl FnPersist<CounterState, im::HashMap<Rc<SourceID>, Option<Window>>> for BasicApp {
    type Store = (CounterState, im::HashMap<Rc<SourceID>, Option<Window>>);

    fn init(&self) -> Self::Store {
        (CounterState { count: -1 }, im::HashMap::new())
    }
    fn call(
        &self,
        mut store: Self::Store,
        args: &CounterState,
    ) -> (Self::Store, im::HashMap<Rc<SourceID>, Option<Window>>) {
        if store.0 != *args {
            let button = {
                let text = Text::<URect> {
                    id: gen_id!().into(),
                    props: Rc::new(FILL_URECT),
                    text: format!("Clicks: {}", args.count),
                    font_size: 30.0,
                    line_height: 42.0,
                    ..Default::default()
                };

                let mut children: im::Vector<Option<Box<OutlineFrom<dyn simple::Prop>>>> =
                    im::Vector::new();
                children.push_back(Some(Box::new(text)));

                /*let rect = RoundRect::<()> {
                    id: gen_id!().into(),
                    fill: Vec4::new(0.2, 0.7, 0.4, 1.0),
                    corners: Vec4::broadcast(10.0),
                    props: (),
                    rect: feather_ui::FILL_URECT,
                    ..Default::default()
                };
                children.push_back(Some(Box::new(rect)));*/

                Button::<SimpleData>::new(
                    gen_id!().into(),
                    SimpleData {
                        area: feather_ui::URect {
                            topleft: feather_ui::UPoint {
                                abs: ultraviolet::Vec2 { x: 0.0, y: 0.0 },
                                rel: feather_ui::RelPoint { x: 0.0, y: 0.0 },
                            },
                            bottomright: feather_ui::UPoint {
                                abs: ultraviolet::Vec2 {
                                    x: f32::INFINITY,
                                    y: 0.0,
                                },
                                rel: feather_ui::RelPoint { x: 0.0, y: 1.0 },
                            },
                        },
                        margin: Default::default(),
                        limits: feather_ui::DEFAULT_LIMITS,
                        anchor: Default::default(),
                        zindex: 0,
                    },
                    Slot(feather_ui::APP_SOURCE_ID.into(), 0),
                    children,
                )
            };

            //let mut children: im::Vector<Option<Box<OutlineFrom<Basic>>>> = im::Vector::new();
            //children.push_back(Some(Box::new(button)));

            /*let region = Region {
                id: gen_id!().into(),
                props: root::Inherited {
                    area: AbsRect::new(90.0, 90.0, f32::INFINITY, 200.0).into(),
                },
                basic: Basic {
                    padding: Default::default(),
                    zindex: 0,
                },
                children,
            };*/
            let window = Window::new(
                gen_id!().into(),
                winit::window::Window::default_attributes()
                    .with_title("basic-rs")
                    .with_resizable(true),
                Box::new(button),
            );

            store.1 = im::HashMap::new();
            store.1.insert(window.id.clone(), Some(window));
            store.0 = args.clone();
        }
        let windows = store.1.clone();
        (store, windows)
    }
}

use feather_ui::WrapEventEx;

fn main() {
    let onclick = Box::new(
        |_: mouse_area::MouseAreaEvent,
         mut appdata: CounterState|
         -> Result<CounterState, CounterState> {
            {
                appdata.count += 1;
                Ok(appdata)
            }
        }
        .wrap(),
    );

    let (mut app, event_loop): (
        App<CounterState, BasicApp>,
        winit::event_loop::EventLoop<()>,
    ) = App::new(CounterState { count: 0 }, vec![onclick], BasicApp {}).unwrap();

    event_loop.run_app(&mut app).unwrap();
}
