// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use core::f32;
use feather_macro::*;
use feather_ui::gen_id;
use feather_ui::layout::base;
use feather_ui::layout::fixed;
use feather_ui::layout::flex;
use feather_ui::layout::leaf;
use feather_ui::layout::list;
use feather_ui::outline::button::Button;
use feather_ui::outline::flexbox::FlexBox;
use feather_ui::outline::listbox::ListBox;
use feather_ui::outline::mouse_area;
use feather_ui::outline::region::Region;
use feather_ui::outline::shape::Shape;
use feather_ui::outline::text::Text;
use feather_ui::outline::window::Window;
use feather_ui::outline::OutlineFrom;
use feather_ui::persist::FnPersist;
use feather_ui::AbsRect;
use feather_ui::App;
use feather_ui::RelRect;
use feather_ui::Slot;
use feather_ui::SourceID;
use feather_ui::URect;
use feather_ui::FILL_URECT;
use feather_ui::UNSIZED_AXIS;
use feather_ui::ZERO_POINT;
use std::rc::Rc;
use ultraviolet::Vec2;
use ultraviolet::Vec4;

#[derive(PartialEq, Clone, Debug)]
struct CounterState {
    count: i32,
}

#[derive(Default, Empty, Area, Anchor, ZIndex)]
struct FixedData {
    area: feather_ui::URect,
    anchor: feather_ui::UPoint,
    zindex: i32,
}

impl base::Padding for FixedData {}
impl base::Limits for FixedData {}
impl base::RLimits for FixedData {}
impl fixed::Prop for FixedData {}
impl fixed::Child for FixedData {}
impl leaf::Prop for FixedData {}
impl leaf::Padded for FixedData {}

#[derive(Default, Empty, Area, Direction, RLimits)]
struct ListData {
    area: feather_ui::URect,
    direction: feather_ui::RowDirection,
    rlimits: feather_ui::RelLimits,
}

impl base::Limits for ListData {}
impl list::Prop for ListData {}
impl fixed::Child for ListData {}

#[derive(Default, Empty, Area, Margin)]
struct ListChild {
    area: feather_ui::URect,
    margin: feather_ui::URect,
}

impl base::Padding for ListChild {}
impl base::Anchor for ListChild {}
impl base::Limits for ListChild {}
impl base::RLimits for ListChild {}
impl base::Order for ListChild {}
impl list::Child for ListChild {}
impl leaf::Prop for ListChild {}
impl leaf::Padded for ListChild {}

#[derive(Default, Empty, Area, FlexChild, Margin)]
struct FlexChild {
    area: feather_ui::URect,
    margin: feather_ui::URect,
    basis: f32,
    grow: f32,
    shrink: f32,
}

impl base::RLimits for FlexChild {}
impl base::Order for FlexChild {}
impl base::Anchor for FlexChild {}
impl base::Limits for FlexChild {}
impl base::Padding for FlexChild {}
impl leaf::Prop for FlexChild {}
impl leaf::Padded for FlexChild {}

#[derive(Default, Empty, Area)]
struct MinimalFlex {
    area: feather_ui::URect,
}

impl base::Obstacles for MinimalFlex {
    fn obstacles(&self) -> &[AbsRect] {
        &[]
    }
}
impl base::Direction for MinimalFlex {}
impl base::ZIndex for MinimalFlex {}
impl base::Limits for MinimalFlex {}
impl base::RLimits for MinimalFlex {}
impl fixed::Child for MinimalFlex {}

impl flex::Prop for MinimalFlex {
    fn wrap(&self) -> bool {
        true
    }

    fn justify(&self) -> flex::FlexJustify {
        flex::FlexJustify::Start
    }

    fn align(&self) -> flex::FlexJustify {
        flex::FlexJustify::Start
    }
}

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
                let text = Text::<FixedData> {
                    id: gen_id!().into(),
                    props: FixedData {
                        area: feather_ui::URect {
                            abs: AbsRect::new(10.0, 15.0, 10.0, 15.0),
                            rel: RelRect::new(0.0, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),
                        },
                        anchor: feather_ui::RelPoint(Vec2 { x: 0.0, y: 0.0 }).into(),
                        ..Default::default()
                    }
                    .into(),
                    text: format!("Boxes: {}", args.count),
                    font_size: 30.0,
                    line_height: 42.0,
                    ..Default::default()
                };

                let rect = Shape::<URect>::round_rect(
                    gen_id!().into(),
                    feather_ui::FILL_URECT.into(),
                    0.0,
                    0.0,
                    Vec4::broadcast(10.0),
                    Vec4::new(0.2, 0.7, 0.4, 1.0),
                    Vec4::zero(),
                );

                let mut children: im::Vector<Option<Box<OutlineFrom<dyn fixed::Prop>>>> =
                    im::Vector::new();
                children.push_back(Some(Box::new(text)));
                children.push_back(Some(Box::new(rect)));

                Button::<FixedData>::new(
                    gen_id!().into(),
                    FixedData {
                        area: feather_ui::URect {
                            abs: AbsRect::new(0.0, 20.0, 0.0, 0.0),
                            rel: RelRect::new(0.5, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),
                        },
                        anchor: feather_ui::RelPoint(Vec2 { x: 0.5, y: 0.0 }).into(),
                        zindex: 0,
                    },
                    Slot(feather_ui::APP_SOURCE_ID.into(), 0),
                    children,
                )
            };

            let rectlist = {
                let mut children: im::Vector<Option<Box<OutlineFrom<dyn list::Prop>>>> =
                    im::Vector::new();
                for i in 0..args.count {
                    children.push_back(Some(Box::new(Shape::<ListChild>::round_rect(
                        gen_id!().into(),
                        ListChild {
                            area: AbsRect::new(0.0, 0.0, 40.0, 40.0).into(),
                            margin: AbsRect::new(8.0, 8.0, 4.0, 4.0).into(),
                        }
                        .into(),
                        0.0,
                        0.0,
                        Vec4::broadcast(8.0),
                        Vec4::new(
                            (0.1 * i as f32) % 1.0,
                            (0.65 * i as f32) % 1.0,
                            (0.2 * i as f32) % 1.0,
                            1.0,
                        ),
                        Vec4::zero(),
                    ))));
                }

                ListBox::<ListData> {
                    id: gen_id!().into(),
                    props: ListData {
                        area: feather_ui::URect {
                            abs: AbsRect::new(0.0, 200.0, 0.0, 0.0),
                            rel: RelRect::new(0.0, 0.0, UNSIZED_AXIS, 1.0),
                        },
                        rlimits: feather_ui::RelLimits::new(
                            ZERO_POINT,
                            Vec2::new(1.0, f32::INFINITY),
                        ),
                        direction: feather_ui::RowDirection::BottomToTop,
                    }
                    .into(),
                    children,
                }
            };

            let flexlist = {
                let mut children: im::Vector<Option<Box<OutlineFrom<dyn flex::Prop>>>> =
                    im::Vector::new();

                for i in 0..args.count {
                    children.push_back(Some(Box::new(Shape::<FlexChild>::round_rect(
                        gen_id!().into(),
                        FlexChild {
                            area: feather_ui::URect {
                                abs: AbsRect::new(0.0, 0.0, 0.0, 40.0),
                                rel: RelRect::new(0.0, 0.0, 1.0, 0.0),
                            },
                            margin: AbsRect::new(8.0, 8.0, 4.0, 4.0).into(),
                            basis: 40.0,
                            grow: 0.0,
                            shrink: 0.0,
                        }
                        .into(),
                        0.0,
                        0.0,
                        Vec4::broadcast(8.0),
                        Vec4::new(
                            (0.1 * i as f32) % 1.0,
                            (0.65 * i as f32) % 1.0,
                            (0.2 * i as f32) % 1.0,
                            1.0,
                        ),
                        Vec4::zero(),
                    ))));
                }

                FlexBox::<MinimalFlex> {
                    id: gen_id!().into(),
                    props: MinimalFlex {
                        area: AbsRect::new(40.0, 40.0, 0.0, 200.0)
                            + RelRect::new(0.0, 0.0, 1.0, 0.0),
                    }
                    .into(),
                    children,
                }
            };

            let mut children: im::Vector<Option<Box<OutlineFrom<dyn fixed::Prop>>>> =
                im::Vector::new();
            children.push_back(Some(Box::new(button)));
            children.push_back(Some(Box::new(flexlist)));
            children.push_back(Some(Box::new(rectlist)));

            let region = Region {
                id: gen_id!().into(),
                props: FixedData {
                    area: FILL_URECT,
                    zindex: 0,
                    ..Default::default()
                }
                .into(),
                children,
            };
            let window = Window::new(
                gen_id!().into(),
                winit::window::Window::default_attributes()
                    .with_title(env!("CARGO_CRATE_NAME"))
                    .with_resizable(true),
                Box::new(region),
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
