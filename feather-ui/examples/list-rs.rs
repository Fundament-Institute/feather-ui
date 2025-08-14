// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use core::f32;
use feather_macro::*;
use feather_ui::color::sRGB;
use feather_ui::component::button::Button;
use feather_ui::component::flexbox::FlexBox;
use feather_ui::component::listbox::ListBox;
use feather_ui::component::region::Region;
use feather_ui::component::shape::{Shape, ShapeKind};
use feather_ui::component::text::Text;
use feather_ui::component::window::Window;
use feather_ui::component::{ChildOf, mouse_area};
use feather_ui::layout::{base, fixed, flex, leaf, list};
use feather_ui::persist::FnPersist;
use feather_ui::{
    AbsRect, App, DRect, DValue, DataID, FILL_DRECT, RelRect, Slot, SourceID, UNSIZED_AXIS, gen_id,
    im,
};
use std::sync::Arc;

#[derive(PartialEq, Clone, Debug)]
struct CounterState {
    count: i32,
}

#[derive(Default, Empty, Area, Anchor, ZIndex)]
struct FixedData {
    area: DRect,
    anchor: feather_ui::DPoint,
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
    area: DRect,
    direction: feather_ui::RowDirection,
    rlimits: feather_ui::RelLimits,
}

impl base::Limits for ListData {}
impl list::Prop for ListData {}
impl fixed::Child for ListData {}

#[derive(Default, Empty, Area, Margin)]
struct ListChild {
    area: DRect,
    margin: DRect,
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
    area: DRect,
    margin: DRect,
    basis: DValue,
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
    area: DRect,
}

impl base::Obstacles for MinimalFlex {
    fn obstacles(&self) -> &[feather_ui::DAbsRect] {
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

impl FnPersist<CounterState, im::HashMap<Arc<SourceID>, Option<Window>>> for BasicApp {
    type Store = (CounterState, im::HashMap<Arc<SourceID>, Option<Window>>);

    fn init(&self) -> Self::Store {
        (CounterState { count: -1 }, im::HashMap::new())
    }
    fn call(
        &mut self,
        mut store: Self::Store,
        args: &CounterState,
    ) -> (Self::Store, im::HashMap<Arc<SourceID>, Option<Window>>) {
        if store.0 != *args {
            let button = {
                let text = Text::<FixedData> {
                    id: gen_id!(),
                    props: FixedData {
                        area: AbsRect::new(10.0, 15.0, 10.0, 15.0)
                            + RelRect::new(0.0, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),

                        anchor: feather_ui::RelPoint::new(0.0, 0.0).into(),
                        ..Default::default()
                    }
                    .into(),
                    text: format!("Boxes: {}", args.count),
                    font_size: 40.0,
                    line_height: 56.0,
                    ..Default::default()
                };

                let rect = Shape::<DRect, { ShapeKind::RoundRect as u8 }>::new(
                    gen_id!(),
                    feather_ui::FILL_DRECT.into(),
                    0.0,
                    0.0,
                    wide::f32x4::splat(10.0),
                    sRGB::new(0.2, 0.7, 0.4, 1.0),
                    sRGB::transparent(),
                );

                Button::<FixedData>::new(
                    gen_id!(),
                    FixedData {
                        area: AbsRect::new(0.0, 20.0, 0.0, 0.0)
                            + RelRect::new(0.5, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),

                        anchor: feather_ui::RelPoint::new(0.5, 0.0).into(),
                        zindex: 0,
                    },
                    Slot(feather_ui::APP_SOURCE_ID.into(), 0),
                    feather_ui::children![fixed::Prop, rect, text],
                )
            };

            let rectlist = {
                let mut children: im::Vector<Option<Box<ChildOf<dyn list::Prop>>>> =
                    im::Vector::new();

                let rect_id = gen_id!();

                for i in 0..args.count {
                    children.push_back(Some(Box::new(Shape::<
                        ListChild,
                        { ShapeKind::RoundRect as u8 },
                    >::new(
                        rect_id.child(DataID::Int(i as i64)),
                        ListChild {
                            area: AbsRect::new(0.0, 0.0, 40.0, 40.0).into(),
                            margin: AbsRect::new(8.0, 8.0, 4.0, 4.0).into(),
                        }
                        .into(),
                        0.0,
                        0.0,
                        wide::f32x4::splat(8.0),
                        sRGB::new(
                            (0.1 * i as f32) % 1.0,
                            (0.65 * i as f32) % 1.0,
                            (0.2 * i as f32) % 1.0,
                            1.0,
                        ),
                        sRGB::transparent(),
                    ))));
                }

                ListBox::<ListData>::new(
                    gen_id!(),
                    ListData {
                        area: AbsRect::new(0.0, 200.0, 0.0, 0.0)
                            + RelRect::new(0.0, 0.0, UNSIZED_AXIS, 1.0),

                        rlimits: feather_ui::RelLimits::new(0.0..1.0, 0.0..),
                        direction: feather_ui::RowDirection::BottomToTop,
                    }
                    .into(),
                    children,
                )
            };

            let flexlist = {
                let mut children: im::Vector<Option<Box<ChildOf<dyn flex::Prop>>>> =
                    im::Vector::new();

                let box_id = gen_id!();
                for i in 0..args.count {
                    children.push_back(Some(Box::new(Shape::<
                        FlexChild,
                        { ShapeKind::RoundRect as u8 },
                    >::new(
                        box_id.child(DataID::Int(i as i64)),
                        FlexChild {
                            area: AbsRect::new(0.0, 0.0, 0.0, 40.0)
                                + RelRect::new(0.0, 0.0, 1.0, 0.0),

                            margin: AbsRect::new(8.0, 8.0, 4.0, 4.0).into(),
                            basis: 40.0.into(),
                            grow: 0.0,
                            shrink: 0.0,
                        }
                        .into(),
                        0.0,
                        0.0,
                        wide::f32x4::splat(8.0),
                        sRGB::new(
                            (0.1 * i as f32) % 1.0,
                            (0.65 * i as f32) % 1.0,
                            (0.2 * i as f32) % 1.0,
                            1.0,
                        ),
                        sRGB::transparent(),
                    ))));
                }

                FlexBox::<MinimalFlex>::new(
                    gen_id!(),
                    MinimalFlex {
                        area: (AbsRect::new(40.0, 40.0, 0.0, 200.0)
                            + RelRect::new(0.0, 0.0, 1.0, 0.0))
                        .into(),
                    }
                    .into(),
                    children,
                )
            };

            let region = Region::new(
                gen_id!(),
                FixedData {
                    area: FILL_DRECT,
                    zindex: 0,
                    ..Default::default()
                }
                .into(),
                feather_ui::children![fixed::Prop, button, flexlist, rectlist],
            );
            let window = Window::new(
                gen_id!(),
                feather_ui::winit::window::Window::default_attributes()
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
        feather_ui::winit::event_loop::EventLoop<()>,
    ) = App::new(
        CounterState { count: 0 },
        vec![onclick],
        BasicApp {},
        |_| (),
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
