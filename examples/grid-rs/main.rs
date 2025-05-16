// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use core::f32;
use feather_macro::*;
use feather_ui::layout::{base, fixed, grid, leaf};
use feather_ui::outline::button::Button;
use feather_ui::outline::gridbox::GridBox;
use feather_ui::outline::region::Region;
use feather_ui::outline::shape::Shape;
use feather_ui::outline::text::Text;
use feather_ui::outline::window::Window;
use feather_ui::outline::{mouse_area, OutlineFrom};
use feather_ui::persist::FnPersist;
use feather_ui::{
    gen_id, AbsRect, App, DAbsRect, DPoint, DRect, DValue, RelRect, Slot, SourceID, FILL_DRECT,
    FILL_URECT, UNSIZED_AXIS, ZERO_POINT,
};
use std::rc::Rc;
use ultraviolet::{Vec2, Vec4};

#[derive(PartialEq, Clone, Debug)]
struct CounterState {
    count: usize,
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

#[derive(Default, Empty, Area, Direction, RLimits, Padding)]
struct GridData {
    area: DRect,
    direction: feather_ui::RowDirection,
    rlimits: feather_ui::RelLimits,
    rows: Vec<DValue>,
    columns: Vec<DValue>,
    spacing: feather_ui::DPoint,
    padding: DAbsRect,
}

impl base::Anchor for GridData {}
impl base::Limits for GridData {}
impl fixed::Child for GridData {}

impl grid::Prop for GridData {
    fn rows(&self) -> &[DValue] {
        &self.rows
    }

    fn columns(&self) -> &[DValue] {
        &self.columns
    }

    fn spacing(&self) -> feather_ui::DPoint {
        self.spacing
    }

    fn direction(&self) -> feather_ui::RowDirection {
        feather_ui::RowDirection::TopToBottom
    }
}

#[derive(Default, Empty, Area)]
struct GridChild {
    area: DRect,
    x: usize,
    y: usize,
}

impl base::Padding for GridChild {}
impl base::Anchor for GridChild {}
impl base::Limits for GridChild {}
impl base::Margin for GridChild {}
impl base::RLimits for GridChild {}
impl base::Order for GridChild {}
impl leaf::Prop for GridChild {}
impl leaf::Padded for GridChild {}

impl grid::Child for GridChild {
    fn index(&self) -> (usize, usize) {
        (self.y, self.x)
    }

    fn span(&self) -> (usize, usize) {
        (1, 1)
    }
}

struct BasicApp {}

impl FnPersist<CounterState, im::HashMap<Rc<SourceID>, Option<Window>>> for BasicApp {
    type Store = (CounterState, im::HashMap<Rc<SourceID>, Option<Window>>);

    fn init(&self) -> Self::Store {
        (CounterState { count: 99999999 }, im::HashMap::new())
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
                        }
                        .into(),
                        anchor: feather_ui::RelPoint(Vec2 { x: 0.0, y: 0.0 }).into(),
                        ..Default::default()
                    }
                    .into(),
                    text: format!("Boxes: {}", args.count),
                    font_size: 40.0,
                    line_height: 56.0,
                    ..Default::default()
                };

                let rect = Shape::<DRect>::round_rect(
                    gen_id!().into(),
                    feather_ui::FILL_DRECT.into(),
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
                        }
                        .into(),
                        anchor: feather_ui::RelPoint(Vec2 { x: 0.5, y: 0.0 }).into(),
                        zindex: 0,
                    },
                    Slot(feather_ui::APP_SOURCE_ID.into(), 0),
                    children,
                )
            };

            const NUM_COLUMNS: usize = 5;
            let rectgrid = {
                let mut children: im::Vector<Option<Box<OutlineFrom<dyn grid::Prop>>>> =
                    im::Vector::new();
                for i in 0..args.count {
                    children.push_back(Some(Box::new(Shape::<GridChild>::round_rect(
                        gen_id!().into(),
                        GridChild {
                            area: feather_ui::URect::from(FILL_URECT).into(),
                            x: i % NUM_COLUMNS,
                            y: i / NUM_COLUMNS,
                        }
                        .into(),
                        0.0,
                        0.0,
                        Vec4::broadcast(4.0),
                        Vec4::new(
                            (0.1 * i as f32) % 1.0,
                            (0.65 * i as f32) % 1.0,
                            (0.2 * i as f32) % 1.0,
                            1.0,
                        ),
                        Vec4::zero(),
                    ))));

                    /*children.push_back(Some(Box::new(Text::<GridChild> {
                        id: gen_id!().into(),
                        props: GridChild {
                            area: feather_ui::URect::from(FILL_URECT).into(),
                            x: i % NUM_COLUMNS,
                            y: i / NUM_COLUMNS,
                        }
                        .into(),
                        text: format!("Cell: {}", i),
                        font_size: 20.0,
                        line_height: 22.0,
                        ..Default::default()
                    })));*/
                }

                GridBox::<GridData> {
                    id: gen_id!().into(),
                    props: GridData {
                        area: feather_ui::URect {
                            abs: AbsRect::new(0.0, 200.0, 0.0, 0.0),
                            rel: RelRect::new(0.0, 0.0, UNSIZED_AXIS, 1.0),
                        }
                        .into(),
                        rlimits: feather_ui::RelLimits::new(
                            ZERO_POINT,
                            Vec2::new(1.0, f32::INFINITY),
                        ),
                        direction: feather_ui::RowDirection::BottomToTop,
                        rows: [40.0, 20.0, 40.0, 20.0, 40.0, 20.0, 10.0]
                            .map(|x| DValue::from(x))
                            .to_vec(),
                        columns: [80.0, 40.0, 80.0, 40.0, 80.0]
                            .map(|x| DValue::from(x))
                            .to_vec(),
                        spacing: DPoint::from(Vec2::new(4.0, 4.0)),
                        padding: AbsRect::new(8.0, 8.0, 8.0, 8.0).into(),
                    }
                    .into(),
                    children,
                }
            };

            let mut children: im::Vector<Option<Box<OutlineFrom<dyn fixed::Prop>>>> =
                im::Vector::new();
            children.push_back(Some(Box::new(button)));
            children.push_back(Some(Box::new(rectgrid)));

            let region = Region {
                id: gen_id!().into(),
                props: FixedData {
                    area: FILL_DRECT,
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
