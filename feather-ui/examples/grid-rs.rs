// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use core::f32;
use feather_macro::*;
use feather_ui::color::sRGB;
use feather_ui::component::button::Button;
use feather_ui::component::gridbox::GridBox;
use feather_ui::component::region::Region;
use feather_ui::component::shape::{Shape, ShapeKind};
use feather_ui::component::text::Text;
use feather_ui::component::window::Window;
use feather_ui::component::{ChildOf, mouse_area};
use feather_ui::layout::{base, fixed, grid, leaf};
use feather_ui::persist::{FnPersist2, FnPersistStore};
use feather_ui::{
    AbsPoint, AbsRect, App, DRect, DSize, DValue, FILL_DRECT, InputResult, RelRect, ScopeID, Slot,
    SourceID, UNSIZED_AXIS, UPerimeter,
};
use std::sync::Arc;

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
    padding: UPerimeter,
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
    fn coord(&self) -> (usize, usize) {
        (self.y, self.x)
    }

    fn span(&self) -> (usize, usize) {
        (1, 1)
    }
}

struct BasicApp {}

impl FnPersistStore for BasicApp {
    type Store = (CounterState, imbl::HashMap<Arc<SourceID>, Option<Window>>);
}

impl FnPersist2<&CounterState, ScopeID<'_>, imbl::HashMap<Arc<SourceID>, Option<Window>>>
    for BasicApp
{
    fn init(&self) -> Self::Store {
        (CounterState { count: 99999999 }, imbl::HashMap::new())
    }
    fn call(
        &mut self,
        mut store: Self::Store,
        args: &CounterState,
        mut scope: ScopeID<'_>,
    ) -> (Self::Store, imbl::HashMap<Arc<SourceID>, Option<Window>>) {
        if store.0 != *args {
            let button = {
                let text = {
                    Text::<FixedData> {
                        id: scope.create(),
                        props: FixedData {
                            area: AbsRect::new(10.0, 15.0, 10.0, 15.0)
                                + RelRect::new(0.0, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),
                            anchor: feather_ui::RelPoint::zero().into(),
                            ..Default::default()
                        }
                        .into(),
                        text: format!("Boxes: {}", args.count),
                        font_size: 40.0,
                        line_height: 56.0,
                        ..Default::default()
                    }
                };

                let rect = Shape::<DRect, { ShapeKind::RoundRect as u8 }>::new(
                    scope.create(),
                    feather_ui::FILL_DRECT,
                    0.0,
                    0.0,
                    wide::f32x4::splat(10.0),
                    sRGB::new(0.2, 0.7, 0.4, 1.0),
                    sRGB::transparent(),
                    DSize::zero(),
                );

                Button::<FixedData>::new(
                    scope.create(),
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

            const NUM_COLUMNS: usize = 5;
            let rectgrid = {
                let mut children: imbl::Vector<Rc<ChildOf<dyn grid::Prop>>> = imbl::Vector::new();
                {
                    for (i, id) in scope.iter(0..args.count) {
                        children.push_back(Box::new(Shape::<
                            GridChild,
                            { ShapeKind::RoundRect as u8 },
                        >::new(
                            id,
                            GridChild {
                                area: FILL_DRECT,
                                x: i % NUM_COLUMNS,
                                y: i / NUM_COLUMNS,
                            },
                            0.0,
                            0.0,
                            wide::f32x4::splat(4.0),
                            sRGB::new(
                                (0.1 * i as f32) % 1.0,
                                (0.65 * i as f32) % 1.0,
                                (0.2 * i as f32) % 1.0,
                                1.0,
                            ),
                            sRGB::transparent(),
                            DSize::zero(),
                        )));
                    }
                }

                GridBox::<GridData>::new(
                    scope.create(),
                    GridData {
                        area: AbsRect::new(0.0, 200.0, 0.0, 0.0)
                            + RelRect::new(0.0, 0.0, UNSIZED_AXIS, 1.0),

                        rlimits: feather_ui::RelLimits::new(0.0..1.0, 0.0..),
                        direction: feather_ui::RowDirection::LeftToRight,
                        rows: [40.0, 20.0, 40.0, 20.0, 40.0, 20.0, 10.0]
                            .map(DValue::from)
                            .to_vec(),
                        columns: [80.0, 40.0, 80.0, 40.0, 80.0].map(DValue::from).to_vec(),
                        spacing: AbsPoint::new(4.0, 4.0).into(),
                        padding: AbsRect::new(8.0, 8.0, 8.0, 8.0).into(),
                    },
                    children,
                )
            };

            let region = Region::new(
                scope.create(),
                FixedData {
                    area: FILL_DRECT,
                    zindex: 0,
                    ..Default::default()
                },
                feather_ui::children![fixed::Prop, button, rectgrid],
            );
            let window = Window::new(
                scope.create(),
                winit::window::Window::default_attributes()
                    .with_title(env!("CARGO_CRATE_NAME"))
                    .with_resizable(true),
                Box::new(region),
            );

            store.1 = imbl::HashMap::new();
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
         mut appdata: feather_ui::AccessCell<CounterState>|
         -> InputResult<()> {
            {
                appdata.count += 1;
                InputResult::Consume(())
            }
        }
        .wrap(),
    );

    let (mut app, event_loop, _, _) = App::<CounterState, BasicApp, ()>::new(
        CounterState { count: 0 },
        vec![onclick],
        BasicApp {},
        None,
        None,
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
