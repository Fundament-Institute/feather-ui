// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use bytemuck::Zeroable;
use feather_macro::*;
use feather_ui::color::{sRGB, sRGB32};
use feather_ui::component::button::Button;
use feather_ui::component::region::Region;
use feather_ui::component::text::Text;
use feather_ui::component::window::Window;
use feather_ui::component::{mouse_area, shape};
use feather_ui::layout::{fixed, leaf};
use feather_ui::persist::{FnPersist2, FnPersistStore};
use feather_ui::{
    AbsRect, App, DAbsPoint, DAbsRect, DPoint, DRect, PxRect, RelRect, ScopeID, Slot, SourceID,
    UNSIZED_AXIS, gen_id, im, winit,
};
use std::rc::Rc;
use std::sync::Arc;

#[derive(PartialEq, Clone, Debug)]
struct CounterState {
    count: i32,
}

#[derive(Default, Empty, Area, Anchor, ZIndex, Limits, RLimits, Padding)]
struct FixedData {
    area: DRect,
    anchor: DPoint,
    limits: feather_ui::DLimits,
    rlimits: feather_ui::RelLimits,
    padding: DAbsRect,
    zindex: i32,
}

impl fixed::Prop for FixedData {}
impl fixed::Child for FixedData {}
impl leaf::Prop for FixedData {}
impl leaf::Padded for FixedData {}

struct BasicApp {}

impl FnPersistStore for BasicApp {
    type Store = (CounterState, im::HashMap<Arc<SourceID>, Option<Window>>);
}

impl FnPersist2<CounterState, ScopeID<'_>, im::HashMap<Arc<SourceID>, Option<Window>>>
    for BasicApp
{
    fn init(&self) -> Self::Store {
        (CounterState { count: -1 }, im::HashMap::new())
    }
    fn call(
        &mut self,
        mut store: Self::Store,
        app: CounterState,
        mut id: ScopeID<'_>,
    ) -> (Self::Store, im::HashMap<Arc<SourceID>, Option<Window>>) {
        if store.0 != app {
            let button = {
                let text = Text::<FixedData> {
                    id: gen_id!(id),
                    props: Rc::new(FixedData {
                        area: AbsRect::new(8.0, 0.0, 8.0, 0.0)
                            + RelRect::new(0.0, 0.5, UNSIZED_AXIS, UNSIZED_AXIS),
                        anchor: feather_ui::RelPoint::new(0.0, 0.5).into(),
                        ..Default::default()
                    }),
                    color: sRGB::new(1.0, 1.0, 0.0, 1.0),
                    text: format!("Clicks: {}", app.count),
                    font_size: 40.0,
                    line_height: 56.0,
                    align: Some(cosmic_text::Align::Center),
                    ..Default::default()
                };

                let rect = shape::round_rect::<DRect>(
                    gen_id!(id),
                    feather_ui::FILL_DRECT,
                    0.0,
                    0.0,
                    wide::f32x4::splat(10.0),
                    sRGB::new(0.2, 0.7, 0.4, 1.0),
                    sRGB::transparent(),
                    DAbsPoint::zero(),
                );

                Button::<FixedData>::new(
                    gen_id!(id),
                    FixedData {
                        area: AbsRect::new(45.0, 45.0, 0.0, 0.0)
                            + RelRect::new(0.0, 0.0, UNSIZED_AXIS, 1.0),

                        ..Default::default()
                    },
                    Slot(feather_ui::APP_SOURCE_ID.into(), 0),
                    feather_ui::children![fixed::Prop, rect, text],
                )
            };

            let block = {
                let text = Text::<FixedData> {
                    id: gen_id!(id),
                    props: Rc::new(FixedData {
                        area: RelRect::new(0.5, 0.0, UNSIZED_AXIS, UNSIZED_AXIS).into(),
                        limits: feather_ui::AbsLimits::new(.., 10.0..200.0).into(),
                        rlimits: feather_ui::RelLimits::new(..1.0, ..),
                        anchor: feather_ui::RelPoint::new(0.5, 0.0).into(),
                        padding: AbsRect::new(8.0, 8.0, 8.0, 8.0).into(),
                        ..Default::default()
                    }),
                    text: (0..app.count).map(|_| "█").collect::<String>(),
                    font_size: 40.0,
                    line_height: 56.0,
                    wrap: feather_ui::cosmic_text::Wrap::WordOrGlyph,
                    align: Some(cosmic_text::Align::Center),
                    ..Default::default()
                };

                let rect = shape::round_rect::<DRect>(
                    gen_id!(id),
                    feather_ui::FILL_DRECT,
                    0.0,
                    0.0,
                    wide::f32x4::splat(10.0),
                    sRGB::new(0.7, 0.2, 0.4, 1.0),
                    sRGB::transparent(),
                    DAbsPoint::zero(),
                );

                Region::<FixedData>::new_layer(
                    gen_id!(id),
                    FixedData {
                        area: AbsRect::new(45.0, 245.0, 0.0, 0.0)
                            + RelRect::new(0.0, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),
                        limits: feather_ui::AbsLimits::new(100.0..300.0, ..).into(),
                        ..Default::default()
                    },
                    sRGB32::from_alpha(128),
                    0.0,
                    feather_ui::children![fixed::Prop, rect, text],
                )
            };

            let pixel = shape::round_rect::<DRect>(
                gen_id!(id),
                PxRect::new(1.0, 1.0, 2.0, 2.0).into(),
                0.0,
                0.0,
                wide::f32x4::zeroed(),
                sRGB::new(1.0, 1.0, 1.0, 1.0),
                sRGB::transparent(),
                DAbsPoint::zero(),
            );

            let region = Region::new(
                gen_id!(id),
                FixedData {
                    area: AbsRect::new(90.0, 90.0, 0.0, 200.0)
                        + RelRect::new(0.0, 0.0, UNSIZED_AXIS, 0.0),
                    zindex: 0,
                    ..Default::default()
                },
                feather_ui::children![fixed::Prop, button, block, pixel],
            );
            let window = Window::new(
                gen_id!(id),
                winit::window::Window::default_attributes()
                    .with_title(env!("CARGO_CRATE_NAME"))
                    .with_resizable(true),
                Box::new(region),
            );

            store.1 = im::HashMap::new();
            store.1.insert(window.id.clone(), Some(window));
            store.0 = app.clone();
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
         -> feather_ui::InputResult<()> {
            {
                appdata.count += 1;
                feather_ui::InputResult::Consume(())
            }
        }
        .wrap(),
    );

    let (mut app, event_loop, _, _) = App::<CounterState, BasicApp>::new::<()>(
        CounterState { count: 0 },
        vec![onclick],
        BasicApp {},
        |_| (),
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
