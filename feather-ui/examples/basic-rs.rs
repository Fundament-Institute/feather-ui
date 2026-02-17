// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use bytemuck::Zeroable;
use feather_macro::*;
use feather_ui::color::{sRGB, sRGB32};
use feather_ui::component::button::Button;
use feather_ui::component::region::Region;
use feather_ui::component::shape::{self, Shape, ShapeKind};
use feather_ui::component::text::Text;
use feather_ui::component::window::Window;
use feather_ui::event::{CONSUME, EventRes, EventStream};
use feather_ui::layout::{fixed, leaf};
use feather_ui::reactive::SignalMap;
use feather_ui::{
    AbsRect, App, DAbsPoint, DAbsRect, DPoint, DRect, PxRect, RelRect, UNSIZED_AXIS, winit,
};
use std::rc::Rc;

#[derive(Default, Empty, Area, Anchor, ZIndex, Limits, RLimits, Padding)]
struct FixedData {
    area: DynSignal<DRect>,
    anchor: DynSignal<DPoint>,
    limits: DynSignal<feather_ui::DLimits>,
    rlimits: DynSignal<feather_ui::RelLimits>,
    padding: DynSignal<DAbsRect>,
    zindex: DynSignal<i32>,
}

impl fixed::Prop for FixedData {}
impl fixed::Child for FixedData {}
impl leaf::Prop for FixedData {}
impl leaf::Padded for FixedData {}

struct BasicApp {
    count: MutableSignal<i32>,
}

use feather_ui::reactive::{ConstSignal, DynSignal, MutableSignal, Signal, const_new};

fn basic_app_ui(app: &BasicApp) -> feather_ui::component::UI {
    let (button, evt, _) = {
        let text = Text::<FixedData> {
            props: Rc::new(FixedData {
                area: const_new(
                    (AbsRect::new(8.0, 0.0, 8.0, 0.0)
                        + RelRect::new(0.0, 0.5, UNSIZED_AXIS, UNSIZED_AXIS)),
                )
                .into(),
                anchor: const_new(feather_ui::RelPoint::new(0.0, 0.5).into()).into(),
                ..Default::default()
            }),
            attributes: const_new(cosmic_text::AttrsOwned::new(
                &cosmic_text::Attrs::new().color(sRGB::new(1.0, 1.0, 0.0, 1.0).into()),
            ))
            .into(),
            text: app
                .count
                .clone()
                .map_ex(|count| format!("Clicks: {}", count))
                .into(),
            font_size: const_new(40.0).into(),
            line_height: const_new(56.0).into(),
            ..Default::default()
        };
        let rect = shape::round_rect::<DRect>(
            feather_ui::FILL_DRECT.into(),
            const_new(0.0).into(),
            const_new(0.0).into(),
            const_new(wide::f32x4::splat(10.0)).into(),
            const_new(sRGB::new(0.2, 0.7, 0.4, 1.0)).into(),
            const_new(sRGB::transparent()).into(),
            const_new(DAbsPoint::zero()).into(),
        );
        Button::<FixedData>::new(
            FixedData {
                area: const_new(
                    AbsRect::new(45.0, 45.0, 0.0, 0.0) + RelRect::new(0.0, 0.0, UNSIZED_AXIS, 1.0),
                )
                .into(),
                ..Default::default()
            },
            const_new(3.0),
            const_new(feather_ui::children![fixed::Prop, rect, text]).into(),
        )
    };

    let count = app.count.clone();

    evt.subscribe(
        move |_: feather_ui::component::mouse_area::MouseAreaEvent| -> EventRes {
            {
                *count.borrow_mut() += 1;
                CONSUME
            }
        },
    );

    let pixel = shape::round_rect::<DRect>(
        PxRect::new(1.0, 1.0, 2.0, 2.0).into(),
        const_new(0.0).into(),
        const_new(0.0).into(),
        const_new(wide::f32x4::zeroed()).into(),
        const_new(sRGB::new(1.0, 1.0, 1.0, 0.5)).into(),
        const_new(sRGB::transparent()).into(),
        const_new(DAbsPoint::zero()).into(),
    );

    let region = Region::new(
        FixedData {
            area: const_new(
                AbsRect::new(90.0, 90.0, 0.0, 200.0) + RelRect::new(0.0, 0.0, UNSIZED_AXIS, 0.0),
            )
            .into(),
            zindex: const_new(0).into(),
            ..Default::default()
        },
        MutableSignal::new(feather_ui::children![fixed::Prop, pixel, button]).into_dyn_signal(),
    );
    let window = Window::new(
        winit::window::Window::default_attributes()
            .with_title(env!("CARGO_CRATE_NAME"))
            .with_resizable(true),
        Box::new(region),
    );

    feather_ui::component::UI {
        children: MutableSignal::new(imbl::vector![Rc::new(window)]).into_dyn_signal(),
    }
}
/*
impl FnPersist2<&CounterState, ScopeID<'_>, imbl::HashMap<Arc<SourceID>, Option<Window>>>
    for BasicApp
{
    fn init(&self) -> Self::Store {
        (CounterState { count: -1 }, imbl::HashMap::new())
    }
    fn call(
        &mut self,
        mut store: Self::Store,
        app: &CounterState,
        mut id: ScopeID<'_>,
    ) -> (Self::Store, imbl::HashMap<Arc<SourceID>, Option<Window>>) {
        if store.0 != *app {
            let button = {
                let text = Text::<FixedData> {
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
                    feather_ui::FILL_DRECT,
                    0.0,
                    0.0,
                    wide::f32x4::splat(10.0),
                    sRGB::new(0.2, 0.7, 0.4, 1.0),
                    sRGB::transparent(),
                    DAbsPoint::zero(),
                );

                Button::<FixedData>::new(
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
                    feather_ui::FILL_DRECT,
                    0.0,
                    0.0,
                    wide::f32x4::splat(10.0),
                    sRGB::new(0.7, 0.2, 0.4, 1.0),
                    sRGB::transparent(),
                    DAbsPoint::zero(),
                );

                Region::<FixedData>::new_layer(
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
                PxRect::new(1.0, 1.0, 2.0, 2.0).into(),
                0.0,
                0.0,
                wide::f32x4::zeroed(),
                sRGB::new(1.0, 1.0, 1.0, 1.0),
                sRGB::transparent(),
                DAbsPoint::zero(),
            );

            let region = Region::new(
                FixedData {
                    area: AbsRect::new(90.0, 90.0, 0.0, 200.0)
                        + RelRect::new(0.0, 0.0, UNSIZED_AXIS, 0.0),
                    zindex: 0,
                    ..Default::default()
                },
                feather_ui::children![fixed::Prop, button, block, pixel],
            );
            let window = Window::new(
                winit::window::Window::default_attributes()
                    .with_title(env!("CARGO_CRATE_NAME"))
                    .with_resizable(true),
                Box::new(region),
            );

            store.1 = imbl::HashMap::new();
            store.1.insert(window.id.clone(), Some(window));
            store.0 = app.clone();
        }
        let windows = store.1.clone();
        (store, windows)
    }
}*/

//use feather_ui::WrapEventEx;

fn main() {
    let (mut app, event_loop) = App::<BasicApp, ()>::new(
        MutableSignal::new(BasicApp {
            count: MutableSignal::new(0),
        })
        .into_dyn_signal(),
        basic_app_ui,
        None,
        None,
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
