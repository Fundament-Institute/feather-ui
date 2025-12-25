// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use bytemuck::Zeroable;
use feather_macro::*;
use feather_ui::color::{sRGB, sRGB32};
//use feather_ui::component::button::Button;
use feather_ui::component::region::Region;
//use feather_ui::component::text::Text;
use feather_ui::component::shape;
use feather_ui::component::window::Window;
use feather_ui::layout::{fixed, leaf};
use feather_ui::{
    AbsRect, App, DAbsPoint, DAbsRect, DPoint, DRect, PxRect, RelRect, UNSIZED_AXIS, winit,
};
use std::rc::Rc;

#[derive(Default, Empty, Area, Anchor, ZIndex, Limits, RLimits, Padding)]
struct FixedData {
    area: MutableSignal<DRect>,
    anchor: MutableSignal<DPoint>,
    limits: MutableSignal<feather_ui::DLimits>,
    rlimits: MutableSignal<feather_ui::RelLimits>,
    padding: MutableSignal<DAbsRect>,
    zindex: MutableSignal<i32>,
}

impl fixed::Prop for FixedData {}
impl fixed::Child for FixedData {}
impl leaf::Prop for FixedData {}
impl leaf::Padded for FixedData {}

struct BasicApp {
    count: i32,
}

use feather_ui::reactive::{DynSignal, MutableSignal, const_signal};

fn basic_app_ui(app: &BasicApp) -> feather_ui::component::UI {
    /*let button = {
        let text = Text::<FixedData> {
            props: Rc::new(FixedData {
                area: (AbsRect::new(8.0, 0.0, 8.0, 0.0)
                    + RelRect::new(0.0, 0.5, UNSIZED_AXIS, UNSIZED_AXIS))
                .to_signal(),
                anchor: const_signal(feather_ui::RelPoint::new(0.0, 0.5).into()),
                ..Default::default()
            }),
            color: const_signal(sRGB::new(1.0, 1.0, 0.0, 1.0)),
            text: app.map(|count| format!("Clicks: {}", count)),
            font_size: 40.0.to_signal(),
            line_height: 56.0.to_signal(),
            ..Default::default()
        };
        let rect = Shape::<DRect, { ShapeKind::RoundRect as u8 }>::new(
            feather_ui::FILL_DRECT.into(),
            0.0,
            0.0,
            wide::f32x4::splat(10.0),
            sRGB::new(0.2, 0.7, 0.4, 1.0),
            sRGB::transparent(),
        );
        Button::<FixedData>::new(
            FixedData {
                area: (AbsRect::new(45.0, 45.0, 0.0, 0.0)
                    + RelRect::new(0.0, 0.0, UNSIZED_AXIS, 1.0))
                .to_signal(),
                ..Default::default()
            },
            Slot(feather_ui::APP_SOURCE_ID.into(), 0), // replace with event stream
            feather_ui::children![fixed::Prop, rect, text],
        )
    };*/

    let pixel = shape::round_rect::<DRect>(
        PxRect::new(1.0, 1.0, 2.0, 2.0).into(),
        const_signal(0.0).into(),
        const_signal(0.0).into(),
        const_signal(wide::f32x4::zeroed()).into(),
        const_signal(sRGB::new(1.0, 1.0, 1.0, 1.0)).into(),
        const_signal(sRGB::transparent()).into(),
        const_signal(DAbsPoint::zero()).into(),
    );

    let region = Region::new(
        FixedData {
            area: MutableSignal::new(
                AbsRect::new(90.0, 90.0, 0.0, 200.0) + RelRect::new(0.0, 0.0, UNSIZED_AXIS, 0.0),
            ),
            zindex: MutableSignal::new(0),
            ..Default::default()
        },
        MutableSignal::new(feather_ui::children![fixed::Prop, pixel]).into_dyn_signal(),
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
    /*let onclick = Box::new(
        |_: mouse_area::MouseAreaEvent,
         mut appdata: feather_ui::AccessCell<CounterState>|
         -> feather_ui::InputResult<()> {
            {
                appdata.count += 1;
                feather_ui::InputResult::Consume(())
            }
        }
        .wrap(),
    );*/

    let (mut app, event_loop) = App::<BasicApp, ()>::new(
        MutableSignal::new(BasicApp { count: 0 }).into_dyn_signal(),
        basic_app_ui,
        None,
        None,
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
