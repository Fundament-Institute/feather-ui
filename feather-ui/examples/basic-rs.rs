// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use bytemuck::Zeroable;
use core::f32;
use feather_macro::*;
use feather_ui::color::{sRGB, sRGB32};
use feather_ui::component::button::Button;
use feather_ui::component::region::Region;
use feather_ui::component::shape::{self};
use feather_ui::component::text::Text;
use feather_ui::component::window::Window;
use feather_ui::event::{CONSUME, EventRes, EventStream, EventStreamPrism};
use feather_ui::layout::{fixed, leaf};
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

use feather_ui::reactive::{
    DynSignal, MutableSignal, SignalNode, const_default, const_new, empty_signal,
};

fn basic_app_ui(app: &BasicApp) -> feather_ui::component::UI {
    let (button, evt, _) = {
        let text = Text::<FixedData> {
            props: Rc::new(FixedData {
                area: const_new(
                    AbsRect::new(8.0, 0.0, 8.0, 0.0)
                        + RelRect::new(0.0, 0.5, UNSIZED_AXIS, UNSIZED_AXIS),
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
            const_default().into(),
            const_default().into(),
            const_new(wide::f32x4::splat(10.0)).into(),
            const_new(sRGB::new(0.2, 0.7, 0.4, 1.0)).into(),
            const_default().into(),
            const_default().into(),
        );
        Button::<FixedData>::new(
            FixedData {
                area: const_new(
                    AbsRect::new(45.0, 45.0, 0.0, 0.0) + RelRect::new(0.0, 0.0, UNSIZED_AXIS, 1.0),
                )
                .into(),
                ..Default::default()
            },
            const_new(f32::INFINITY),
            const_new(feather_ui::children![fixed::Prop, rect, text]).into(),
        )
    };

    let block = {
        let text = Text::<FixedData> {
            props: Rc::new(FixedData {
                area: const_new(RelRect::new(0.5, 0.0, UNSIZED_AXIS, UNSIZED_AXIS).into()).into(),
                limits: const_new(feather_ui::AbsLimits::new(.., 10.0..200.0).into()).into(),
                rlimits: const_new(feather_ui::RelLimits::new(..1.0, ..)).into(),
                anchor: const_new(feather_ui::RelPoint::new(0.5, 0.0).into()).into(),
                padding: const_new(AbsRect::new(8.0, 8.0, 8.0, 8.0).into()).into(),
                ..Default::default()
            }),
            text: app
                .count
                .clone()
                .map_ex(|count| (0..*count).map(|_| "█").collect::<String>())
                .into(),
            font_size: const_new(40.0).into(),
            line_height: const_new(56.0).into(),
            wrap: const_new(feather_ui::cosmic_text::Wrap::WordOrGlyph).into(),
            align: const_new(Some(cosmic_text::Align::Center)).into(),
            ..Default::default()
        };

        let rect = shape::round_rect::<DRect>(
            feather_ui::FILL_DRECT,
            const_default().into(),
            const_default().into(),
            const_new(wide::f32x4::splat(10.0)).into(),
            const_new(sRGB::new(0.7, 0.2, 0.4, 1.0)).into(),
            const_default().into(),
            const_default().into(),
        );

        Region::<FixedData>::new_layer(
            FixedData {
                area: const_new(
                    AbsRect::new(45.0, 245.0, 0.0, 0.0)
                        + RelRect::new(0.0, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),
                )
                .into(),
                limits: const_new(feather_ui::AbsLimits::new(100.0..300.0, ..).into()).into(),
                ..Default::default()
            },
            const_new(sRGB32::from_alpha(128)).into(),
            const_default().into(),
            const_new(feather_ui::children![fixed::Prop, rect, text]).into(),
        )
    };

    let count = app.count.clone();

    evt.prism().OnClick.subscribe(move |_| -> EventRes {
        {
            println!("{}", feather_ui::reactive::trace_deps(count.clone()));
            *count.borrow_mut() += 1;

            CONSUME
        }
    });

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
        const_new(feather_ui::children![fixed::Prop, button]).into_dyn(),
    );
    let window = Window::new(
        winit::window::Window::default_attributes()
            .with_title(env!("CARGO_CRATE_NAME"))
            .with_resizable(true),
        Box::new(region),
    );

    feather_ui::component::UI {
        children: const_new(imbl::vector![Rc::new(window)]).into_dyn(),
    }
}

fn main() {
    let (mut app, event_loop) = App::<BasicApp, ()>::new(
        const_new(BasicApp {
            count: MutableSignal::new(3),
        })
        .into_dyn(),
        basic_app_ui,
        None,
        None,
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
