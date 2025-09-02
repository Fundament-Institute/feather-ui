// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use feather_macro::*;
use feather_ui::color::sRGB;
use feather_ui::component::image::Image;
use feather_ui::component::region::Region;
use feather_ui::component::shape::{Shape, ShapeKind};
use feather_ui::component::window::Window;
use feather_ui::layout::{fixed, leaf};
use feather_ui::persist::{FnPersist2, FnPersistStore};
use feather_ui::{
    AbsPoint, AbsRect, App, DAbsRect, DPoint, DRect, PxRect, RelRect, ScopeID, SourceID,
    UNSIZED_AXIS, gen_id, im, winit,
};
use std::path::PathBuf;
use std::sync::Arc;

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
    type Store = im::HashMap<Arc<SourceID>, Option<Window>>;
}

impl FnPersist2<(), ScopeID<'_>, im::HashMap<Arc<SourceID>, Option<Window>>> for BasicApp {
    fn init(&self) -> Self::Store {
        im::HashMap::new()
    }

    fn call(
        &mut self,
        _: Self::Store,
        _: (),
        mut scope: ScopeID<'_>,
    ) -> (Self::Store, im::HashMap<Arc<SourceID>, Option<Window>>) {
        let pixel = Shape::<DRect, { ShapeKind::RoundRect as u8 }>::new(
            scope.create(),
            PxRect::new(1.0, 1.0, 2.0, 2.0).into(),
            0.0,
            0.0,
            wide::f32x4::splat(0.0),
            sRGB::new(1.0, 1.0, 1.0, 1.0),
            sRGB::transparent(),
            feather_ui::DAbsPoint::zero(),
        );

        let mut children: im::Vector<Option<Box<feather_ui::component::ChildOf<dyn fixed::Prop>>>> =
            im::Vector::new();
        children.push_back(Some(Box::new(pixel)));

        let mut genimage = |pos: AbsPoint,
                            w: Option<f32>,
                            h: Option<f32>,
                            res: &dyn feather_ui::resource::Location,
                            size: Option<AbsPoint>| {
            Image::<DRect>::new(
                scope.create(),
                AbsRect::new(
                    pos.x,
                    pos.y,
                    w.map(|x| x + pos.x).unwrap_or_default(),
                    h.map(|y| y + pos.y).unwrap_or_default(),
                ) + RelRect::new(
                    0.0,
                    0.0,
                    if w.is_none() { UNSIZED_AXIS } else { 0.0 },
                    if h.is_none() { UNSIZED_AXIS } else { 0.0 },
                ),
                res,
                size.unwrap_or_default().into(),
                false,
            )
        };

        #[cfg(feature = "png")]
        {
            let testimage = PathBuf::from("./premul_test.png");

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(0.0, 0.0),
                Some(100.0),
                Some(100.0),
                &testimage,
                None,
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(100.0, 0.0),
                None,
                Some(100.0),
                &testimage,
                None,
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(0.0, 100.0),
                None,
                None,
                &testimage,
                Some(AbsPoint::new(100.0, 100.0)),
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(100.0, 100.0),
                None,
                None,
                &testimage,
                None,
            ))));
        }

        #[cfg(feature = "svg")]
        {
            let testsvg = PathBuf::from("./FRI_logo.svg");

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(200.0, 0.0),
                Some(100.0),
                Some(100.0),
                &testsvg,
                None,
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(300.0, 0.0),
                None,
                Some(100.0),
                &testsvg,
                None,
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(200.0, 100.0),
                None,
                None,
                &testsvg,
                Some(AbsPoint::new(100.0, 100.0)),
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(300.0, 100.0),
                None,
                None,
                &testsvg,
                None,
            ))));
        }

        #[cfg(feature = "png")]
        {
            let testimage = PathBuf::from("./test_color.png");

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(0.0, 200.0),
                Some(100.0),
                Some(100.0),
                &testimage,
                None,
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(100.0, 200.0),
                Some(100.0),
                None,
                &testimage,
                None,
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(0.0, 300.0),
                None,
                None,
                &testimage,
                Some(AbsPoint::new(100.0, 100.0)),
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(100.0, 300.0),
                None,
                None,
                &testimage,
                None,
            ))));
        }

        #[cfg(feature = "jxl")]
        {
            let testimage = PathBuf::from("./dice.jxl");

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(200.0, 200.0),
                Some(100.0),
                Some(100.0),
                &testimage,
                None,
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(300.0, 200.0),
                Some(100.0),
                None,
                &testimage,
                None,
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(200.0, 300.0),
                None,
                None,
                &testimage,
                Some(AbsPoint::new(100.0, 100.0)),
            ))));

            children.push_back(Some(Box::new(genimage(
                AbsPoint::new(300.0, 300.0),
                None,
                None,
                &testimage,
                None,
            ))));
        }

        let region = Region::new(
            gen_id!(scope),
            FixedData {
                area: AbsRect::new(10.0, 10.0, -10.0, -10.0) + RelRect::new(0.0, 0.0, 1.0, 1.0),

                zindex: 0,
                ..Default::default()
            },
            children,
        );

        #[cfg(feature = "svg")]
        let icon = Some(
            feather_ui::resource::load_icon(&std::path::PathBuf::from("./FRI_logo.svg")).unwrap(),
        );
        #[cfg(not(feature = "svg"))]
        let icon = None;

        let window = Window::new(
            gen_id!(scope),
            winit::window::Window::default_attributes()
                .with_title(env!("CARGO_CRATE_NAME"))
                .with_resizable(true)
                .with_window_icon(icon),
            Box::new(region),
        );

        let mut store = im::HashMap::new();
        store.insert(window.id.clone(), Some(window));
        let windows = store.clone();
        (store, windows)
    }
}

fn main() {
    let (mut app, event_loop, _, _) =
        App::<(), BasicApp>::new::<()>((), Vec::new(), BasicApp {}, |_| ()).unwrap();

    event_loop.run_app(&mut app).unwrap();
}
