// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use feather_macro::*;
use feather_ui::color::sRGB;
use feather_ui::component::image::Image;
use feather_ui::component::mouse_area;
use feather_ui::component::region::Region;
use feather_ui::component::shape::{Shape, ShapeKind};
use feather_ui::component::window::Window;
use feather_ui::layout::{fixed, leaf};
use feather_ui::persist::FnPersist;
use feather_ui::ultraviolet::{Vec2, Vec4};
use feather_ui::{
    AbsRect, App, DAbsRect, DPoint, DRect, RelRect, SourceID, UNSIZED_AXIS, URect, ZERO_RECT,
    ZERO_RELRECT, gen_id, im, winit,
};
use std::path::PathBuf;
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
            let pixel = Shape::<DRect, { ShapeKind::RoundRect as u8 }>::new(
                gen_id!(),
                Rc::new(DRect {
                    px: AbsRect::new(1.0, 1.0, 2.0, 2.0),
                    dp: ZERO_RECT,
                    rel: ZERO_RELRECT,
                }),
                0.0,
                0.0,
                Vec4::broadcast(0.0),
                sRGB::new(1.0, 1.0, 1.0, 1.0),
                sRGB::transparent(),
            );

            let mut children: im::Vector<
                Option<Box<feather_ui::component::ChildOf<dyn fixed::Prop>>>,
            > = im::Vector::new();
            children.push_back(Some(Box::new(pixel)));

            let genimage = |id: Arc<SourceID>,
                            pos: Vec2,
                            w: Option<f32>,
                            h: Option<f32>,
                            res: &dyn feather_ui::resource::Location,
                            size: Option<Vec2>| {
                Image::<DRect>::new(
                    id,
                    Rc::new(DRect {
                        px: ZERO_RECT,
                        dp: AbsRect::new(
                            pos.x,
                            pos.y,
                            w.map(|x| x + pos.x).unwrap_or_default(),
                            h.map(|y| y + pos.y).unwrap_or_default(),
                        ),
                        rel: RelRect::new(
                            0.0,
                            0.0,
                            if w.is_none() { UNSIZED_AXIS } else { 0.0 },
                            if h.is_none() { UNSIZED_AXIS } else { 0.0 },
                        ),
                    }),
                    res,
                    size.unwrap_or_default().into(),
                    false,
                )
            };

            #[cfg(feature = "png")]
            {
                let testimage = PathBuf::from("./premul_test.png");

                children.push_back(Some(Box::new(genimage(
                    gen_id!(),
                    Vec2::new(0.0, 0.0),
                    Some(100.0),
                    Some(100.0),
                    &testimage,
                    None,
                ))));

                children.push_back(Some(Box::new(genimage(
                    gen_id!(),
                    Vec2::new(100.0, 0.0),
                    None,
                    Some(100.0),
                    &testimage,
                    None,
                ))));

                children.push_back(Some(Box::new(genimage(
                    gen_id!(),
                    Vec2::new(0.0, 100.0),
                    None,
                    None,
                    &testimage,
                    Some(Vec2::broadcast(100.0)),
                ))));

                children.push_back(Some(Box::new(genimage(
                    gen_id!(),
                    Vec2::new(100.0, 100.0),
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
                    gen_id!(),
                    Vec2::new(200.0, 0.0),
                    Some(100.0),
                    Some(100.0),
                    &testsvg,
                    None,
                ))));

                children.push_back(Some(Box::new(genimage(
                    gen_id!(),
                    Vec2::new(300.0, 0.0),
                    None,
                    Some(100.0),
                    &testsvg,
                    None,
                ))));

                children.push_back(Some(Box::new(genimage(
                    gen_id!(),
                    Vec2::new(200.0, 100.0),
                    None,
                    None,
                    &testsvg,
                    Some(Vec2::broadcast(100.0)),
                ))));

                children.push_back(Some(Box::new(genimage(
                    gen_id!(),
                    Vec2::new(300.0, 100.0),
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
                    gen_id!(),
                    Vec2::new(0.0, 200.0),
                    Some(100.0),
                    Some(100.0),
                    &testimage,
                    None,
                ))));

                children.push_back(Some(Box::new(genimage(
                    gen_id!(),
                    Vec2::new(100.0, 200.0),
                    Some(100.0),
                    None,
                    &testimage,
                    None,
                ))));

                children.push_back(Some(Box::new(genimage(
                    gen_id!(),
                    Vec2::new(0.0, 300.0),
                    None,
                    None,
                    &testimage,
                    Some(Vec2::broadcast(100.0)),
                ))));

                children.push_back(Some(Box::new(genimage(
                    gen_id!(),
                    Vec2::new(100.0, 300.0),
                    None,
                    None,
                    &testimage,
                    None,
                ))));
            }

            let region = Region::new(
                gen_id!(),
                FixedData {
                    area: URect {
                        abs: AbsRect::new(10.0, 10.0, -10.0, -10.0),
                        rel: RelRect::new(0.0, 0.0, 1.0, 1.0),
                    }
                    .into(),
                    zindex: 0,
                    ..Default::default()
                }
                .into(),
                children,
            );

            let window = Window::new(
                gen_id!(),
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
    ) = App::new(
        CounterState { count: 0 },
        vec![onclick],
        BasicApp {},
        |_| (),
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
