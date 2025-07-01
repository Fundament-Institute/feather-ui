// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use feather_macro::*;
use feather_ui::color::sRGB;
use feather_ui::component::button::Button;
use feather_ui::component::region::Region;
use feather_ui::component::shape::{Shape, ShapeKind};
use feather_ui::component::text::Text;
use feather_ui::component::window::Window;
use feather_ui::component::{ChildOf, mouse_area};
use feather_ui::layout::{fixed, leaf};
use feather_ui::persist::FnPersist;
use feather_ui::ultraviolet::{Vec2, Vec4};
use feather_ui::util::create_hotloader;
use feather_ui::{
    AbsRect, App, DAbsRect, DPoint, DRect, RelRect, Slot, SourceID, UNSIZED_AXIS, URect, ZERO_RECT,
    ZERO_RELRECT, gen_id, im, winit,
};
use std::rc::Rc;
use std::sync::RwLock;

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

impl FnPersist<CounterState, im::HashMap<Rc<SourceID>, Option<Window>>> for BasicApp {
    type Store = (CounterState, im::HashMap<Rc<SourceID>, Option<Window>>);

    fn init(&self) -> Self::Store {
        (CounterState { count: -1 }, im::HashMap::new())
    }
    fn call(
        &mut self,
        mut store: Self::Store,
        args: &CounterState,
    ) -> (Self::Store, im::HashMap<Rc<SourceID>, Option<Window>>) {
        if store.0 != *args {
            let button = {
                let text = Text::<FixedData> {
                    id: gen_id!(),
                    props: Rc::new(FixedData {
                        area: URect {
                            abs: AbsRect::new(8.0, 0.0, 8.0, 0.0),
                            rel: RelRect::new(0.0, 0.5, UNSIZED_AXIS, UNSIZED_AXIS),
                        }
                        .into(),
                        anchor: feather_ui::RelPoint(Vec2 { x: 0.0, y: 0.5 }).into(),
                        ..Default::default()
                    }),
                    text: format!("Clicks: {}", args.count),
                    font_size: 40.0,
                    line_height: 56.0,
                    ..Default::default()
                };

                let mut children: im::Vector<Option<Box<ChildOf<dyn fixed::Prop>>>> =
                    im::Vector::new();

                let rect = Shape::<DRect, { ShapeKind::RoundRect as u8 }>::new(
                    gen_id!(),
                    feather_ui::FILL_DRECT.into(),
                    0.0,
                    0.0,
                    Vec4::broadcast(10.0),
                    sRGB::new(0.2, 0.7, 0.4, 1.0),
                    sRGB::transparent(),
                );
                children.push_back(Some(Box::new(rect)));
                children.push_back(Some(Box::new(text)));

                Button::<FixedData>::new(
                    gen_id!(),
                    FixedData {
                        area: URect {
                            abs: AbsRect::new(45.0, 45.0, 0.0, 0.0),
                            rel: RelRect::new(0.0, 0.0, UNSIZED_AXIS, 1.0),
                        }
                        .into(),
                        ..Default::default()
                    },
                    Slot(feather_ui::APP_SOURCE_ID.into(), 0),
                    children,
                )
            };

            let unusedbutton = {
                let text = Text::<FixedData> {
                    id: gen_id!(),
                    props: Rc::new(FixedData {
                        area: RelRect::new(0.5, 0.0, UNSIZED_AXIS, UNSIZED_AXIS).into(),
                        limits: feather_ui::AbsLimits::new(
                            Vec2::new(f32::NEG_INFINITY, 10.0),
                            Vec2::new(f32::INFINITY, 200.0),
                        )
                        .into(),
                        rlimits: feather_ui::RelLimits::new(
                            Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
                            Vec2::new(1.0, f32::INFINITY),
                        ),
                        anchor: feather_ui::RelPoint(Vec2 { x: 0.5, y: 0.0 }).into(),
                        padding: AbsRect::new(8.0, 8.0, 8.0, 8.0).into(),
                        ..Default::default()
                    }),
                    text: (0..args.count).map(|_| "█").collect::<String>(),
                    font_size: 40.0,
                    line_height: 56.0,
                    wrap: feather_ui::cosmic_text::Wrap::WordOrGlyph,
                    ..Default::default()
                };

                let mut children: im::Vector<Option<Box<ChildOf<dyn fixed::Prop>>>> =
                    im::Vector::new();

                let rect = Shape::<DRect, { ShapeKind::RoundRect as u8 }>::new(
                    gen_id!(),
                    feather_ui::FILL_DRECT.into(),
                    0.0,
                    0.0,
                    Vec4::broadcast(10.0),
                    sRGB::new(0.7, 0.2, 0.4, 1.0),
                    sRGB::transparent(),
                );
                children.push_back(Some(Box::new(rect)));
                children.push_back(Some(Box::new(text)));

                Button::<FixedData>::new(
                    gen_id!(),
                    FixedData {
                        area: URect {
                            abs: AbsRect::new(45.0, 245.0, 0.0, 0.0),
                            rel: RelRect::new(0.0, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),
                        }
                        .into(),
                        limits: feather_ui::AbsLimits::new(
                            Vec2::new(100.0, f32::NEG_INFINITY),
                            Vec2::new(300.0, f32::INFINITY),
                        )
                        .into(),
                        ..Default::default()
                    },
                    Slot(feather_ui::APP_SOURCE_ID.into(), 0),
                    children,
                )
            };

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

            let mut children: im::Vector<Option<Box<ChildOf<dyn fixed::Prop>>>> =
                im::Vector::new();
            children.push_back(Some(Box::new(button)));
            children.push_back(Some(Box::new(unusedbutton)));
            children.push_back(Some(Box::new(pixel)));

            let region = Region::new(
                gen_id!(),
                FixedData {
                    area: URect {
                        abs: AbsRect::new(90.0, 90.0, 0.0, 200.0),
                        rel: RelRect::new(0.0, 0.0, UNSIZED_AXIS, 0.0),
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
static LOADERS: RwLock<Vec<feather_ui::notify::RecommendedWatcher>> = RwLock::new(Vec::new());

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
        |driver| {
            let exe = std::env::current_exe().unwrap();
            LOADERS.write().unwrap().push(
                    create_hotloader::<
                        feather_ui::render::shape::Shape<{ ShapeKind::RoundRect as u8 }>,
                    >(
                        &exe.join("../../../feather-ui/src/shaders/shape.wgsl"),
                        "Shape",
                        driver,
                    )
                    .unwrap(),
                );
        },
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
