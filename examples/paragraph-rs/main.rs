// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use feather_ui::color::sRGB;
use feather_ui::layout::{fixed, flex, leaf};
use feather_ui::{DAbsRect, DValue, gen_id};

use feather_ui::component::ComponentFrom;
use feather_ui::component::paragraph::Paragraph;
use feather_ui::component::region::Region;
use feather_ui::component::shape::{Shape, ShapeKind};
use feather_ui::component::window::Window;
use feather_ui::cosmic_text;
use feather_ui::layout::base;
use feather_ui::persist::FnPersist;
use feather_ui::ultraviolet::Vec4;
use feather_ui::{AbsRect, App, DRect, FILL_DRECT, RelRect, SourceID};
use std::f32;
use std::rc::Rc;

#[derive(PartialEq, Clone, Debug)]
struct Blocker {
    area: AbsRect,
}

struct BasicApp {}

#[derive(Default, Clone, feather_macro::Area)]
struct MinimalFlexChild {
    area: DRect,
}

impl flex::Child for MinimalFlexChild {
    fn grow(&self) -> f32 {
        0.0
    }

    fn shrink(&self) -> f32 {
        1.0
    }

    fn basis(&self) -> DValue {
        100.0.into()
    }
}

impl base::Order for MinimalFlexChild {}
impl base::Anchor for MinimalFlexChild {}
impl base::Padding for MinimalFlexChild {}
impl base::Margin for MinimalFlexChild {}
impl base::Limits for MinimalFlexChild {}
impl base::RLimits for MinimalFlexChild {}
impl leaf::Prop for MinimalFlexChild {}
impl leaf::Padded for MinimalFlexChild {}

#[derive(Default, Clone, feather_macro::Empty, feather_macro::Area)]
struct MinimalArea {
    area: DRect,
}

impl base::ZIndex for MinimalArea {}
impl base::Anchor for MinimalArea {}
impl base::Limits for MinimalArea {}
impl fixed::Prop for MinimalArea {}

#[derive(Default, Clone, feather_macro::Empty, feather_macro::Area)]
struct MinimalFlex {
    obstacles: Vec<DAbsRect>,
    area: DRect,
}
impl base::Direction for MinimalFlex {}
impl base::ZIndex for MinimalFlex {}
impl base::Limits for MinimalFlex {}
impl base::RLimits for MinimalFlex {}
impl fixed::Child for MinimalFlex {}

impl base::Obstacles for MinimalFlex {
    fn obstacles(&self) -> &[DAbsRect] {
        &self.obstacles
    }
}

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

impl FnPersist<Blocker, im::HashMap<Rc<SourceID>, Option<Window>>> for BasicApp {
    type Store = (Blocker, im::HashMap<Rc<SourceID>, Option<Window>>);

    fn init(&self) -> Self::Store {
        (
            Blocker {
                area: AbsRect::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN),
            },
            im::HashMap::new(),
        )
    }
    fn call(
        &self,
        mut store: Self::Store,
        args: &Blocker,
    ) -> (Self::Store, im::HashMap<Rc<SourceID>, Option<Window>>) {
        if store.0 != *args {
            let flex = {
                let rect = Shape::<MinimalFlexChild, { ShapeKind::RoundRect as u8 }>::new(
                    gen_id!(),
                    MinimalFlexChild {
                        area: AbsRect::new(0.0, 0.0, 40.0, 40.0).into(),
                    }
                    .into(),
                    0.0,
                    0.0,
                    Vec4::broadcast(10.0),
                    sRGB::new(0.2, 0.7, 0.4, 1.0),
                    sRGB::transparent(),
                );

                let mut p = Paragraph::new(
                    gen_id!(),
                    MinimalFlex {
                        area: FILL_DRECT,
                        obstacles: vec![AbsRect::new(200.0, 30.0, 300.0, 150.0).into()],
                    },
                );

                let text = "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?";
                p.set_text(
                    text,
                    40.0,
                    56.0,
                    cosmic_text::FamilyOwned::SansSerif,
                    sRGB::white(),
                    Default::default(),
                    Default::default(),
                    true,
                );
                p.prepend(Box::new(rect.clone()));
                p.append(Box::new(rect.clone()));
                p.append(Box::new(rect.clone()));

                p
            };

            let mut children: im::Vector<Option<Box<ComponentFrom<dyn fixed::Prop>>>> =
                im::Vector::new();
            children.push_back(Some(Box::new(flex)));

            let region = Region::new(
                gen_id!(),
                MinimalArea {
                    area: feather_ui::URect {
                        abs: AbsRect::new(90.0, 90.0, -90.0, -90.0),
                        rel: RelRect::new(0.0, 0.0, 1.0, 1.0),
                    }
                    .into(),
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

fn main() {
    let (mut app, event_loop): (App<Blocker, BasicApp>, winit::event_loop::EventLoop<()>) =
        App::new(
            Blocker {
                area: AbsRect::new(-1.0, -1.0, -1.0, -1.0),
            },
            vec![],
            BasicApp {},
            |_| (),
        )
        .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
