// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use feather_ui::gen_id;
use feather_ui::layout::flex;
use feather_ui::layout::leaf;
use feather_ui::layout::simple;

use feather_ui::layout::base;
use feather_ui::outline::paragraph::Paragraph;
use feather_ui::outline::region::Region;
use feather_ui::outline::shape::Shape;
use feather_ui::outline::window::Window;
use feather_ui::outline::OutlineFrom;
use feather_ui::persist::FnPersist;
use feather_ui::AbsRect;
use feather_ui::App;
use feather_ui::SourceID;
use feather_ui::URect;
use std::f32;
use std::rc::Rc;
use ultraviolet::Vec2;
use ultraviolet::Vec4;

#[derive(PartialEq, Clone, Debug)]
struct Blocker {
    area: AbsRect,
}

struct BasicApp {}
struct MinimalFlexChild {
    area: URect,
}

impl flex::Child for MinimalFlexChild {
    fn grow(&self) -> f32 {
        0.0
    }

    fn shrink(&self) -> f32 {
        1.0
    }

    fn basis(&self) -> f32 {
        100.0
    }
}

impl base::Order for MinimalFlexChild {
    fn order(&self) -> i64 {
        0
    }
}

impl base::Margin for MinimalFlexChild {
    fn margin(&self) -> &URect {
        &feather_ui::ZERO_URECT
    }
}

impl base::Limits for MinimalFlexChild {
    fn limits(&self) -> &feather_ui::ULimits {
        &feather_ui::DEFAULT_LIMITS
    }
}

impl base::Area for MinimalFlexChild {
    fn area(&self) -> &URect {
        &self.area
    }
}

impl leaf::Prop for MinimalFlexChild {}

#[derive(Default, Clone)]
struct MinimalArea {
    area: URect,
}

impl base::Empty for MinimalArea {}

impl base::ZIndex for MinimalArea {
    fn zindex(&self) -> i32 {
        0
    }
}

impl base::Margin for MinimalArea {
    fn margin(&self) -> &URect {
        &feather_ui::ZERO_URECT
    }
}

impl base::Anchor for MinimalArea {
    fn anchor(&self) -> &feather_ui::UPoint {
        &feather_ui::ZERO_UPOINT
    }
}

impl base::Limits for MinimalArea {
    fn limits(&self) -> &feather_ui::ULimits {
        &feather_ui::DEFAULT_LIMITS
    }
}

impl base::Area for MinimalArea {
    fn area(&self) -> &URect {
        &self.area
    }
}

impl simple::Prop for MinimalArea {}

struct MinimalFlex {
    obstacles: Vec<AbsRect>,
}
impl base::Empty for MinimalFlex {}

impl base::ZIndex for MinimalFlex {
    fn zindex(&self) -> i32 {
        0
    }
}

impl base::Obstacles for MinimalFlex {
    fn obstacles(&self) -> &[AbsRect] {
        &self.obstacles
    }
}

impl flex::Prop for MinimalFlex {
    fn direction(&self) -> flex::FlexDirection {
        flex::FlexDirection::LeftToRight
    }

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
                let rect = Shape::round_rect(
                    gen_id!().into(),
                    MinimalFlexChild {
                        area: AbsRect::new(0.0, 0.0, 40.0, 40.0).into(),
                    }
                    .into(),
                    0.0,
                    0.0,
                    Vec4::broadcast(10.0),
                    Vec4::new(0.2, 0.7, 0.4, 1.0),
                    Vec4::zero(),
                );

                let mut p = Paragraph::new(
                    gen_id!().into(),
                    MinimalFlex {
                        obstacles: vec![AbsRect {
                            topleft: Vec2::new(200.0, 30.0),
                            bottomright: Vec2::new(300.0, 150.0),
                        }],
                    },
                );

                let text = "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?";
                p.set_text(
                    text,
                    30.0,
                    42.0,
                    glyphon::FamilyOwned::SansSerif,
                    glyphon::Color::rgba(255, 255, 255, 255),
                    Default::default(),
                    Default::default(),
                    true,
                );
                p.children.push_front(Some(Box::new(rect.clone())));
                p.children.push_back(Some(Box::new(rect.clone())));
                p.children.push_back(Some(Box::new(rect.clone())));

                p
            };

            let mut children: im::Vector<Option<Box<OutlineFrom<dyn simple::Prop>>>> =
                im::Vector::new();
            children.push_back(Some(Box::new(flex)));

            let region = Region {
                id: gen_id!().into(),
                props: MinimalArea {
                    area: feather_ui::URect {
                        topleft: feather_ui::UPoint {
                            abs: Vec2::new(90.0, 90.0),
                            rel: Vec2::new(0.0, 0.0).into(),
                        },
                        bottomright: feather_ui::UPoint {
                            abs: Vec2::new(-90.0, -90.0),
                            rel: Vec2::new(1.0, 1.0).into(),
                        },
                    },
                }
                .into(),
                children,
            };
            let window = Window::new(
                gen_id!().into(),
                winit::window::Window::default_attributes()
                    .with_title("paragraph-rs")
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
        )
        .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
