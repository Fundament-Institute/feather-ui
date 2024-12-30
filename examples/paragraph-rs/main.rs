// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2024 Fundament Software SPC <https://fundament.software>

use feather_ui::gen_id;
use feather_ui::layout::basic::Basic;
use feather_ui::layout::flex;
use feather_ui::layout::flex::Flex;

use feather_ui::layout::root;
use feather_ui::outline::flexbox::FlexBox;
use feather_ui::outline::region::Region;
use feather_ui::outline::round_rect::RoundRect;
use feather_ui::outline::text::Text;
use feather_ui::outline::window::Window;
use feather_ui::outline::OutlineFrom;
use feather_ui::persist::FnPersist;
use feather_ui::AbsRect;
use feather_ui::App;
use feather_ui::SourceID;
use std::default;
use std::f32;
use std::rc::Rc;
use ultraviolet::Vec2;
use ultraviolet::Vec4;

#[derive(PartialEq, Clone, Debug)]
struct Blocker {
    area: AbsRect,
}

struct BasicApp {}

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
                let rect = RoundRect::<flex::Inherited> {
                    id: gen_id!().into(),
                    fill: Vec4::new(0.2, 0.7, 0.4, 1.0),
                    corners: Vec4::broadcast(10.0),
                    props: flex::Inherited {
                        margin: Default::default(),
                        limits: feather_ui::DEFAULT_LIMITS,
                        order: 0,
                        grow: 0.0,
                        shrink: 1.0,
                        basis: 100.0,
                    },
                    rect: AbsRect::new(0.0, 0.0, 40.0, 40.0).into(),
                    ..Default::default()
                };

                let mut children: im::Vector<Option<Box<OutlineFrom<Flex>>>> = im::Vector::new();
                children.push_back(Some(Box::new(rect.clone())));

                let paragraph = "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?".split(' ');
                for word in paragraph {
                    let text = Text::<flex::Inherited> {
                        id: gen_id!().into(),
                        props: flex::Inherited {
                            margin: Default::default(),
                            limits: feather_ui::DEFAULT_LIMITS,
                            order: 0,
                            grow: 1.0,
                            shrink: 0.0,
                            basis: f32::INFINITY,
                        },
                        text: word.to_owned() + " ",
                        font_size: 30.0,
                        line_height: 42.0,
                        ..Default::default()
                    };
                    children.push_back(Some(Box::new(text)));
                }

                children.push_back(Some(Box::new(rect.clone())));
                children.push_back(Some(Box::new(rect.clone())));

                FlexBox {
                    id: gen_id!().into(),
                    props: (),
                    flex: Flex {
                        zindex: 0,
                        direction: flex::FlexDirection::LeftToRight,
                        wrap: true,
                        justify: flex::FlexJustify::Start,
                        align: flex::FlexJustify::Start,
                        obstacles: Default::default(),
                    },
                    children,
                }
            };

            let mut children: im::Vector<Option<Box<OutlineFrom<Basic>>>> = im::Vector::new();
            children.push_back(Some(Box::new(flex)));

            let region = Region {
                id: gen_id!().into(),
                props: root::Inherited {
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
                },
                basic: Basic {
                    padding: Default::default(),
                    zindex: 0,
                },
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

use feather_ui::WrapEventEx;

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
