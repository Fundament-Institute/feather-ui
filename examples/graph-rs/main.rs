// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use feather_ui::gen_id;

use feather_ui::layout::root;
use feather_ui::layout::simple;
use feather_ui::outline::domain_line::DomainLine;
use feather_ui::outline::domain_point::DomainPoint;
use feather_ui::outline::draggable;
use feather_ui::outline::draggable::Draggable;
use feather_ui::outline::shape;
use feather_ui::outline::sized_region::SizedRegion;
use feather_ui::outline::window::Window;
use feather_ui::outline::CrossReferenceDomain;
use feather_ui::outline::OutlineFrom;
use feather_ui::persist::FnPersist;
use feather_ui::App;
use feather_ui::DataID;
use feather_ui::Slot;
use feather_ui::SourceID;
use feather_ui::WrapEventEx;
use feather_ui::FILL_URECT;
use std::collections::HashSet;
use std::f32;
use std::rc::Rc;
use ultraviolet::Vec2;
use ultraviolet::Vec4;

#[derive(PartialEq, Clone, Debug)]
struct GraphState {
    nodes: Vec<Vec2>,
    edges: HashSet<(usize, usize)>,
    offset: Vec2,
    selected: Option<usize>,
}

struct BasicApp {}

const NODE_RADIUS: f32 = 25.0;

impl FnPersist<GraphState, im::HashMap<Rc<SourceID>, Option<Window>>> for BasicApp {
    type Store = (GraphState, im::HashMap<Rc<SourceID>, Option<Window>>);

    fn init(&self) -> Self::Store {
        (
            GraphState {
                nodes: Vec::new(),
                edges: HashSet::new(),
                offset: Vec2::zero(),
                selected: None,
            },
            im::HashMap::new(),
        )
    }
    fn call(
        &self,
        mut store: Self::Store,
        args: &GraphState,
    ) -> (Self::Store, im::HashMap<Rc<SourceID>, Option<Window>>) {
        if store.0 != *args {
            let mut children: im::Vector<Option<Box<OutlineFrom<Basic>>>> = im::Vector::new();
            let domain: Rc<CrossReferenceDomain> = Default::default();

            let mut node_ids: Vec<Rc<SourceID>> = Vec::new();

            let node_id = Rc::new(gen_id!());

            for i in 0..args.nodes.len() {
                let node = args.nodes[i];
                let base = Vec4::new(0.2, 0.7, 0.4, 1.0);

                let iter_id = Rc::new(node_id.child(DataID::Int(i as i64)));
                node_ids.push(iter_id.clone());

                let mut contents: im::Vector<Option<Box<OutlineFrom<Simple>>>> = im::Vector::new();

                let point = DomainPoint::<()>::new(iter_id.clone(), (), domain.clone());

                let circle = Circle::<()> {
                    id: gen_id!(iter_id).into(),
                    fill: if args.selected == Some(i) {
                        Vec4::new(0.7, 1.0, 0.8, 1.0)
                    } else {
                        base
                    },
                    outline: base,
                    radii: Vec2::new(0.0, 20.0),
                    props: (),
                    rect: FILL_URECT,
                    ..Default::default()
                };

                contents.push_back(Some(Box::new(point)));
                contents.push_back(Some(Box::new(circle)));

                let bag = SizedRegion::<()> {
                    id: gen_id!(iter_id).into(),
                    props: (),
                    simple: Simple {
                        margin: Default::default(),
                        anchor: Default::default(),
                        limits: Default::default(),
                        zindex: 0,
                        area: feather_ui::AbsRect::new(
                            node.x - NODE_RADIUS,
                            node.y - NODE_RADIUS,
                            node.x + NODE_RADIUS,
                            node.y + NODE_RADIUS,
                        )
                        .into(),
                    },
                    children: contents,
                };

                children.push_back(Some(Box::new(bag)));
            }

            let edge_id = Rc::new(gen_id!());

            for (a, b) in &args.edges {
                let line = DomainLine::<()> {
                    id: Rc::new(edge_id.child(DataID::Int(*a as i64)))
                        .child(DataID::Int(*b as i64))
                        .into(),
                    fill: Vec4::broadcast(1.0),
                    domain: domain.clone(),
                    start: node_ids[*a].clone(),
                    end: node_ids[*b].clone(),
                    props: (),
                };

                children.push_back(Some(Box::new(line)));
            }

            let region = Draggable {
                id: gen_id!().into(),
                props: root::Child {
                    area: feather_ui::URect {
                        topleft: feather_ui::UPoint {
                            abs: args.offset + Vec2::new(0.0, 0.0),
                            rel: Vec2::new(0.0, 0.0).into(),
                        },
                        bottomright: feather_ui::UPoint {
                            abs: args.offset + Vec2::new(10000.0, 10000.0),
                            rel: Vec2::new(0.0, 0.0).into(),
                        },
                    },
                },
                basic: Basic {
                    padding: Default::default(),
                    zindex: 0,
                },
                children,
                slots: [
                    Some(Slot(feather_ui::APP_SOURCE_ID.into(), 0)),
                    Some(Slot(feather_ui::APP_SOURCE_ID.into(), 0)),
                    Some(Slot(feather_ui::APP_SOURCE_ID.into(), 0)),
                ],
            };
            let window = Window::new(
                gen_id!().into(),
                winit::window::Window::default_attributes()
                    .with_title("graph-rs")
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
    let handle_input = Box::new(
        |e: draggable::DraggableEvent, mut appdata: GraphState| -> Result<GraphState, GraphState> {
            match e {
                draggable::DraggableEvent::OnClick(pos) => {
                    if let Some(selected) = appdata.selected {
                        for i in 0..appdata.nodes.len() {
                            let diff = appdata.nodes[i] - pos + appdata.offset;
                            if diff.dot(diff) < NODE_RADIUS * NODE_RADIUS {
                                if appdata.edges.contains(&(selected, i)) {
                                    appdata.edges.remove(&(i, selected));
                                    appdata.edges.remove(&(selected, i));
                                } else {
                                    appdata.edges.insert((selected, i));
                                    appdata.edges.insert((i, selected));
                                }
                                break;
                            }
                        }

                        appdata.selected = None;
                    } else {
                        // Check to see if we're anywhere near a node (yes this is inefficient but we don't care right now)
                        for i in 0..appdata.nodes.len() {
                            let diff = appdata.nodes[i] - pos + appdata.offset;
                            if diff.dot(diff) < NODE_RADIUS * NODE_RADIUS {
                                appdata.selected = Some(i);
                                return Ok(appdata);
                            }
                        }

                        // TODO: maybe make this require shift click
                        appdata.nodes.push(pos - appdata.offset);
                    }

                    Ok(appdata)
                }
                draggable::DraggableEvent::OnDblClick(pos) => {
                    // TODO: winit currently doesn't capture double clicks
                    appdata.nodes.push(pos - appdata.offset);
                    Ok(appdata)
                }
                draggable::DraggableEvent::OnDrag(diff) => {
                    appdata.offset += diff;
                    Ok(appdata)
                }
            }
        }
        .wrap(),
    );

    let (mut app, event_loop): (App<GraphState, BasicApp>, winit::event_loop::EventLoop<()>) =
        App::new(
            GraphState {
                nodes: vec![],
                edges: HashSet::new(),
                offset: Vec2::new(-5000.0, -5000.0),
                selected: None,
            },
            vec![handle_input],
            BasicApp {},
        )
        .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
