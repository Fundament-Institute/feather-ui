// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use feather_ui::color::sRGB;
use feather_ui::{AbsPoint, AbsVector, ScopeID, gen_id};

use feather_ui::component::domain_line::DomainLine;
use feather_ui::component::domain_point::DomainPoint;
use feather_ui::component::mouse_area::MouseArea;
use feather_ui::component::region::Region;
use feather_ui::component::shape::{Shape, ShapeKind};
use feather_ui::component::window::Window;
use feather_ui::component::{ChildOf, mouse_area};
use feather_ui::input::MouseButton;
use feather_ui::layout::{base, fixed, leaf};
use feather_ui::persist::{FnPersist2, FnPersistStore};
use feather_ui::{
    AbsRect, App, CrossReferenceDomain, DRect, FILL_DRECT, Slot, SourceID, WrapEventEx, im,
};
use std::collections::HashSet;
use std::f32;
use std::sync::Arc;

#[derive(PartialEq, Clone, Debug, Default)]
struct GraphState {
    nodes: Vec<AbsPoint>,
    edges: HashSet<(usize, usize)>,
    offset: AbsVector,
    selected: Option<usize>,
}

struct BasicApp {}

#[derive(Default, Clone, feather_macro::Area)]
struct MinimalArea {
    area: DRect,
}

impl base::Empty for MinimalArea {}
impl base::ZIndex for MinimalArea {}
impl base::Margin for MinimalArea {}
impl base::Anchor for MinimalArea {}
impl base::Limits for MinimalArea {}
impl base::RLimits for MinimalArea {}
impl fixed::Prop for MinimalArea {}
impl fixed::Child for MinimalArea {}
impl leaf::Prop for MinimalArea {}

const NODE_RADIUS: f32 = 25.0;

impl FnPersistStore for BasicApp {
    type Store = (GraphState, im::HashMap<Arc<SourceID>, Option<Window>>);
}

impl FnPersist2<GraphState, ScopeID<'_>, im::HashMap<Arc<SourceID>, Option<Window>>> for BasicApp {
    fn init(&self) -> Self::Store {
        (Default::default(), im::HashMap::new())
    }
    fn call(
        &mut self,
        mut store: Self::Store,
        args: GraphState,
        mut scope: ScopeID<'_>,
    ) -> (Self::Store, im::HashMap<Arc<SourceID>, Option<Window>>) {
        if store.0 != args {
            let mut children: im::Vector<Option<Box<ChildOf<dyn fixed::Prop>>>> = im::Vector::new();
            let domain: Arc<CrossReferenceDomain> = Default::default();

            let mut node_ids: Vec<Arc<SourceID>> = Vec::new();

            for (i, id) in scope.iter(0..args.nodes.len()) {
                let node = args.nodes[i];
                const BASE: sRGB = sRGB::new(0.2, 0.7, 0.4, 1.0);

                let point = DomainPoint::new(id, domain.clone());
                node_ids.push(point.id.clone());

                let circle = Shape::<DRect, { ShapeKind::Circle as u8 }>::new(
                    gen_id!(point.id),
                    FILL_DRECT,
                    0.0,
                    0.0,
                    [0.0, 20.0],
                    if args.selected == Some(i) {
                        sRGB::new(0.7, 1.0, 0.8, 1.0)
                    } else {
                        BASE
                    },
                    BASE,
                );

                let bag = Region::<MinimalArea>::new(
                    gen_id!(point.id),
                    MinimalArea {
                        area: AbsRect::new(
                            node.x - NODE_RADIUS,
                            node.y - NODE_RADIUS,
                            node.x + NODE_RADIUS,
                            node.y + NODE_RADIUS,
                        )
                        .into(),
                    },
                    feather_ui::children![fixed::Prop, point, circle],
                );

                children.push_back(Some(Box::new(bag)));
            }

            for ((a, b), id) in scope.iter(&args.edges) {
                let line = DomainLine::<()> {
                    id,
                    fill: sRGB::white(),
                    domain: domain.clone(),
                    start: node_ids[*a].clone(),
                    end: node_ids[*b].clone(),
                    props: ().into(),
                };

                children.push_back(Some(Box::new(line)));
            }

            let subregion = Region::new(
                gen_id!(scope),
                MinimalArea {
                    area: AbsRect::new(
                        args.offset.x,
                        args.offset.y,
                        args.offset.x + 10000.0,
                        args.offset.y + 10000.0,
                    )
                    .into(),
                },
                children,
            );

            let mousearea: MouseArea<MinimalArea> = MouseArea::new(
                gen_id!(scope),
                MinimalArea { area: FILL_DRECT },
                Some(4.0),
                [
                    Some(Slot(feather_ui::APP_SOURCE_ID.into(), 0)),
                    Some(Slot(feather_ui::APP_SOURCE_ID.into(), 0)),
                    Some(Slot(feather_ui::APP_SOURCE_ID.into(), 0)),
                    None,
                    None,
                    None,
                ],
            );

            let region = Region::new(
                gen_id!(scope),
                MinimalArea { area: FILL_DRECT },
                feather_ui::children![fixed::Prop, subregion, mousearea],
            );

            let window = Window::new(
                gen_id!(scope),
                feather_ui::winit::window::Window::default_attributes()
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
    let handle_input = Box::new(
        |e: mouse_area::MouseAreaEvent,
         mut appdata: GraphState|
         -> Result<GraphState, GraphState> {
            match e {
                mouse_area::MouseAreaEvent::OnClick(MouseButton::Left, pos) => {
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
                mouse_area::MouseAreaEvent::OnDblClick(MouseButton::Left, pos) => {
                    // TODO: winit currently doesn't capture double clicks
                    appdata.nodes.push(pos - appdata.offset);
                    Ok(appdata)
                }
                mouse_area::MouseAreaEvent::OnDrag(MouseButton::Left, diff) => {
                    appdata.offset += diff;
                    Ok(appdata)
                }
                _ => Ok(appdata),
            }
        }
        .wrap(),
    );

    let (mut app, event_loop, _, _) = App::<GraphState, BasicApp>::new::<()>(
        GraphState {
            nodes: vec![],
            edges: HashSet::new(),
            offset: AbsVector::new(-5000.0, -5000.0),
            selected: None,
        },
        vec![handle_input],
        BasicApp {},
        |_| (),
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
