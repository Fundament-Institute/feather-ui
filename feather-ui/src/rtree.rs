// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::rc::Rc;

use crate::component::window::RcNode;
use crate::input::{MouseState, RawEvent, RawEventKind, TouchState};
use crate::persist::{FnPersist2, VectorFold};
use crate::{AbsRect, Dispatchable, SourceID, StateManager, WindowStateMachine};
use eyre::Result;
use ultraviolet::Vec2;

pub struct Node {
    pub area: AbsRect, // This is the calculated area of the node from the layout relative to the topleft corner of the parent.
    pub extent: AbsRect, // This is the minimal bounding rectangle of the children's extent relative to OUR topleft corner.
    pub top: i32, // 2D R-tree nodes are actually 3 dimensional, but the z-axis can never overlap (because layout rects have no depth).
    pub bottom: i32,
    pub id: std::rc::Weak<SourceID>,
    pub children: im::Vector<Option<Rc<Node>>>,
}

pub struct ParentTuple<'a>(&'a Rc<Node>, Option<&'a ParentTuple<'a>>);

impl Node {
    pub fn new(
        area: AbsRect,
        z: Option<i32>,
        children: im::Vector<Option<Rc<Node>>>,
        id: std::rc::Weak<SourceID>,
    ) -> Self {
        let fold = VectorFold::new(
            |(rect, top, bottom): &(AbsRect, i32, i32),
             n: &Option<Rc<Node>>|
             -> (AbsRect, i32, i32) {
                let n = n.as_ref().unwrap();
                (
                    rect.extend(n.area),
                    (*top).max(n.top),
                    (*bottom).min(n.bottom),
                )
            },
        );

        // If no z index is provided for this node, try to use a zindex from the first child. If there is no first child, default to 0
        let z = z.unwrap_or_else(|| {
            children
                .front()
                .map(|x| x.as_ref().unwrap().top)
                .unwrap_or(0)
        });
        let (_, (extent, top, bottom)) =
            fold.call(fold.init(), &(Default::default(), z, z), &children);

        Self {
            area,
            extent,
            top,
            bottom,
            id,
            children,
        }
    }

    pub(crate) fn inject_event(
        self: &Rc<Self>,
        event: &RawEvent,
        kind: RawEventKind,
        dpi: Vec2,
        offset: Vec2,
        manager: &mut StateManager,
    ) -> Result<(), ()> {
        if let Some(id) = self.id.upgrade() {
            if let Ok(state) = manager.get_trait(&id) {
                let masks = state.input_masks();
                for (i, k) in masks.iter().enumerate() {
                    if (kind as u64 & *k) != 0
                        && manager
                            .process(
                                event.clone().extract(),
                                &crate::Slot(id.clone(), i as u64),
                                dpi,
                                self.area + offset,
                            )
                            .is_ok()
                    {
                        return Ok(());
                    }
                }
            }
        }
        Err(())
    }

    pub(crate) fn offset(nodes: &[RcNode]) -> Vec2 {
        let mut offset = Vec2::zero();
        for node in nodes {
            offset += node.0.area.topleft();
        }
        offset
    }

    pub(crate) fn assemble_node_chain(mut parent: &ParentTuple, nodes: &mut Vec<RcNode>) {
        nodes.push(RcNode(parent.0.clone()));

        while let Some(n) = parent.1 {
            parent = n;
            nodes.push(RcNode(parent.0.clone()));
        }
    }

    pub fn process(
        self: &Rc<Self>,
        event: &RawEvent,
        kind: RawEventKind,
        position: Vec2,
        mut offset: Vec2,
        dpi: Vec2,
        manager: &mut StateManager,
        window_id: Rc<SourceID>,
        parent: Option<&ParentTuple>,
    ) -> Result<(), ()> {
        let parent = ParentTuple(self, parent);
        if self.area.contains(position - offset) {
            if let Ok(()) = self.inject_event(event, kind, dpi, offset, manager) {
                match event {
                    // If we successfully process a mousemove event, this node gains hover
                    RawEvent::MouseMove {
                        device_id,
                        pos,
                        modifiers,
                        all_buttons,
                        driver,
                    } => {
                        let state: &mut WindowStateMachine =
                            manager.get_mut(&window_id).map_err(|_| ())?;
                        let window = state.state.as_mut().unwrap();

                        // Tell the old node that it lost hover (if it cares).
                        if let Some(old) = window.hover.insert(*device_id, RcNode(self.clone())) {
                            let evt = RawEvent::MouseOff {
                                device_id: *device_id,
                                modifiers: *modifiers,
                                all_buttons: *all_buttons,
                                driver: driver.clone(),
                            };

                            // We don't care about the result of this event
                            let _ =
                                old.0
                                    .inject_event(&evt, evt.kind(), dpi, Vec2::zero(), manager);
                        }
                        let evt = RawEvent::MouseOn {
                            device_id: *device_id,
                            modifiers: *modifiers,
                            pos: *pos,
                            all_buttons: *all_buttons,
                            driver: driver.clone(),
                        };
                        let _ = self.inject_event(&evt, evt.kind(), dpi, offset, manager);
                    }
                    // If we successfully process a mouse event, this node gains focus in it's parent window
                    RawEvent::Mouse { device_id, .. } | RawEvent::Touch { device_id, .. } => {
                        let isdown = match event {
                            RawEvent::Mouse { state, .. } => Some(*state != MouseState::Up),
                            RawEvent::Touch {
                                state: TouchState::Start,
                                ..
                            } => Some(false),
                            RawEvent::Touch {
                                state: TouchState::Move,
                                ..
                            } => None,
                            RawEvent::Touch {
                                state: TouchState::End,
                                ..
                            } => Some(true),
                            _ => None,
                        };

                        let state: &mut WindowStateMachine =
                            manager.get_mut(&window_id).map_err(|_| ())?;
                        let window = state.state.as_mut().unwrap();
                        let inner = window.window.clone();

                        // On any mousedown event, capture the cursor if it wasn't captured already
                        if let Some(true) = isdown {
                            window
                                .capture
                                .entry(*device_id)
                                .and_modify(|v| Self::assemble_node_chain(&parent, v))
                                .or_insert_with(|| {
                                    let mut v = Vec::new();
                                    Self::assemble_node_chain(&parent, &mut v);
                                    let (_, list) = v.split_first().unwrap();
                                    assert_eq!(Self::offset(list), offset);
                                    v
                                });
                        }
                        if let Some(false) = isdown {
                            // On any mouseup event, uncapture the cursor if no buttons are down
                            match event {
                                RawEvent::Mouse { all_buttons: 0, .. } | RawEvent::Touch { .. } => {
                                    if let Some(v) = window.capture.get_mut(device_id) {
                                        v.clear();
                                    }
                                }
                                _ => (),
                            }
                        }

                        if let Some(true) = isdown {
                            // Tell the old node that it lost focus (if it cares).
                            if let Some(old) = window.focus.insert(*device_id, RcNode(self.clone()))
                            {
                                let evt = RawEvent::Focus {
                                    acquired: false,
                                    window: inner.clone(),
                                };

                                // We don't care about the result of this event
                                let _ = old.0.inject_event(
                                    &evt,
                                    evt.kind(),
                                    dpi,
                                    Vec2::zero(),
                                    manager,
                                );
                            }

                            let evt = RawEvent::Focus {
                                acquired: true,
                                window: inner,
                            };
                            let _ = self.inject_event(&evt, evt.kind(), dpi, offset, manager);
                        }
                    }
                    _ => (),
                }
                return Ok(());
            }

            offset += self.area.topleft();
            // Children should be sorted from top to bottom
            for child in self.children.iter() {
                if child
                    .as_ref()
                    .unwrap()
                    .process(
                        event,
                        kind,
                        position,
                        offset,
                        dpi,
                        manager,
                        window_id.clone(),
                        Some(&parent),
                    )
                    .is_ok()
                {
                    // At this point, we should've already set focus, and are simply walking back up the stack
                    return Ok(());
                }
            }
        }
        Err(())
    }
}

/*
// 2.5D node which contains a 2D r-tree, embedded inside the parent 3D space.
struct Node25 {
    pub area: AbsRect,
    pub extent: AbsRect,
    pub z: f32, // there is only one z coordinate because the contained area must be flat.
    pub transform: Rotor3,
    pub id: std::rc::Weak<SourceID>,
    pub children: im::Vector<Option<Rc<Node>>>,
}

// 3D node capable of arbitrary translation (though it's AABB must still be fully contained within it's parent node)[]
struct Node3D {
    pub area: AbsVolume,
    pub extent: AbsVolume,
    pub transform: Rotor3,
    pub id: std::rc::Weak<SourceID>,
    pub children: im::Vector<Either<Rc<Node3D>, Rc<Node25>>>,
}
*/
