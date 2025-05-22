// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::rc::Rc;

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

    pub fn process(
        self: &Rc<Self>,
        event: &RawEvent,
        kind: RawEventKind,
        mut pos: Vec2,
        mut offset: Vec2,
        dpi: Vec2,
        manager: &mut StateManager,
        window_id: Rc<SourceID>,
    ) -> Result<(), ()> {
        if self.area.contains(pos) {
            if let Ok(()) = self.inject_event(event, kind, dpi, offset, manager) {
                // If we successfully process a mouse event, this node gains focus in it's parent window
                match event {
                    RawEvent::Mouse { device_id, .. } | RawEvent::Touch { device_id, .. } => {
                        if match event {
                            RawEvent::Mouse { state, .. } => *state != MouseState::Up,
                            RawEvent::Touch { state, .. } => *state != TouchState::Start,
                            _ => false,
                        } {
                            let state: &mut WindowStateMachine =
                                manager.get_mut(&window_id).map_err(|_| ())?;
                            let window = state.state.as_mut().unwrap();

                            // Tell the old node that it lost focus (if it cares).
                            if let Some(old) = window
                                .focus
                                .insert(*device_id, crate::component::window::RcNode(self.clone()))
                            {
                                let evt = RawEvent::Focus { acquired: false };
                                // We don't care about the result of this event
                                let _ = old.0.inject_event(
                                    &evt,
                                    evt.kind(),
                                    dpi,
                                    Vec2::zero(), // Focus events shouldn't care about area offset
                                    manager,
                                );
                            }

                            let evt = RawEvent::Focus { acquired: true };
                            let _ = self.inject_event(&evt, evt.kind(), dpi, offset, manager);
                        }
                    }
                    _ => (),
                }
                return Ok(());
            }

            offset += self.area.topleft();
            pos -= self.area.topleft();
            // Children should be sorted from top to bottom
            for child in self.children.iter() {
                if child
                    .as_ref()
                    .unwrap()
                    .process(event, kind, pos, offset, dpi, manager, window_id.clone())
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
