// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::rc::Rc;

use crate::input::RawEvent;
use crate::input::RawEventKind;
use crate::persist::FnPersist2;
use crate::persist::VectorFold;
use crate::AbsRect;
use crate::Dispatchable;
use crate::SourceID;
use crate::StateManager;
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
    fn process(
        &self,
        event: &RawEvent,
        kind: RawEventKind,
        mut pos: Vec2,
        mut offset: Vec2,
        manager: &mut StateManager,
    ) -> Result<(), ()> {
        if self.area.contains(pos) {
            if let Some(id) = self.id.upgrade() {
                if let Ok(state) = manager.get_trait(&id) {
                    let masks = state.input_masks();
                    for (i, k) in masks.iter().enumerate() {
                        if (kind as u64 & *k) != 0
                            && manager
                                .process(
                                    event.clone().extract(),
                                    &crate::Slot(id.clone(), i as u64),
                                    self.area + offset,
                                )
                                .is_ok()
                        {
                            return Ok(());
                        }
                    }
                }
            }

            offset += self.area.topleft();
            pos -= self.area.topleft();
            // Children should be sorted from top to bottom
            for child in self.children.iter() {
                if child
                    .as_ref()
                    .unwrap()
                    .process(event, kind, pos, offset, manager)
                    .is_ok()
                {
                    return Ok(());
                }
            }
        }
        Err(())
    }
    pub fn on_event(&self, event: &RawEvent, manager: &mut StateManager) -> Result<(), ()> {
        match event {
            RawEvent::Mouse { pos, .. } => {
                self.process(event, RawEventKind::Mouse, *pos, Vec2::zero(), manager)
            }
            RawEvent::MouseMove { pos, .. } => {
                self.process(event, RawEventKind::MouseMove, *pos, Vec2::zero(), manager)
            }
            RawEvent::MouseScroll { pos, .. } => self.process(
                event,
                RawEventKind::MouseScroll,
                *pos,
                Vec2::zero(),
                manager,
            ),
            RawEvent::Touch { pos, .. } => {
                self.process(event, RawEventKind::Touch, pos.xy(), Vec2::zero(), manager)
            }
            _ => Err(()),
        }
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
