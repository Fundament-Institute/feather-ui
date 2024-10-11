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

pub struct Node<AppData> {
    pub area: AbsRect, // This is the calculated area of the node from the layout relative to the topleft corner of the parent.
    pub extent: AbsRect, // This is the minimal bounding rectangle of the children's extent relative to OUR topleft corner.
    pub top: i32, // 2D R-tree nodes are actually 3 dimensional, but the z-axis can never overlap (because layout rects have no depth).
    pub bottom: i32,
    pub id: std::rc::Weak<SourceID>,
    //transform: Rotor2, // TODO: build a 3D node where this is a 3D rotor and project it back on to a 2D plane.
    pub children: im::Vector<Option<Rc<Node<AppData>>>>,
}

impl<AppData> Node<AppData> {
    pub fn new(
        area: AbsRect,
        z: Option<i32>,
        children: im::Vector<Option<Rc<Node<AppData>>>>,
        id: std::rc::Weak<SourceID>,
    ) -> Self {
        let fold = VectorFold::new(
            |(rect, top, bottom): &(AbsRect, i32, i32),
             n: &Option<Rc<Node<AppData>>>|
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
        manager: &mut StateManager,
    ) -> Result<(), ()> {
        if self.area.contains(pos) {
            if let Some(id) = self.id.upgrade() {
                let state = manager.get_trait(&id).map_err(|_| ())?;
                let masks = state.input_masks();
                for (i, k) in masks.iter().enumerate() {
                    if (kind as u64 & *k) != 0 {
                        if manager
                            .process(event.clone().extract(), &crate::Slot(id.clone(), i as u64))
                            .is_ok()
                        {
                            return Ok(());
                        }
                    }
                }
            }

            pos -= self.area.topleft;
            // Children should be sorted from top to bottom
            for child in self.children.iter() {
                if child
                    .as_ref()
                    .unwrap()
                    .process(event, kind, pos, manager)
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
            RawEvent::Mouse { pos, .. } => self.process(event, RawEventKind::Mouse, *pos, manager),
            RawEvent::MouseMove { pos, .. } => {
                self.process(event, RawEventKind::MouseMove, *pos, manager)
            }
            RawEvent::MouseScroll { pos, .. } => {
                self.process(event, RawEventKind::MouseScroll, *pos, manager)
            }
            RawEvent::Touch { pos, .. } => {
                self.process(event, RawEventKind::Touch, pos.xy(), manager)
            }
            _ => Err(()),
        }
    }

    // After a node has been modified we might need to fix the parent's extent.
    /*pub fn fix_extent(&mut self) {
        if let Some(parent_ref) = self.parent.as_ref() {
            let mut parent = parent_ref.borrow_mut();

            let mut a = self.area.topleft;
            let mut b = self.area.bottomright;
            self.transform.rotate_vec(&mut a);
            self.transform.rotate_vec(&mut b);
            let aabb = build_aabb(a, b);

            if aabb.topleft.x < parent.extent.topleft.x || aabb.topleft.y < parent.extent.topleft.y
            {
                parent.extent.topleft.x = parent.extent.topleft.x.min(aabb.topleft.x);
                parent.extent.topleft.y = parent.extent.topleft.y.min(aabb.topleft.y);
            }

            if aabb.bottomright.x > parent.extent.bottomright.x
                || aabb.bottomright.y > parent.extent.bottomright.y
            {
                parent.extent.bottomright.x = parent.extent.bottomright.x.max(aabb.bottomright.x);
                parent.extent.bottomright.y = parent.extent.bottomright.y.max(aabb.bottomright.y);
            }
        }
    }*/
}
