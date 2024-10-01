use std::rc::Rc;

use crate::input::EventKind;
use crate::layout::EventList;
use crate::persist::FnPersist2;
use crate::persist::VectorFold;
use crate::AbsRect;
use crate::Event;
use ultraviolet::Vec2;

pub struct Node<AppData> {
    pub area: AbsRect, // This is the calculated area of the node from the layout relative to the topleft corner of the parent.
    pub extent: AbsRect, // This is the minimal bounding rectangle of the children's extent relative to OUR topleft corner.
    pub top: i32, // 2D R-tree nodes are actually 3 dimensional, but the z-axis can never overlap (because layout rects have no depth).
    pub bottom: i32,
    pub events: Option<Rc<EventList<AppData>>>,
    //transform: Rotor2, // TODO: build a 3D node where this is a 3D rotor and project it back on to a 2D plane.
    pub children: im::Vector<Rc<Node<AppData>>>,
}

impl<AppData> Node<AppData> {
    pub fn new(
        area: AbsRect,
        z: Option<i32>,
        children: im::Vector<Rc<Node<AppData>>>,
        events: Option<Rc<EventList<AppData>>>,
    ) -> Self {
        let fold = VectorFold::new(
            |(rect, top, bottom): &(AbsRect, i32, i32),
             n: &Rc<Node<AppData>>|
             -> (AbsRect, i32, i32) {
                (
                    rect.extend(n.area),
                    (*top).max(n.top),
                    (*bottom).min(n.bottom),
                )
            },
        );

        // If no z index is provided for this node, try to use a zindex from the first child. If there is no first child, default to 0
        let z = z.unwrap_or_else(|| children.front().map(|x| x.top).unwrap_or(0));
        let (_, (extent, top, bottom)) =
            fold.call(Default::default(), &(Default::default(), z, z), &children);

        Self {
            area,
            extent,
            top,
            bottom,
            events,
            children,
        }
    }
    fn process(&self, event: Event, kind: EventKind, mut pos: Vec2, mut data: AppData) -> AppData {
        if self.area.contains(pos) {
            if let Some(events) = self.events.as_ref() {
                for (k, e) in events.iter() {
                    if (kind as u8 & *k) != 0 {
                        data = (e.borrow_mut())(event, self.area, data);
                    }
                }
            }

            pos -= self.area.topleft;
            // Children should be sorted from top to bottom
            for child in self.children.iter() {
                data = child.process(event, kind, pos, data);
            }
        }
        data
    }
    pub fn on_event(&self, event: Event, data: AppData) -> AppData {
        match event {
            Event::Mouse(_, pos, _, _, _) => self.process(event, EventKind::Mouse, pos, data),
            Event::MouseMove(_, pos, _, _) => self.process(event, EventKind::MouseMove, pos, data),
            Event::MouseScroll(pos, _, _) => self.process(event, EventKind::MouseScroll, pos, data),
            Event::Touch(_, pos, _, _, _, _) => {
                self.process(event, EventKind::Touch, pos.xy(), data)
            }
            _ => data,
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
