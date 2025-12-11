// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::component::window::WindowNodeTrack;
use crate::input::{MouseState, RawEvent, RawEventKind, TouchState};
use crate::layout::Staged;
use crate::reactive::{
    self, DynSignal, Identity, ToSignal, const_signal, fold_vec, map_vec, zip_pair,
};
use crate::render::compositor::Layer;
use crate::{Pixel, PxPoint, PxRect, PxVector, RelDim};
use guillotiere::euclid::Point3D;
use std::rc::Rc;
use winit::dpi::PhysicalPosition;

pub struct Node {
    pub area: DynSignal<PxRect>, /* This is the calculated area of the node from the layout relative to the
                                  * topleft corner of the parent. */
    pub extent: DynSignal<PxRect>, /* This is the minimal bounding rectangle of the children's extent
                                    * relative to OUR topleft corner. */
    pub top: DynSignal<i32>,
    pub depth: DynSignal<i32>, // Positive or negative range starting from the top that this node encompasses. Must be 0 if staged is non-null.
    pub mask: AtomicU64,
    pub children: Option<DynSignal<imbl::Vector<Rc<Node>>>>,
    pub staged: Option<Box<dyn Staged>>,
    pub callback: RefCell<Option<Box<dyn crate::event::StreamCallback<RawEvent>>>>,
}

// A tuple like this is necessary to build a chain of parent nodes down the
// recursive process call, but we currently don't need it. This is left here as
// reference.

pub struct ParentTuple<'a>(&'a Rc<Node>, Option<&'a ParentTuple<'a>>);

impl<'a> ParentTuple<'a> {
    pub fn root(&'a self) -> &'a Rc<Node> {
        if let Some(n) = self.1 {
            n.root()
        } else {
            self.0
        }
    }
}

impl Node {
    pub fn new(
        area: DynSignal<PxRect>,
        z: Option<DynSignal<i32>>,
        children: Option<DynSignal<imbl::Vector<Rc<Node>>>>,
        staged: Option<Box<dyn Staged>>,
    ) -> Self {
        let z = z.unwrap_or_else(|| 0.to_signal().into_dyn_signal());
        if let Some(children) = children {
            let areas = map_vec(
                |v| v.area.clone(),
                |v| Identity(v.clone()),
                children.clone(),
            );

            let extent = reactive::join(fold_vec(
                |l, r| zip_pair(l, r, |x, y| x.extend(y)).into(),
                areas,
                const_signal(PxRect::zero()).into(),
            ))
            .into_dyn_signal();

            Self {
                area,
                extent,
                top: z,
                depth: 0.to_signal().into(),
                children: Some(children),
                mask: u64::MAX.into(),
                staged,
                callback: None.into(),
            }
        } else {
            Self {
                extent: area.clone(),
                area,
                top: z,
                depth: 0.to_signal().into(),
                children: None,
                mask: u64::MAX.into(),
                staged,
                callback: None.into(),
            }
        }
    }

    pub fn new_subtree(children: DynSignal<imbl::Vector<Rc<Node>>>) -> Self {
        let areas = map_vec(
            |v| v.area.clone(),
            |v| Identity(v.clone()),
            children.clone(),
        );

        let extent = reactive::join(fold_vec(
            |l, r| zip_pair(l, r, |x, y| x.extend(y)).into(),
            areas,
            const_signal(PxRect::zero()).into(),
        ))
        .into_dyn_signal();

        let tops = map_vec(|v| v.top.clone(), |v| Identity(v.clone()), children.clone());

        let top = reactive::join(fold_vec(
            |l, r| zip_pair(l, r, |x, y| x.min(y)).into(),
            tops,
            0.to_signal().into_dyn_signal(),
        ));

        let bottoms = map_vec(
            |v| (v.top.clone() + v.depth.clone()).into_dyn_signal(),
            |v| Identity(v.clone()),
            children.clone(),
        );

        let bottom = reactive::join(fold_vec(
            |l, r| zip_pair(l, r, |x, y| x.max(y)).into(),
            bottoms,
            0.to_signal().into_dyn_signal(),
        ));

        Self {
            area: extent.clone(),
            extent,
            depth: zip_pair(top.clone(), bottom, |t, b| b - t).into_dyn_signal(),
            top: top.into_dyn_signal(),
            children: Some(children),
            mask: u64::MAX.into(),
            staged: None,
            callback: None.into(),
        }
    }

    pub(crate) fn render(
        self: &Rc<Self>,
        parent_pos: PxPoint,
        // cliprect: PxRect,
        driver: &crate::graphics::Driver,
        compositor: &mut crate::CompositorView<'_>,
        dependents: &mut Vec<DynSignal<Layer>>,
    ) -> Result<(), crate::Error> {
        if let Some(staged) = self.staged.as_ref() {
            let area = *crate::sample(&self.area);
            // let extent = *crate::sample(&self.extent);

            // TODO: Pass down the clip area through the render stack so the r-tree can clip things correctly
            //if (extent + parent_pos).intersect(clip) {
            let children = self.children.as_ref().map(|x| crate::sample(x));
            staged.render(
                parent_pos,
                area,
                driver,
                compositor,
                match &children {
                    Some(x) => Some(&x),
                    None => None,
                },
                dependents,
            )
            //} else {
            //    Ok(())
            //}
        } else {
            Ok(())
        }
    }

    // This handles event postprocessing that must always happen, even for directly
    // injected events

    pub(crate) fn postprocess(
        self: &Rc<Self>,
        event: &RawEvent,
        window: &mut crate::WindowState,
        root: &Rc<Self>,
    ) {
        match event {
            // If we successfully process a mousemove event, this node gains hover
            RawEvent::MouseMove {
                device_id,
                pos,
                modifiers,
                all_buttons,
            } => {
                // Either replace the old node, or simply remove it if this is not a valid focus
                // target

                let old = window.set(WindowNodeTrack::Hover, *device_id, Rc::downgrade(self));

                // Tell the old node that it lost hover (if it cares).
                if let Some(old) = old.and_then(|x| x.upgrade()) {
                    let evt = RawEvent::MouseOff {
                        device_id: *device_id,
                        modifiers: *modifiers,
                        all_buttons: *all_buttons,
                    };

                    // We don't care about the result of this event
                    let _ = old.inject_event(&evt, evt.kind(), window, root);
                }

                // We delay injecting MouseOn until after an old node gets MouseOff to present
                // events in a sensible order
                let evt = RawEvent::MouseOn {
                    device_id: *device_id,
                    modifiers: *modifiers,
                    pos: *pos,
                    all_buttons: *all_buttons,
                };
                let _ = self.inject_event(&evt, evt.kind(), window, root);
            }
            RawEvent::Mouse {
                device_id,
                state: MouseState::Up,
                all_buttons: 0,
                pos: crate::PxPoint { x, y, .. },
                ..
            }
            | RawEvent::Touch {
                device_id,
                state: TouchState::End,
                pos: Point3D::<f32, Pixel> { x, y, .. },
                ..
            } => {
                // On any mouseup event, uncapture the cursor if no buttons are down
                window.remove(WindowNodeTrack::Capture, device_id);
                let driver = Arc::downgrade(&window.driver);

                // We don't care if this is accepted or not
                let _ = crate::component::window::Window::on_window_event(
                    window,
                    root.clone(),
                    winit::event::WindowEvent::CursorMoved {
                        device_id: *device_id,
                        position: PhysicalPosition::<f64>::new(*x as f64, *y as f64),
                    },
                    driver,
                );
            }
            _ => (),
        };
    }

    pub(crate) fn inject_event(
        self: &Rc<Self>,
        event: &RawEvent,
        kind: RawEventKind,
        window: &mut crate::WindowState,
        root: &Rc<Self>,
    ) -> (bool, u64) {
        let mut cell = self.callback.borrow_mut();
        if cell.is_some() {
            let mask = self.mask.load(Ordering::Relaxed);
            if (kind as u64 & mask) != 0 {
                let e = cell.as_mut().unwrap().send(event.clone());
                if e.claim {
                    self.postprocess(event, window, root);
                }
                if e.cancel {
                    cell.take();
                }
                return (e.claim, mask);
            }
            return (false, mask);
        }
        return (false, u64::MAX);
    }

    pub(crate) fn offset(self: Rc<Self>, parent: DynSignal<PxVector>) -> DynSignal<PxVector> {
        zip_pair(parent, self.area.clone(), |l, r| {
            (r.topleft() + l).to_vector()
        })
        .into_dyn_signal()
    }

    pub fn process(
        self: &Rc<Self>,
        event: &RawEvent,
        kind: RawEventKind,
        position: PxPoint,
        offset: PxVector,
        dpi: RelDim,
        driver: &std::sync::Weak<crate::Driver>,
        window: &mut crate::WindowState,
        root: &Rc<Self>,
    ) -> bool {
        let area = crate::sample(&self.area);
        if (self.mask.load(Ordering::Acquire) & kind as u64) != 0
            && area.contains(position - offset)
        {
            let child_offset = area.topleft() + offset;

            let mut mask = 0;
            if let Some(children) = &self.children {
                // Children should be sorted from top to bottom
                for child in crate::sample(&children).iter().rev() {
                    // TODO: Split these iterations into positive and negative z indexes, then call
                    // this node after processing index 0 but before negative indices.
                    let claimed = child.process(
                        event,
                        kind,
                        position,
                        child_offset.to_vector(),
                        dpi,
                        driver,
                        window,
                        root,
                    );
                    if claimed {
                        // At this point, we should've already set focus, and are simply walking back up
                        // the stack
                        return true;
                    }

                    mask |= child.mask.load(Ordering::Relaxed);
                }
            }

            let (claimed, m) = self.inject_event(event, kind, window, root);
            mask |= m;

            // This is only ever stored when a message has been rejected by all children and
            // this node. It's mostly used as an optimization for large sets of
            // non-interactive nodes, but it could be made more aggressive.
            self.mask.store(mask, Ordering::Release);

            if claimed {
                match event {
                    // If we successfully process a mouse event, this node gains focus in it's
                    // parent window
                    RawEvent::Mouse {
                        device_id,
                        state: MouseState::Down,
                        ..
                    }
                    | RawEvent::Touch {
                        device_id,
                        state: TouchState::Start,
                        ..
                    } => {
                        let inner = window.window.clone();

                        // On any mousedown event, capture the cursor if it wasn't captured
                        // already
                        window.set(WindowNodeTrack::Capture, *device_id, Rc::downgrade(self));

                        let old =
                            window.set(WindowNodeTrack::Focus, *device_id, Rc::downgrade(self));

                        // Tell the old node that it lost focus (if it cares).
                        if let Some(old) = old.and_then(|old| old.upgrade()) {
                            let evt = RawEvent::Focus {
                                acquired: false,
                                window: inner.clone(),
                            };

                            // We don't care about the result of this event
                            let _ = old.inject_event(&evt, evt.kind(), window, root);
                        }

                        let evt = RawEvent::Focus {
                            acquired: true,
                            window: inner,
                        };
                        let _ = self.inject_event(&evt, evt.kind(), window, root);
                    }
                    _ => (),
                }
            }

            return claimed;
        }

        false
    }
}

impl<'l> crate::event::EventStream<'static, RawEvent> for Node {
    type Subscription<H: crate::event::StreamCallback<RawEvent> + 'static> = Node;

    fn subscribe<H: crate::event::StreamCallback<RawEvent> + 'static>(
        self,
        h: H,
    ) -> Self::Subscription<H> {
        self.callback.replace(Some(Box::new(h)));
        self
    }
}

impl<H: crate::event::StreamCallback<RawEvent>> crate::event::Unsubscribe<RawEvent, Node, H>
    for Node
{
    fn unsubscribe(self) -> (Node, H)
    where
        H: Sized,
    {
        // SAFETY: Because Nodes are put into Rc objects immediately after creation, this can never
        // be called under normal circumstances.
        unreachable!()
    }
}
/*
// 2.5D node which contains a 2D r-tree, embedded inside the parent 3D space.
struct Node25 {
    pub area: AnyRect,
    pub extent: AnyRect,
    pub z: f32, // there is only one z coordinate because the contained area must be flat.
    pub transform: Rotor3,
    pub id: std::sync::Weak<SourceID>,
    pub children: imbl::Vector<Rc<Node>>>,
}

// 3D node capable of arbitrary translation (though it's AABB must still be fully contained within it's parent node)[]
struct Node3D {
    pub area: AbsVolume,
    pub extent: AbsVolume,
    pub transform: Rotor3,
    pub id: std::sync::Weak<SourceID>,
    pub children: imbl::Vector<Either<Rc<Node3D>, Rc<Node25>>>,
}
*/
