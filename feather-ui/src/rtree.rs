// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::component::window::WindowNodeTrack;
use crate::event::StreamCallback;
use crate::input::{MouseState, RawEvent, RawEventKind, TouchState};
use crate::layout::Staged;
use crate::reactive::{
    self, ConstSignal, DynSignal, Identity, SignalMap, ToSignal, fold_vec, zip_pair,
};
use crate::render::compositor::Layer;
use crate::{Pixel, PxPoint, PxRect, PxVector, RelDim};
use guillotiere::euclid::Point3D;
use std::rc::Rc;
use winit::dpi::PhysicalPosition;

// Nonsense taken from `dtolnay/typeid` crate that lets us take a non-'static typeid.
#[must_use]
#[inline(always)]
pub fn typeid_of<T>() -> std::any::TypeId
where
    T: ?Sized,
{
    trait NonStaticAny {
        fn get_type_id(&self) -> std::any::TypeId
        where
            Self: 'static;
    }

    impl<T: ?Sized> NonStaticAny for PhantomData<T> {
        #[inline(always)]
        fn get_type_id(&self) -> std::any::TypeId
        where
            Self: 'static,
        {
            std::any::TypeId::of::<T>()
        }
    }

    let phantom_data = PhantomData::<T>;
    NonStaticAny::get_type_id(unsafe {
        std::mem::transmute::<&dyn NonStaticAny, &(dyn NonStaticAny + 'static)>(&phantom_data)
    })
}

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
    pub callback: RefCell<Option<(Box<dyn StreamCallback<RawEvent>>, std::any::TypeId)>>,
}

// A tuple like this is necessary to build a chain of parent nodes down the
// recursive process call, but we currently don't need it. This is left here as
// reference.
/*
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
*/

impl Node {
    pub fn new(
        area: DynSignal<PxRect>,
        z: Option<DynSignal<i32>>,
        children: Option<DynSignal<imbl::Vector<Rc<Node>>>>,
        staged: Option<Box<dyn Staged>>,
    ) -> Self {
        let z = z.unwrap_or_else(|| 0.to_signal().into_dyn_signal());
        if let Some(children) = children {
            let areas = children
                .clone()
                .map_elements(|v| v.area.clone(), |v| Identity(v.clone()));

            let extent = reactive::join(fold_vec(
                |l, r| zip_pair(l, r, |x, y| x.extend(*y)).into(),
                areas,
                ConstSignal::new(PxRect::zero()).into(),
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
        let areas = children
            .clone()
            .map_elements(|v| v.area.clone(), |v| Identity(v.clone()));

        let extent = reactive::join(fold_vec(
            |l, r| zip_pair(l, r, |x, y| x.extend(*y)).into(),
            areas,
            ConstSignal::new(PxRect::zero()).into(),
        ))
        .into_dyn_signal();

        let tops = children
            .clone()
            .map_elements(|v| v.top.clone(), |v| Identity(v.clone()));

        let top = reactive::join(fold_vec(
            |l, r| reactive::cmp::min(l, r).into(),
            tops,
            0.to_signal().into_dyn_signal(),
        ));

        let bottoms = children.clone().map_elements(
            |v| (v.top.clone() + v.depth.clone()).into_dyn_signal(),
            |v| Identity(v.clone()),
        );

        let bottom = reactive::join(fold_vec(
            |l, r| reactive::cmp::max(l, r).into(),
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
        driver: &crate::graphics::Driver,
        compositor: &mut crate::CompositorView<'_>,
        dependents: &mut Vec<std::rc::Weak<Layer>>,
    ) -> Result<(), crate::RenderError> {
        if let Some(staged) = self.staged.as_ref() {
            let extent = *crate::sample(&self.extent);
            compositor.redraw.add_parent(&self.extent);

            if (extent + parent_pos).collides(&compositor.current_clip()) {
                let children = self.children.as_ref().map(|x| crate::sample(x).clone());
                staged.render(
                    parent_pos,
                    driver,
                    compositor,
                    match &children {
                        Some(x) => Some(&x),
                        None => None,
                    },
                    dependents,
                )
            } else {
                Ok(())
            }
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
        offset: Option<PxVector>,
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
                    let _ = old.inject_event(&evt, evt.kind(), window, root, offset);
                }

                // We delay injecting MouseOn until after an old node gets MouseOff to present
                // events in a sensible order
                let evt = RawEvent::MouseOn {
                    device_id: *device_id,
                    modifiers: *modifiers,
                    pos: *pos,
                    all_buttons: *all_buttons,
                };
                let _ = self.inject_event(&evt, evt.kind(), window, root, offset);
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
        offset: Option<PxVector>,
    ) -> (bool, u64) {
        let mut cell = self.callback.borrow_mut();
        if cell.is_some() {
            let mask = self.mask.load(Ordering::Relaxed);
            if (kind as u64 & mask) != 0 {
                let e = cell.as_mut().unwrap().0.send(event.reposition(offset));
                if e.cancel {
                    cell.take();
                }
                // Enforce that we drop the reference before calling postprocess, which is allowed to inject an event that we might need to process again.
                std::mem::drop(cell);
                if e.claim {
                    self.postprocess(event, window, root, offset);
                }
                return (e.claim, mask);
            }
            return (false, mask);
        }
        return (false, u64::MAX);
    }

    fn target_event_inner(
        self: &Rc<Self>,
        event: &RawEvent,
        pos: PxPoint,
        offset: PxVector,
        kind: RawEventKind,
        window: &mut crate::WindowState,
        root: &Rc<Self>,
        target: &Rc<Self>,
    ) -> Option<(bool, u64)> {
        let area = *crate::sample(&self.area);
        if area.contains(pos - offset) {
            if Rc::ptr_eq(target, self) {
                // ladies and gentlemen, we gottem
                return Some(self.inject_event(event, kind, window, root, Some(offset)));
            }
            let child_offset = (area.topleft() + offset).to_vector();

            if let Some(children) = &self.children {
                // The ordering of children here is irrelevent, since we are searching for a particular node.
                for child in crate::sample(&children).iter() {
                    if let Some(v) = child.target_event_inner(
                        event,
                        pos,
                        child_offset,
                        kind,
                        window,
                        root,
                        target,
                    ) {
                        return Some(v);
                    }
                }
            }
        }
        None
    }

    /// target_event is used for events being sent to a particular node, usually for hover, focus, or capture events,
    /// where we need to send the offset, but only if the mouse cursor is actually inside the node in question. This
    /// ignores all normal event processing, and traverses the r-tree solely to search for whether any nodes under the
    /// event's position are the targeted node. If the search fails or if the event doesn't have a position, the event
    /// is still injected to the targeted node, but without a position.
    pub fn target_event(
        self: &Rc<Self>,
        event: &RawEvent,
        kind: RawEventKind,
        window: &mut crate::WindowState,
        target: &Rc<Self>,
    ) -> (bool, u64) {
        let result = match event {
            RawEvent::Drop { pos, .. }
            | RawEvent::Mouse { pos, .. }
            | RawEvent::MouseOn { pos, .. }
            | RawEvent::MouseMove { pos, .. }
            | RawEvent::MouseScroll { pos, .. } => {
                self.target_event_inner(event, *pos, PxVector::zero(), kind, window, self, target)
            }
            RawEvent::Touch { pos, .. } => self.target_event_inner(
                event,
                pos.xy(),
                PxVector::zero(),
                kind,
                window,
                self,
                target,
            ),
            _ => None,
        };

        result.unwrap_or_else(|| target.inject_event(event, kind, window, self, None))
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
        let area = *crate::sample(&self.area);
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

            let (claimed, m) = self.inject_event(event, kind, window, root, Some(offset));
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
                            let _ = old.inject_event(&evt, evt.kind(), window, root, Some(offset));
                        }

                        let evt = RawEvent::Focus {
                            acquired: true,
                            window: inner,
                        };
                        let _ = self.inject_event(&evt, evt.kind(), window, root, Some(offset));
                    }
                    _ => (),
                }
            }

            return claimed;
        }

        false
    }
}

#[repr(transparent)]
pub struct NodeSubscription<H>(std::rc::Weak<Node>, PhantomData<H>);

impl crate::event::EventStream<'static, RawEvent> for Rc<Node> {
    type Subscription<H: StreamCallback<RawEvent> + 'static> = NodeSubscription<H>;

    fn subscribe<H: StreamCallback<RawEvent> + 'static>(self, h: H) -> Self::Subscription<H> {
        let boxed: Box<dyn StreamCallback<RawEvent>> = Box::new(h);

        self.callback.replace(Some((boxed, typeid_of::<H>())));
        NodeSubscription(Rc::downgrade(&self), PhantomData)
    }
}
impl<H: StreamCallback<RawEvent> + 'static> crate::event::Unsubscribe<RawEvent, Rc<Node>, H>
    for NodeSubscription<H>
{
    fn unsubscribe(self) -> (Rc<Node>, H) {
        let node = self
            .0
            .upgrade()
            .expect("Tried to unsubscribe from node that doesn't exist!");
        let (boxed, ty) = node
            .callback
            .borrow_mut()
            .take()
            .expect("Tried to unsubscribe from Node, but subscription was invalid!");

        // This ensures the pointer we extract out is the type we expect
        assert_eq!(ty, typeid_of::<H>());
        let raw = Box::into_raw(boxed) as *mut H;

        // Using from_raw here is important because manually deallocating will segfault if H is zero-sized.
        let h = unsafe { Box::<H>::from_raw(raw) };
        (node, *h)
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
