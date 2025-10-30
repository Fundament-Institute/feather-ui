// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::component::window::WindowNodeTrack;
use crate::input::{MouseState, RawEvent, RawEventKind, TouchState};
use crate::layout::Staged;
use crate::persist::{FnPersist2, Persist2, VectorFold};
use crate::reactive::{
    self, AsSignal, DynSignal, Identity, SignalMap, fold_vec, map_vec, zip_pair,
};
use crate::{
    Dispatchable, InputResult, Pixel, PxPoint, PxRect, PxVector, RelDim, SourceID, StateManager,
};
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
    pub state: Option<Rc<dyn crate::StateMachineChild>>,
}

// A tuple like this is necessary to build a chain of parent nodes down the
// recursive process call, but we currently don't need it. This is left here as
// reference. pub struct ParentTuple<'a>(&'a Rc<Node>, Option<&'a
// ParentTuple<'a>>);

impl Node {
    pub fn new(
        area: DynSignal<PxRect>,
        z: Option<DynSignal<i32>>,
        children: Option<DynSignal<imbl::Vector<Rc<Node>>>>,
        staged: Option<Box<dyn Staged>>,
        state: Option<Rc<dyn crate::StateMachineChild>>,
    ) -> Self {
        let z = z.unwrap_or_else(|| 0.to_signal().into_dyn_signal());
        if let Some(children) = children {
            let areas = map_vec(|v| v.area, |v| Identity(v), &children);

            let extent = reactive::join(fold_vec(
                |l, r| zip_pair(l, r, |x, y| x.extend(y)),
                areas,
                PxRect::zero().to_signal().into(),
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
                state,
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
                state,
            }
        }
    }

    pub fn new_subtree(children: DynSignal<imbl::Vector<Rc<Node>>>) -> Self {
        let areas = map_vec(|v| v.area, |v| Identity(v), &children);

        let extent = reactive::join(fold_vec(
            |l, r| zip_pair(l, r, |x, y| x.extend(y)),
            areas,
            PxRect::zero().to_signal().into(),
        ))
        .into_dyn_signal();

        let tops = map_vec(|v| v.top, |v| Identity(v), &children);

        let top = reactive::join(fold_vec(
            |l, r| zip_pair(l, r, |x, y| x.min(y)),
            tops,
            0.to_signal().into_dyn_signal(),
        ));

        let bottoms = map_vec(
            |v| zip_pair(v.top, v.depth, |l, r| l + r),
            |v| Identity(v),
            &children,
        );

        let bottom = reactive::join(fold_vec(
            |l, r| zip_pair(l, r, |x, y| x.max(y)),
            tops,
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
            state: None,
        }
    }

    pub(crate) fn render(
        self: &Rc<Self>,
        parent_pos: PxPoint,
        // cliprect: PxRect,
        driver: &crate::graphics::Driver,
        compositor: &mut crate::CompositorView<'_>,
        dependents: &mut Vec<std::sync::Weak<SourceID>>,
    ) -> Result<(), crate::Error> {
        if let Some(staged) = self.staged.as_ref() {
            let area = *crate::sample(&self.area);

            // TODO: Pass down the clip area through the render stack so the r-tree can clip things correctly
            //if (area + parent_pos).intersect(clip) {
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
        dpi: RelDim,
        offset: PxVector,
        window: &mut crate::WindowState,
        manager: &mut StateManager,
    ) -> InputResult<()> {
        match event {
            // If we successfully process a mousemove event, this node gains hover
            RawEvent::MouseMove {
                device_id,
                pos,
                modifiers,
                all_buttons,
            } => {

                // TODO: replace this with a different method of tracking r-tree nodes
                // Either replace the old node, or simply remove it if this is not a valid focus
                // target
                /*
                let (old, valid) = if let Some(id) = self.id.upgrade() {
                    (
                        window.set(WindowNodeTrack::Hover, *device_id, id, Rc::downgrade(self)),
                        true,
                    )
                } else {
                    (window.remove(WindowNodeTrack::Hover, device_id), false)
                };

                let driver = Arc::downgrade(&window.driver);

                // Tell the old node that it lost hover (if it cares).
                if let Some(old) = old.and_then(|x| x.upgrade()) {
                    let evt = RawEvent::MouseOff {
                        device_id: *device_id,
                        modifiers: *modifiers,
                        all_buttons: *all_buttons,
                    };

                    // We don't care about the result of this event
                    let _ = old.inject_event(
                        &evt,
                        evt.kind(),
                        dpi,
                        PxVector::zero(),
                        window_id.clone(),
                        &driver,
                        manager,
                    );
                }

                // We delay injecting MouseOn until after an old node gets MouseOff to present
                // events in a sensible order
                if valid {
                    let evt = RawEvent::MouseOn {
                        device_id: *device_id,
                        modifiers: *modifiers,
                        pos: *pos,
                        all_buttons: *all_buttons,
                    };
                    let _ = self.inject_event(
                        &evt,
                        evt.kind(),
                        dpi,
                        offset,
                        window_id,
                        &driver,
                        manager,
                    );
                }*/
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
                    Self::find_root(self.clone()),
                    winit::event::WindowEvent::CursorMoved {
                        device_id: *device_id,
                        position: PhysicalPosition::<f64>::new(*x as f64, *y as f64),
                    },
                    manager,
                    driver,
                );
            }
            _ => (),
        };

        InputResult::Consume(())
    }

    pub(crate) fn inject_event(
        self: &Rc<Self>,
        event: &RawEvent,
        kind: RawEventKind,
        dpi: &crate::MutableSignal<RelDim>,
        offset: PxVector,
        window: &mut crate::WindowState,
        driver: &std::sync::Weak<crate::Driver>,
        manager: &mut StateManager,
    ) -> InputResult<u64> {
        if let Some(id) = self.id.upgrade()
            && let Ok(state) = manager.get_trait(&id)
        {
            let mask = state.input_mask();
            if (kind as u64 & mask) != 0 {
                match manager.process(
                    event.clone().extract(),
                    &crate::Slot(id.clone(), 0), /* TODO: We currently don't use the slot index
                                                  * here, but we might need to later */
                    dpi,
                    self.area + offset,
                    self.extent,
                    driver,
                ) {
                    Ok(false) => return InputResult::Forward(mask),
                    Ok(true) => (),
                    Err(e) => return InputResult::Error(e),
                }

                return self
                    .postprocess(event, dpi, offset, window, manager)
                    .map(|_| mask);
            }
            return InputResult::Forward(mask);
        }
        InputResult::Forward(u64::MAX)
    }

    // We allow this to return an invalid weak pointer because returning an
    // *invalid* root is a more obvious problem than returning the *wrong* node
    // as if it were the root (which can be very confusing).
    fn find_root(mut node: Rc<Node>) -> std::rc::Weak<Node> {
        while let Some(parent) = node.parent.get() {
            if let Some(n) = parent.upgrade() {
                node = n;
            } else {
                return parent.clone();
            }
        }
        Rc::downgrade(&node)
    }

    pub(crate) fn offset(mut node: Rc<Node>) -> PxVector {
        let mut offset = PxVector::zero();
        while let Some(parent) = node.parent.get().and_then(|x| x.upgrade()) {
            offset = (parent.area.topleft() + offset).to_vector();
            node = parent;
        }
        offset
    }

    pub fn process(
        self: &Rc<Self>,
        event: &RawEvent,
        kind: RawEventKind,
        position: PxPoint,
        offset: PxVector,
        dpi: RelDim,
        driver: &std::sync::Weak<crate::Driver>,
        manager: &mut StateManager,
        window: &mut crate::WindowState,
    ) -> InputResult<()> {
        if (self.mask.load(Ordering::Acquire) & kind as u64) != 0
            && self.area.contains(position - offset)
        {
            let child_offset = self.area.topleft() + offset;

            let mut mask = 0;
            // Children should be sorted from top to bottom
            for child in self.children.iter().rev() {
                // TODO: Split these iterations into positive and negative z indexes, then call
                // this node after processing index 0 but before negative indices.
                let r = child.process(
                    event,
                    kind,
                    position,
                    child_offset.to_vector(),
                    dpi,
                    driver,
                    manager,
                    window,
                );
                if !r.is_reject() {
                    // At this point, we should've already set focus, and are simply walking back up
                    // the stack
                    return r;
                }

                mask |= child.mask.load(Ordering::Relaxed);
            }

            let e = self.inject_event(event, kind, dpi, offset, window, driver, manager);
            match e {
                InputResult::Consume(m) | InputResult::Forward(m) => mask |= m,
                _ => (),
            };

            // This is only ever stored when a message has been rejected by all children and
            // this node. It's mostly used as an optimization for large sets of
            // non-interactive nodes, but it could be made more aggressive.
            self.mask.store(mask, Ordering::Release);

            if e.is_accept() {
                match event {
                    // If we successfully process a mouse event, this node gains focus in it's
                    // parent window
                    RawEvent::Mouse {
                        device_id,
                        state: MouseState::Down,
                        pos: crate::PxPoint { x, y, .. },
                        ..
                    }
                    | RawEvent::Touch {
                        device_id,
                        state: TouchState::Start,
                        pos: Point3D::<f32, Pixel> { x, y, .. },
                        ..
                    } => {
                        let inner = window.window.clone();

                        // TODO: Redo how focus works
                        // Either replace the old node, or simply remove it if this is not a valid
                        // focus target
                        /*
                        let (old, valid) = if let Some(id) = self.id.upgrade() {
                            // On any mousedown event, capture the cursor if it wasn't captured
                            // already
                            window.set(
                                WindowNodeTrack::Capture,
                                *device_id,
                                id.clone(),
                                Rc::downgrade(self),
                            );
                            (
                                window.set(
                                    WindowNodeTrack::Focus,
                                    *device_id,
                                    id,
                                    Rc::downgrade(self),
                                ),
                                true,
                            )
                        } else {
                            window.remove(WindowNodeTrack::Capture, device_id);
                            (window.remove(WindowNodeTrack::Focus, device_id), false)
                        };

                        // Tell the old node that it lost focus (if it cares).
                        if let Some(old) = old.and_then(|old| old.upgrade()) {
                            let evt = RawEvent::Focus {
                                acquired: false,
                                window: inner.clone(),
                            };

                            // We don't care about the result of this event
                            let _ = old.inject_event(
                                &evt,
                                evt.kind(),
                                dpi,
                                PxVector::zero(),
                                window_id.clone(),
                                driver,
                                manager,
                            );
                        }

                        // We delay injecting Focus until after the old node gets it's own Focus
                        // event to preserve a sensible ordering
                        if valid {
                            let evt = RawEvent::Focus {
                                acquired: true,
                                window: inner,
                            };
                            let _ = self.inject_event(
                                &evt,
                                evt.kind(),
                                dpi,
                                offset,
                                window_id.clone(),
                                driver,
                                manager,
                            );
                        } else {
                            // If this wasn't a valid node, we removed capture but didn't replace
                            // it, so we have to inject a mousemove event
                            let _ = crate::component::window::Window::on_window_event(
                                window_id.clone(),
                                Self::find_root(self.clone()),
                                winit::event::WindowEvent::CursorMoved {
                                    device_id: *device_id,
                                    position: PhysicalPosition::<f64>::new(*x as f64, *y as f64),
                                },
                                manager,
                                driver.clone(),
                            );
                        }*/
                    }
                    _ => (),
                }
            }
            return e.map(|_| ());
        }
        InputResult::Forward(())
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
