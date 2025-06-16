// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::rc::Rc;

use crate::component::window::WindowNodeTrack;
use crate::input::{MouseState, RawEvent, RawEventKind, TouchState};
use crate::persist::{FnPersist2, VectorFold};
use crate::{AbsRect, Dispatchable, SourceID, StateManager, WindowStateMachine};
use eyre::Result;
use std::sync::Arc;
use ultraviolet::Vec2;
use winit::dpi::PhysicalPosition;

pub struct Node {
    pub area: AbsRect, // This is the calculated area of the node from the layout relative to the topleft corner of the parent.
    pub extent: AbsRect, // This is the minimal bounding rectangle of the children's extent relative to OUR topleft corner.
    pub top: i32, // 2D R-tree nodes are actually 3 dimensional, but the z-axis can never overlap (because layout rects have no depth).
    pub bottom: i32,
    pub id: std::rc::Weak<SourceID>,
    pub children: im::Vector<Option<Rc<Node>>>,
    pub parent: std::cell::OnceCell<std::rc::Weak<Node>>,
}

// A tuple like this is necessary to build a chain of parent nodes down the recursive process call, but we currently don't need it.
// This is left here as reference.
//pub struct ParentTuple<'a>(&'a Rc<Node>, Option<&'a ParentTuple<'a>>);

impl Node {
    pub fn new(
        area: AbsRect,
        z: Option<i32>,
        children: im::Vector<Option<Rc<Node>>>,
        id: std::rc::Weak<SourceID>,
        window: &mut crate::component::window::WindowState,
    ) -> Rc<Self> {
        let this = Rc::new_cyclic(|this| {
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

            // TODO: This is inefficient for large trees, but the alternative is to somehow maintain a "capture" pointer on each rtree node,
            // which requires cooperation from the persistent data structure to maintain, which we don't have right now.
            for child in &children {
                child.as_ref().unwrap().parent.get_or_init(|| this.clone());
            }

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
                id: id.clone(),
                children,
                parent: Default::default(),
            }
        });

        if let Some(id) = id.upgrade() {
            window.update_node(id, Rc::downgrade(&this));
        }

        this
    }

    // This handles event postprocessing that must always happen, even for directly injected events
    pub(crate) fn postprocess(
        self: &Rc<Self>,
        event: &RawEvent,
        dpi: Vec2,
        offset: Vec2,
        window_id: Rc<SourceID>,
        manager: &mut StateManager,
    ) -> Result<(), ()> {
        match event {
            // If we successfully process a mousemove event, this node gains hover
            RawEvent::MouseMove {
                device_id,
                pos,
                modifiers,
                all_buttons,
                graphics,
            } => {
                let state: &mut WindowStateMachine = manager.get_mut(&window_id).map_err(|_| ())?;
                let window = state.state.as_mut().unwrap();

                // Either replace the old node, or simply remove it if this is not a valid focus target
                let (old, valid) = if let Some(id) = self.id.upgrade() {
                    (
                        window.set(WindowNodeTrack::Hover, *device_id, id, Rc::downgrade(self)),
                        true,
                    )
                } else {
                    (window.remove(WindowNodeTrack::Hover, device_id), false)
                };

                // Tell the old node that it lost hover (if it cares).
                if let Some(old) = old.and_then(|x| x.upgrade()) {
                    let evt = RawEvent::MouseOff {
                        device_id: *device_id,
                        modifiers: *modifiers,
                        all_buttons: *all_buttons,
                        graphics: graphics.clone(),
                    };

                    // We don't care about the result of this event
                    let _ = old.inject_event(
                        &evt,
                        evt.kind(),
                        dpi,
                        Vec2::zero(),
                        window_id.clone(),
                        manager,
                    );
                }

                // We delay injecting MouseOn until after an old node gets MouseOff to present events in a sensible order
                if valid {
                    let evt = RawEvent::MouseOn {
                        device_id: *device_id,
                        modifiers: *modifiers,
                        pos: *pos,
                        all_buttons: *all_buttons,
                        graphics: graphics.clone(),
                    };
                    let _ = self.inject_event(&evt, evt.kind(), dpi, offset, window_id, manager);
                }
            }
            RawEvent::Mouse {
                device_id,
                state: MouseState::Up,
                all_buttons: 0,
                pos: PhysicalPosition { x, y },
                ..
            }
            | RawEvent::Touch {
                device_id,
                state: TouchState::End,
                pos: ultraviolet::Vec3 { x, y, z: _ },
                ..
            } => {
                // On any mouseup event, uncapture the cursor if no buttons are down
                let state: &mut WindowStateMachine = manager.get_mut(&window_id).map_err(|_| ())?;
                let window = state.state.as_mut().unwrap();
                window.remove(WindowNodeTrack::Capture, device_id);
                let graphics = Arc::downgrade(&window.graphics);

                // We don't care if this is accepted or not
                let _ = crate::component::window::Window::on_window_event(
                    window_id,
                    Self::find_root(self.clone()),
                    winit::event::WindowEvent::CursorMoved {
                        device_id: *device_id,
                        position: PhysicalPosition::<f64>::new(*x as f64, *y as f64),
                    },
                    manager,
                    graphics,
                );
            }
            _ => (),
        };

        Ok(())
    }

    pub(crate) fn inject_event(
        self: &Rc<Self>,
        event: &RawEvent,
        kind: RawEventKind,
        dpi: Vec2,
        offset: Vec2,
        window_id: Rc<SourceID>,
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
                        return self.postprocess(event, dpi, offset, window_id, manager);
                    }
                }
            }
        }
        Err(())
    }

    // We allow this to return an invalid weak pointer because returning an *invalid* root is a more obvious problem than returning
    // the *wrong* node as if it were the root (which can be very confusing).
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

    pub(crate) fn offset(mut node: Rc<Node>) -> Vec2 {
        let mut offset = Vec2::zero();
        while let Some(parent) = node.parent.get().and_then(|x| x.upgrade()) {
            offset += parent.area.topleft();
            node = parent;
        }
        offset
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
    ) -> Result<(), ()> {
        if self.area.contains(position - offset) {
            if let Ok(()) = self.inject_event(event, kind, dpi, offset, window_id.clone(), manager)
            {
                match event {
                    // If we successfully process a mouse event, this node gains focus in it's parent window
                    RawEvent::Mouse {
                        device_id,
                        state: MouseState::Down,
                        pos: PhysicalPosition { x, y },
                        ..
                    }
                    | RawEvent::Touch {
                        device_id,
                        state: TouchState::Start,
                        pos: ultraviolet::Vec3 { x, y, z: _ },
                        ..
                    } => {
                        let state: &mut WindowStateMachine =
                            manager.get_mut(&window_id).map_err(|_| ())?;
                        let window = state.state.as_mut().unwrap();
                        let inner = window.window.clone();

                        // Either replace the old node, or simply remove it if this is not a valid focus target
                        let (old, valid) = if let Some(id) = self.id.upgrade() {
                            // On any mousedown event, capture the cursor if it wasn't captured already
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

                        let graphics = Arc::downgrade(&window.graphics);

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
                                Vec2::zero(),
                                window_id.clone(),
                                manager,
                            );
                        }

                        // We delay injecting Focus until after the old node gets it's own Focus event to preserve a sensible ordering
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
                                manager,
                            );
                        } else {
                            // If this wasn't a valid node, we removed capture but didn't replace it, so we have to inject a mousemove event
                            let _ = crate::component::window::Window::on_window_event(
                                window_id.clone(),
                                Self::find_root(self.clone()),
                                winit::event::WindowEvent::CursorMoved {
                                    device_id: *device_id,
                                    position: PhysicalPosition::<f64>::new(*x as f64, *y as f64),
                                },
                                manager,
                                graphics,
                            );
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
