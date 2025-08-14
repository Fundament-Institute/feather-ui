// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::StateMachine;
use crate::component::{ChildOf, Layout};
use crate::input::{MouseButton, MouseState, RawEvent, RawEventKind};
use crate::layout::{Desc, base, fixed};
use crate::persist::{FnPersist, VectorMap};
use crate::{
    AbsRect, AbsVector, AnyRect, Dispatchable, PxPoint, PxRect, PxVector, RelDim, RelVector, Slot,
    SourceID, UNSIZED_AXIS, UnResolve, layout,
};
use core::f32;
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use winit::event::DeviceId;
use winit::keyboard::NamedKey;

#[derive(Default, Clone)]
struct MinimalArea {
    area: crate::DRect,
}

impl base::Area for MinimalArea {
    fn area(&self) -> &crate::DRect {
        &self.area
    }
}
impl base::Empty for MinimalArea {}
impl base::ZIndex for MinimalArea {}
impl base::Margin for MinimalArea {}
impl base::Anchor for MinimalArea {}
impl base::Limits for MinimalArea {}
impl base::RLimits for MinimalArea {}
impl fixed::Prop for MinimalArea {}
impl fixed::Child for MinimalArea {}

#[derive(Debug, Dispatch, EnumVariantType, Clone, PartialEq)]
#[evt(derive(Clone), module = "scroll_area_event")]
pub enum ScrollAreaEvent {
    OnScroll(AbsVector),
}

#[derive(Default, Clone, PartialEq)]
struct ScrollAreaState {
    lastdown: HashMap<(DeviceId, u64), (PxPoint, bool)>,
    scroll: PxVector,
    stepsize: (Option<f32>, Option<f32>),
    extension: crate::DAbsRect,
}

impl ScrollAreaState {
    fn apply_scroll(
        &mut self,
        mut change: PxVector,
        area: &AnyRect,
        extent: &AnyRect,
        dpi: RelDim,
    ) -> ScrollAreaEvent {
        let bounds = area.dim();
        //let extension = self.extension.resolve(dpi);

        let mut scroll = self.scroll + change;
        let max = (bounds - extent.dim()).min(crate::AnyDim::zero());

        // We should never scroll by a positive amount (this would scroll past topleft corner), and we should
        // never scroll by an amount that would put us past the bottomright corner.
        scroll = scroll.max(max.to_vector().cast_unit());
        scroll = scroll.min(PxVector::zero());

        change = scroll - self.scroll;
        self.scroll += change;

        ScrollAreaEvent::OnScroll(change.unresolve(dpi))
    }

    fn stepvec(&self) -> RelVector {
        RelVector::new(
            if self.stepsize.0.is_some() { 1.0 } else { 0.0 },
            if self.stepsize.1.is_some() { 1.0 } else { 0.0 },
        )
    }
}

impl super::EventRouter for ScrollAreaState {
    type Input = RawEvent;
    type Output = ScrollAreaEvent;

    fn process(
        mut self,
        input: Self::Input,
        area: AnyRect,
        extent: AnyRect,
        dpi: crate::RelDim,
        _: &std::sync::Weak<crate::Driver>,
    ) -> eyre::Result<
        (Self, smallvec::SmallVec<[Self::Output; 1]>),
        (Self, smallvec::SmallVec<[Self::Output; 1]>),
    > {
        match input {
            RawEvent::Key {
                down: true,
                logical_key: winit::keyboard::Key::Named(code),
                ..
            } => {
                if let Some(change) = match (code, self.stepsize.0, self.stepsize.1) {
                    (NamedKey::ArrowUp, _, Some(y)) => Some(PxVector::new(0.0, -y)),
                    (NamedKey::ArrowDown, _, Some(y)) => Some(PxVector::new(0.0, y)),
                    (NamedKey::ArrowLeft, Some(x), _) => Some(PxVector::new(-x, 0.0)),
                    (NamedKey::ArrowRight, Some(x), _) => Some(PxVector::new(x, 0.0)),
                    (NamedKey::PageUp, _, Some(_)) => Some(PxVector::new(0.0, -area.dim().height)),
                    (NamedKey::PageDown, _, Some(_)) => Some(PxVector::new(0.0, area.dim().height)),
                    _ => None,
                } {
                    let e = self.apply_scroll(change, &area, &extent, dpi);
                    return Ok((self, [e].into()));
                }
            }
            RawEvent::MouseScroll { delta, .. } => {
                let change = match delta {
                    Ok(change) => change,
                    Err(change) => PxVector::new(
                        change.x * self.stepsize.0.unwrap_or_default(),
                        change.y * self.stepsize.1.unwrap_or_default(),
                    ),
                };

                let e = self.apply_scroll(change, &area, &extent, dpi);
                return Ok((self, [e].into()));
            }
            RawEvent::MouseMove { device_id, pos, .. } => {
                let stepvec = self.stepvec();
                if let Some((last_pos, drag)) = self.lastdown.get_mut(&(device_id, 0)) {
                    let diff = (pos - *last_pos).component_mul(stepvec.cast_unit());
                    if !*drag {
                        *drag = true;
                    }

                    if *drag {
                        *last_pos = pos;
                        let e = self.apply_scroll(diff, &area, &extent, dpi);
                        return Ok((self, [e].into()));
                    }
                    return Ok((self, SmallVec::new()));
                }
            }
            RawEvent::Mouse {
                device_id,
                state,
                pos,
                button,
                ..
            } => match (state, button) {
                (MouseState::Down, MouseButton::Left) => {
                    if area.contains(pos.to_untyped()) {
                        self.lastdown.insert((device_id, 0), (pos, false));
                        return Ok((self, SmallVec::new()));
                    }
                }
                (MouseState::Up, MouseButton::Left) => {
                    if let Some((last_pos, drag)) = self.lastdown.remove(&(device_id, 0))
                        && area.contains(pos.to_untyped())
                    {
                        let e = self.apply_scroll(pos - last_pos, &area, &extent, dpi);
                        return Ok((self, if drag { [e].into() } else { SmallVec::new() }));
                    }
                }
                _ => (),
            },
            RawEvent::Touch {
                device_id,
                index,
                state,
                pos,
                ..
            } => match state {
                crate::input::TouchState::Start => {
                    if area.contains(pos.xy().to_untyped()) {
                        self.lastdown
                            .insert((device_id, index as u64), (pos.xy(), false));
                        return Ok((self, SmallVec::new()));
                    }
                }
                crate::input::TouchState::Move => {
                    let stepvec = self.stepvec();
                    if let Some((last_pos, drag)) =
                        self.lastdown.get_mut(&(device_id, index as u64))
                    {
                        let diff = (pos.xy() - *last_pos).component_mul(stepvec.cast_unit());
                        if !*drag {
                            *drag = true;
                        }

                        let e = self.apply_scroll(diff, &area, &extent, dpi);
                        return Ok((self, [e].into()));
                    }
                }
                crate::input::TouchState::End => {
                    // TODO: implement kinetic drag
                    if let Some((last_pos, drag)) = self.lastdown.remove(&(device_id, index as u64))
                        && area.contains(pos.xy().to_untyped())
                    {
                        let e = self.apply_scroll(
                            (pos.xy() - last_pos).component_mul(self.stepvec().cast_unit()),
                            &area,
                            &extent,
                            dpi,
                        );
                        return Ok((self, if drag { [e].into() } else { SmallVec::new() }));
                    }
                }
            },
            _ => (),
        }

        Err((self, SmallVec::new()))
    }
}

#[derive_where(Clone)]
pub struct ScrollArea<T> {
    pub id: Arc<SourceID>,
    props: Rc<T>,
    stepsize: (Option<f32>, Option<f32>),
    extension: crate::DAbsRect,
    children: im::Vector<Option<Box<ChildOf<dyn fixed::Prop>>>>,
    slots: [Option<Slot>; ScrollAreaEvent::SIZE],
}

impl<T: fixed::Prop + 'static> ScrollArea<T> {
    pub fn new(
        id: Arc<SourceID>,
        props: T,
        stepsize: (Option<f32>, Option<f32>),
        extension: crate::DAbsRect,
        children: im::Vector<Option<Box<ChildOf<dyn fixed::Prop>>>>,
        slots: [Option<Slot>; ScrollAreaEvent::SIZE],
    ) -> Self {
        super::set_children(Self {
            id,
            props: props.into(),
            children,
            slots,
            stepsize,
            extension,
        })
    }
}

impl<T: fixed::Prop + 'static> crate::StateMachineChild for ScrollArea<T> {
    fn id(&self) -> Arc<SourceID> {
        self.id.clone()
    }
    fn init(
        &self,
        _: &std::sync::Weak<crate::Driver>,
    ) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        Ok(Box::new(StateMachine {
            state: Some(ScrollAreaState {
                ..Default::default()
            }),
            input_mask: RawEventKind::MouseScroll as u64
                | RawEventKind::Mouse as u64
                | RawEventKind::MouseMove as u64
                | RawEventKind::Touch as u64
                | RawEventKind::Key as u64,
            output: self.slots.clone(),
            changed: true,
        }))
    }
    fn apply_children(
        &self,
        f: &mut dyn FnMut(&dyn crate::StateMachineChild) -> eyre::Result<()>,
    ) -> eyre::Result<()> {
        self.children
            .iter()
            .try_for_each(|x| f(x.as_ref().unwrap().as_ref()))
    }
}

impl<T: fixed::Prop + 'static> super::Component for ScrollArea<T>
where
    for<'a> &'a T: Into<&'a (dyn fixed::Prop + 'static)>,
{
    type Props = T;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        driver: &crate::graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T> + 'static> {
        let scroll = manager
            .get_mut::<StateMachine<ScrollAreaState, { ScrollAreaEvent::SIZE }>>(&self.id)
            .and_then(|state| {
                state
                    .state
                    .as_mut()
                    .ok_or(crate::Error::InternalFailure.into())
            })
            .map(|state| {
                state.stepsize = self.stepsize;
                state.extension = self.extension;
                state.scroll
            })
            .unwrap();

        // To create a scroll area, we create an intermediate layout node to hold the children, which is always unsized, which we then move around to scroll.
        let mut map = VectorMap::new(
            |child: &Option<Box<ChildOf<dyn fixed::Prop>>>| -> Option<Box<dyn Layout<<dyn fixed::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(manager, driver, window))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        let scrollable: Box<dyn layout::Layout<MinimalArea>> =
            Box::new(layout::Node::<MinimalArea, dyn fixed::Prop> {
                props: Rc::new(MinimalArea {
                    area: crate::DRect {
                        px: PxRect::zero(),
                        dp: AbsRect::new(scroll.x, scroll.y, 0.0, 0.0),
                        rel: crate::RelRect::new(
                            0.0,
                            0.0,
                            if self.stepsize.0.is_some() {
                                UNSIZED_AXIS
                            } else {
                                1.0
                            },
                            if self.stepsize.1.is_some() {
                                UNSIZED_AXIS
                            } else {
                                1.0
                            },
                        ),
                    },
                }),
                children,
                id: std::sync::Weak::new(),
                renderable: None,
                layer: None,
            });

        let inner: Box<dyn layout::Layout<dyn fixed::Child>> = Box::new(scrollable);

        Box::new(layout::Node::<T, dyn fixed::Prop> {
            props: self.props.clone(),
            children: im::vector![Some(inner)],
            id: Arc::downgrade(&self.id),
            renderable: None,
            layer: Some((crate::color::sRGB32::white(), 0.0)),
        })
    }
}

/*
#[derive_where(Clone)]
pub struct Node<T: fixed::Prop> {
    pub id: std::sync::Weak<SourceID>,
    pub props: Rc<T>,
    pub children: im::Vector<Option<Box<dyn Layout<dyn fixed::Child>>>>,
    pub size: Rc<RefCell<PxDim>>,
    //pub renderable: Rc<dyn crate::render::Renderable>,
}

impl<T: fixed::Prop + 'static> Layout<T> for Node<T> {
    fn get_props(&self) -> &T {
        &self.props
    }

    fn stage<'a>(
        &self,
        area: AbsRect,
        limits: crate::AbsLimits,
        window: &mut crate::component::window::WindowState,
    ) -> Box<dyn layout::Staged + 'a> {
        let r = <dyn fixed::Prop>::stage(
            self.props.as_ref().into(),
            area,
            limits,
            &self.children,
            self.id.clone(),
            None,
            window,
        );

        *self.size.borrow_mut() = r.get_area().dim().0;
        r
    }
}
*/
