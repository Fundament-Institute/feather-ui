// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::input::{MouseButton, MouseMoveState, MouseState, RawEvent, RawEventKind};
use crate::layout;
use crate::layout::basic::Basic;
use crate::layout::Desc;
use crate::layout::Layout;
use crate::outline::OutlineFrom;
use crate::outline::StateMachine;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::DispatchPair;
use crate::Dispatchable;
use crate::Error;
use crate::Outline;
use crate::SourceID;
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use std::any::TypeId;
use std::collections::HashMap;
use std::rc::Rc;
use ultraviolet::Vec2;

#[derive(Debug, EnumVariantType, Clone)]
#[evt(derive(Clone), module = "draggable_event")]
pub enum DraggableEvent {
    OnClick(Vec2),
    OnDblClick(Vec2),
    OnDrag(Vec2),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum DraggableEventKind {
    OnClick = 1,
    OnDblClick = 2,
    OnDrag = 4,
}

impl Dispatchable for DraggableEvent {
    const SIZE: usize = 3;

    fn extract(self) -> DispatchPair {
        match self {
            DraggableEvent::OnClick(_) => (
                DraggableEventKind::OnClick as u64,
                Box::new(draggable_event::OnClick::try_from(self).unwrap()),
            ),
            DraggableEvent::OnDblClick(_) => (
                DraggableEventKind::OnDblClick as u64,
                Box::new(draggable_event::OnDblClick::try_from(self).unwrap()),
            ),
            DraggableEvent::OnDrag(_) => (
                DraggableEventKind::OnDrag as u64,
                Box::new(draggable_event::OnDrag::try_from(self).unwrap()),
            ),
        }
    }

    fn restore(pair: DispatchPair) -> Result<Self, Error> {
        const KIND_ONCLICK: u64 = DraggableEventKind::OnClick as u64;
        const KIND_ONDBLCLICK: u64 = DraggableEventKind::OnDblClick as u64;
        const KIND_DRAG: u64 = DraggableEventKind::OnDrag as u64;
        let typeid = (*pair.1).type_id();
        match pair.0 {
            KIND_ONCLICK => Ok(DraggableEvent::from(
                *pair.1.downcast::<draggable_event::OnClick>().map_err(|_| {
                    Error::MismatchedEnumTag(
                        pair.0,
                        TypeId::of::<draggable_event::OnClick>(),
                        typeid,
                    )
                })?,
            )),
            KIND_ONDBLCLICK => Ok(DraggableEvent::from(
                *pair
                    .1
                    .downcast::<draggable_event::OnDblClick>()
                    .map_err(|_| {
                        Error::MismatchedEnumTag(
                            pair.0,
                            TypeId::of::<draggable_event::OnDblClick>(),
                            typeid,
                        )
                    })?,
            )),
            KIND_DRAG => Ok(DraggableEvent::from(
                *pair.1.downcast::<draggable_event::OnDrag>().map_err(|_| {
                    Error::MismatchedEnumTag(
                        pair.0,
                        TypeId::of::<draggable_event::OnDrag>(),
                        typeid,
                    )
                })?,
            )),
            _ => Err(Error::InvalidEnumTag(pair.0)),
        }
    }
}

#[derive(Default, Clone)]
struct DraggableState {
    lastdrag: HashMap<winit::event::DeviceId, ultraviolet::Vec2>,
    lastdown: HashMap<winit::event::DeviceId, ultraviolet::Vec2>,
    lasttouchmove: HashMap<(winit::event::DeviceId, u64), ultraviolet::Vec3>,
    lasttouch: HashMap<(winit::event::DeviceId, u64), ultraviolet::Vec3>,
}

#[derive_where(Clone)]
pub struct Draggable<Parent: Clone> {
    pub id: Rc<SourceID>,
    pub props: Parent,
    pub basic: Basic,
    pub children: im::Vector<Option<Box<dyn Outline<<Basic as Desc>::Impose>>>>,
    pub slots: [Option<crate::Slot>; DraggableEvent::SIZE],
}

impl<Parent: Clone + 'static> super::Outline<Parent> for Draggable<Parent> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        let ondrag = Box::new(
            crate::wrap_event::<RawEvent, DraggableEvent, DraggableState>(|e, area, mut data| {
                match e {
                    RawEvent::MouseMove {
                        device_id,
                        state,
                        pos,
                        all_buttons,
                        ..
                    } => match state {
                        MouseMoveState::Move => {
                            if (all_buttons & MouseButton::L as u8) != 0 {
                                if let Some(last_pos) = data.lastdown.get(&device_id) {
                                    let diff = pos - *last_pos;
                                    if diff.dot(diff) > 4.0 {
                                        data.lastdown.remove(&device_id).unwrap();
                                        data.lastdrag.insert(device_id, pos);
                                        return Ok((data, vec![DraggableEvent::OnDrag(diff)]));
                                    }
                                }
                                if let Some(last_pos) = data.lastdrag.get(&device_id).map(|x| *x) {
                                    data.lastdrag.insert(device_id, pos);
                                    return Ok((
                                        data,
                                        vec![DraggableEvent::OnDrag(pos - last_pos)],
                                    ));
                                }
                            }
                        }
                        _ => {}
                    },
                    RawEvent::Mouse {
                        device_id,
                        state,
                        pos,
                        ..
                    } => {
                        match state {
                            MouseState::Down => {
                                if area.contains(pos) {
                                    data.lastdown.insert(device_id, pos);
                                    return Ok((data, vec![]));
                                }
                            } // TODO: if this is inside the area, capture the mouse input.
                            MouseState::Up => {
                                if data.lastdown.contains_key(&device_id) {
                                    data.lastdown.remove(&device_id);
                                    if area.contains(pos) {
                                        return Ok((data, vec![DraggableEvent::OnClick(pos)]));
                                    }
                                }
                                if let Some(last_pos) = data.lastdrag.get(&device_id).map(|x| *x) {
                                    data.lastdrag.remove(&device_id);
                                    return Ok((
                                        data,
                                        vec![DraggableEvent::OnDrag(pos - last_pos)],
                                    ));
                                }
                            }
                            MouseState::DblClick => {
                                if data.lastdown.contains_key(&device_id) {
                                    data.lastdown.remove(&device_id);
                                    if area.contains(pos) {
                                        return Ok((data, vec![DraggableEvent::OnDblClick(pos)]));
                                    }
                                }
                            }
                        }
                    }
                    RawEvent::Touch {
                        device_id,
                        index,
                        state,
                        pos,
                        ..
                    } => match state {
                        crate::input::TouchState::Start => {
                            if area.contains(pos.xy()) {
                                data.lasttouch.insert((device_id, index), pos);
                                return Ok((data, vec![]));
                            }
                        }
                        crate::input::TouchState::Move => {
                            if data.lasttouch.contains_key(&(device_id, index)) {
                                let last_pos = data.lasttouch.remove(&(device_id, index)).unwrap();
                                data.lasttouchmove.insert((device_id, index), pos);
                                return Ok((
                                    data,
                                    vec![DraggableEvent::OnDrag((pos - last_pos).xy())],
                                ));
                            }
                            if let Some(last_pos) =
                                data.lasttouchmove.get(&(device_id, index)).map(|x| *x)
                            {
                                data.lasttouchmove.insert((device_id, index), pos);
                                return Ok((
                                    data,
                                    vec![DraggableEvent::OnDrag((pos - last_pos).xy())],
                                ));
                            }
                        }
                        crate::input::TouchState::End => {
                            if data.lasttouch.contains_key(&(device_id, index)) {
                                if area.contains(pos.xy()) {
                                    return Ok((data, vec![DraggableEvent::OnClick(pos.xy())]));
                                }
                                data.lasttouch.remove(&(device_id, index));
                            }
                            if let Some(last_pos) =
                                data.lasttouchmove.get(&(device_id, index)).map(|x| *x)
                            {
                                data.lasttouchmove.remove(&(device_id, index));
                                return Ok((
                                    data,
                                    vec![DraggableEvent::OnDrag((pos - last_pos).xy())],
                                ));
                            }
                        }
                    },
                    _ => (),
                }
                Err((data, vec![]))
            }),
        );

        //<DraggableEvent, DraggableState, 1, 3>
        Ok(Box::new(StateMachine {
            state: Some(Default::default()),
            input: [(
                RawEventKind::MouseMove as u64
                    | RawEventKind::Mouse as u64
                    | RawEventKind::Touch as u64,
                ondrag,
            )],
            output: self.slots.clone(),
        }))
    }

    fn init_all(&self, manager: &mut crate::StateManager) -> eyre::Result<()> {
        for child in self.children.iter() {
            manager.init_outline(child.as_ref().unwrap().as_ref())?;
        }
        Ok(())
    }

    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &crate::DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Parent>> {
        let map = VectorMap::new(
            |child: &Option<Box<OutlineFrom<Basic>>>|
             -> Option<Box<dyn Layout<<Basic as Desc>::Impose>>> { Some(child.as_ref().unwrap().layout(state, driver, config)) },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<Basic, Parent> {
            props: self.basic.clone(),
            imposed: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
