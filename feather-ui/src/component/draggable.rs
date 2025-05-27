// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::component::{ComponentFrom, StateMachine};
use crate::input::{MouseButton, MouseMoveState, MouseState, RawEvent, RawEventKind};
use crate::layout::{self, Desc, Layout, fixed};
use crate::persist::{FnPersist, VectorMap};
use crate::{Dispatchable, SourceID};
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use std::collections::HashMap;
use std::rc::Rc;
use ultraviolet::Vec2;

#[derive(Debug, Dispatch, EnumVariantType, Clone, PartialEq)]
#[evt(derive(Clone), module = "draggable_event")]
pub enum DraggableEvent {
    OnClick(Vec2),
    OnDblClick(Vec2),
    OnDrag(Vec2),
}

#[derive(Default, Clone, PartialEq)]
struct DraggableState {
    lastdrag: HashMap<winit::event::DeviceId, ultraviolet::Vec2>,
    lastdown: HashMap<winit::event::DeviceId, ultraviolet::Vec2>,
    lasttouchmove: HashMap<(winit::event::DeviceId, u64), ultraviolet::Vec3>,
    lasttouch: HashMap<(winit::event::DeviceId, u64), ultraviolet::Vec3>,
}

#[derive_where(Clone)]
pub struct Draggable<T: fixed::Prop + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub children: im::Vector<Option<Box<ComponentFrom<dyn fixed::Prop>>>>,
    pub slots: [Option<crate::Slot>; DraggableEvent::SIZE],
}

impl<T: fixed::Prop + 'static> super::Component<T> for Draggable<T> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        let ondrag = Box::new(
            crate::wrap_event::<RawEvent, DraggableEvent, DraggableState>(
                |e, area, dpi, mut data| {
                    match e {
                        RawEvent::MouseMove {
                            device_id,
                            state: MouseMoveState::Move,
                            pos,
                            all_buttons,
                            ..
                        } => {
                            let pos = crate::pixel_to_vec(pos);
                            if (all_buttons & MouseButton::L as u8) != 0 {
                                if let Some(last_pos) = data.lastdown.get(&device_id) {
                                    let diff = pos - *last_pos;
                                    if diff.dot(diff) > 4.0 {
                                        data.lastdown.remove(&device_id).unwrap();
                                        data.lastdrag.insert(device_id, pos);
                                        return Ok((
                                            data,
                                            vec![DraggableEvent::OnDrag(diff / dpi)],
                                        ));
                                    }
                                }
                                if let Some(last_pos) = data.lastdrag.get(&device_id).copied() {
                                    data.lastdrag.insert(device_id, pos);
                                    return Ok((
                                        data,
                                        vec![DraggableEvent::OnDrag((pos - last_pos) / dpi)],
                                    ));
                                }
                            }
                        }
                        RawEvent::Mouse {
                            device_id,
                            state,
                            pos,
                            ..
                        } => {
                            let pos = crate::pixel_to_vec(pos);
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
                                            return Ok((
                                                data,
                                                vec![DraggableEvent::OnClick(pos / dpi)],
                                            ));
                                        }
                                    }
                                    if let Some(last_pos) = data.lastdrag.get(&device_id).copied() {
                                        data.lastdrag.remove(&device_id);
                                        return Ok((
                                            data,
                                            vec![DraggableEvent::OnDrag((pos - last_pos) / dpi)],
                                        ));
                                    }
                                }
                                MouseState::DblClick => {
                                    if data.lastdown.contains_key(&device_id) {
                                        data.lastdown.remove(&device_id);
                                        if area.contains(pos) {
                                            return Ok((
                                                data,
                                                vec![DraggableEvent::OnDblClick(pos / dpi)],
                                            ));
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
                                    let last_pos =
                                        data.lasttouch.remove(&(device_id, index)).unwrap();
                                    data.lasttouchmove.insert((device_id, index), pos);
                                    return Ok((
                                        data,
                                        vec![DraggableEvent::OnDrag((pos - last_pos).xy() / dpi)],
                                    ));
                                }
                                if let Some(last_pos) =
                                    data.lasttouchmove.get(&(device_id, index)).copied()
                                {
                                    data.lasttouchmove.insert((device_id, index), pos);
                                    return Ok((
                                        data,
                                        vec![DraggableEvent::OnDrag((pos - last_pos).xy() / dpi)],
                                    ));
                                }
                            }
                            crate::input::TouchState::End => {
                                if data.lasttouch.contains_key(&(device_id, index)) {
                                    if area.contains(pos.xy()) {
                                        return Ok((
                                            data,
                                            vec![DraggableEvent::OnClick(pos.xy() / dpi)],
                                        ));
                                    }
                                    data.lasttouch.remove(&(device_id, index));
                                }
                                if let Some(last_pos) =
                                    data.lasttouchmove.get(&(device_id, index)).copied()
                                {
                                    data.lasttouchmove.remove(&(device_id, index));
                                    return Ok((
                                        data,
                                        vec![DraggableEvent::OnDrag((pos - last_pos).xy() / dpi)],
                                    ));
                                }
                            }
                        },
                        _ => (),
                    }
                    Err((data, vec![]))
                },
            ),
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
            manager.init_component(child.as_ref().unwrap().as_ref())?;
        }
        Ok(())
    }

    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &crate::DriverState,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let map = VectorMap::new(
            |child: &Option<Box<ComponentFrom<dyn fixed::Prop>>>| -> Option<Box<dyn Layout<<dyn fixed::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(state, driver, window,config))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<T, dyn fixed::Prop> {
            props: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}

crate::gen_component_wrap!(Draggable, fixed::Prop);
