// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::StateMachine;
use crate::component::Layout;
use crate::input::{MouseState, RawEvent, RawEventKind};
use crate::layout::leaf;
use crate::{layout, pixel_to_vec, Dispatchable, Slot, SourceID};
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Debug, Dispatch, EnumVariantType, Clone)]
#[evt(derive(Clone), module = "mouse_area_event")]
pub enum MouseAreaEvent {
    OnClick,
    Hover,
    Active,
}

#[derive(Default, Clone)]
struct MouseAreaState {
    lastdown: HashSet<winit::event::DeviceId>,
    lasttouch: HashSet<(winit::event::DeviceId, u64)>,
}

#[derive_where(Clone)]
pub struct MouseArea<T: leaf::Prop + 'static> {
    pub id: Rc<SourceID>,
    props: Rc<T>,
    slots: [Option<Slot>; MouseAreaEvent::SIZE],
}

impl<T: leaf::Prop + 'static> MouseArea<T> {
    pub fn new(id: Rc<SourceID>, props: T, slots: [Option<Slot>; MouseAreaEvent::SIZE]) -> Self {
        Self {
            id,
            props: props.into(),
            slots,
        }
    }
}

impl<T: leaf::Prop + 'static> super::Component<T> for MouseArea<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Prop + 'static)>,
{
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, _: &mut crate::StateManager) -> eyre::Result<()> {
        Ok(())
    }

    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        let onclick = Box::new(
            crate::wrap_event::<RawEvent, MouseAreaEvent, MouseAreaState>(
                |e, area, _dpi, mut data| {
                    match e {
                        RawEvent::Mouse {
                            device_id,
                            state,
                            pos,
                            ..
                        } => {
                            match state {
                                MouseState::Down => {
                                    if area.contains(pixel_to_vec(pos)) {
                                        data.lastdown.insert(device_id);
                                        return Ok((data, vec![]));
                                    }
                                } // TODO: if this is inside the area, capture the mouse input.
                                MouseState::Up => {
                                    if data.lastdown.contains(&device_id) {
                                        data.lastdown.remove(&device_id);
                                        if area.contains(pixel_to_vec(pos)) {
                                            return Ok((data, vec![MouseAreaEvent::OnClick]));
                                        }
                                    }
                                }
                                MouseState::DblClick => (), // TODO: some OSes may send this *instead of* a mouse up, in which case this should be treated as one.
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
                                    data.lasttouch.insert((device_id, index));
                                    return Ok((data, vec![]));
                                }
                            }
                            crate::input::TouchState::Move => (),
                            crate::input::TouchState::End => {
                                if data.lasttouch.contains(&(device_id, index)) {
                                    if area.contains(pos.xy()) {
                                        return Ok((data, vec![MouseAreaEvent::OnClick]));
                                    }
                                    data.lasttouch.remove(&(device_id, index));
                                }
                            }
                        },
                        _ => (),
                    }
                    Err((data, vec![]))
                },
            ),
        );

        //<MouseAreaEvent, MouseAreaState, 1, 3>
        Ok(Box::new(StateMachine {
            state: Some(Default::default()),
            input: [(
                RawEventKind::Mouse as u64 | RawEventKind::Touch as u64,
                onclick,
            )],
            output: self.slots.clone(),
        }))
    }

    fn layout(
        &self,
        _: &crate::StateManager,
        _: &crate::DriverState,
        _: crate::Vec2,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T> + 'static> {
        Box::new(layout::Node::<T, dyn leaf::Prop> {
            props: self.props.clone(),
            children: Default::default(),
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
