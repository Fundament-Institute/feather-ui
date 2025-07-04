// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::StateMachine;
use crate::component::Layout;
use crate::input::{MouseButton, MouseState, RawEvent, RawEventKind};
use crate::layout::leaf;
use crate::{Dispatchable, Slot, SourceID, layout};
use core::f32;
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use ultraviolet::Vec2;
use winit::event::DeviceId;
use winit::keyboard::NamedKey;

#[derive(Debug, Dispatch, EnumVariantType, Clone, PartialEq)]
#[evt(derive(Clone), module = "mouse_area_event")]
pub enum MouseAreaEvent {
    OnClick(MouseButton, Vec2),
    OnDblClick(MouseButton, Vec2),
    OnDrag(MouseButton, Vec2),
    Default,
    Hover,
    Active,
}

#[derive(Default, Clone, PartialEq)]
struct MouseAreaState {
    lastdown: HashMap<(DeviceId, u64), (Vec2, bool)>,
    hover: bool,
    deadzone: f32,
}

impl MouseAreaState {
    fn hover_event(buttons: u16, hover: bool) -> MouseAreaEvent {
        let active = (buttons & MouseButton::Left as u16) != 0;
        match (active, hover) {
            (true, true) => MouseAreaEvent::Active,
            (true, false) => MouseAreaEvent::Hover,
            (false, true) => MouseAreaEvent::Hover,
            (false, false) => MouseAreaEvent::Default,
        }
    }
}

impl super::EventStream for MouseAreaState {
    type Input = RawEvent;
    type Output = MouseAreaEvent;

    fn process(
        mut self,
        input: Self::Input,
        area: crate::AbsRect,
        _: crate::AbsRect,
        dpi: crate::Vec2,
        _: &std::sync::Weak<crate::Driver>,
    ) -> eyre::Result<
        (Self, smallvec::SmallVec<[Self::Output; 1]>),
        (Self, smallvec::SmallVec<[Self::Output; 1]>),
    > {
        match input {
            RawEvent::Key {
                down,
                logical_key: winit::keyboard::Key::Named(code),
                ..
            } => {
                if (code == NamedKey::Enter || code == NamedKey::Accept) && down {
                    return Ok((
                        self,
                        [MouseAreaEvent::OnClick(
                            crate::input::MouseButton::Left,
                            Vec2::zero(),
                        )]
                        .into(),
                    ));
                }
            }
            RawEvent::MouseOn { all_buttons, .. } | RawEvent::MouseOff { all_buttons, .. } => {
                self.hover = matches!(input, RawEvent::MouseOff { .. });
                let hover = Self::hover_event(all_buttons, self.hover);
                return Ok((self, [hover].into()));
            }
            RawEvent::MouseMove {
                device_id,
                pos,
                all_buttons,
                ..
            } => {
                let hover = Self::hover_event(all_buttons, self.hover);
                let pos = crate::graphics::pixel_to_vec(pos);
                for i in 0..5 {
                    if let Some((last_pos, drag)) = self.lastdown.get_mut(&(device_id, (1 << i))) {
                        let diff = pos - *last_pos;
                        if !*drag && diff.dot(diff) > self.deadzone {
                            *drag = true;
                        }

                        let b = match i {
                            0 => MouseButton::Left,
                            1 => MouseButton::Middle,
                            2 => MouseButton::Right,
                            3 => MouseButton::Back,
                            4 => MouseButton::Forward,
                            _ => panic!("Impossible number"),
                        };
                        if *drag {
                            *last_pos = pos;
                            return Ok((
                                self,
                                SmallVec::from_iter([hover, MouseAreaEvent::OnDrag(b, diff / dpi)]),
                            ));
                        }
                    }
                }

                return Ok((self, [hover].into()));
            }
            RawEvent::Mouse {
                device_id,
                state,
                pos,
                button,
                ..
            } => {
                let pos = crate::graphics::pixel_to_vec(pos);
                let hover = Self::hover_event(button as u16, self.hover);
                match state {
                    MouseState::Down => {
                        if area.contains(pos) {
                            self.lastdown
                                .insert((device_id, button as u64), (pos, false));
                            return Ok((self, [hover].into()));
                        }
                    }
                    MouseState::Up => {
                        if let Some((last_pos, drag)) =
                            self.lastdown.remove(&(device_id, button as u64))
                        {
                            if area.contains(pos) {
                                return Ok((
                                    self,
                                    SmallVec::from_iter([
                                        if drag {
                                            let diff = pos - last_pos;
                                            MouseAreaEvent::OnDrag(button, diff / dpi)
                                        } else {
                                            MouseAreaEvent::OnClick(button, pos / dpi)
                                        },
                                        hover,
                                    ]),
                                ));
                            }
                        }
                    }
                    MouseState::DblClick => {
                        if let Some((last_pos, drag)) =
                            self.lastdown.remove(&(device_id, button as u64))
                        {
                            if area.contains(pos) {
                                return Ok((
                                    self,
                                    if drag {
                                        SmallVec::from_iter([
                                            MouseAreaEvent::OnClick(button, pos / dpi),
                                            MouseAreaEvent::OnDblClick(button, pos / dpi),
                                            hover,
                                        ])
                                    } else {
                                        SmallVec::from_iter([
                                            if drag {
                                                let diff = pos - last_pos;
                                                MouseAreaEvent::OnDrag(button, diff / dpi)
                                            } else {
                                                MouseAreaEvent::OnClick(button, pos / dpi)
                                            },
                                            hover,
                                        ])
                                    },
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
                    let hover = Self::hover_event(MouseButton::Left as u16, self.hover);
                    if area.contains(pos.xy()) {
                        self.lastdown.insert((device_id, index), (pos.xy(), false));
                        return Ok((self, [hover].into()));
                    }
                }
                crate::input::TouchState::Move => {
                    let hover = Self::hover_event(MouseButton::Left as u16, self.hover);
                    if let Some((last_pos, drag)) = self.lastdown.get_mut(&(device_id, index)) {
                        let diff = pos.xy() - *last_pos;
                        if !*drag && diff.dot(diff) > self.deadzone {
                            *drag = true;
                        }
                        if *drag {
                            return Ok((
                                self,
                                SmallVec::from_iter([
                                    hover,
                                    MouseAreaEvent::OnDrag(MouseButton::Left, diff / dpi),
                                ]),
                            ));
                        }
                        return Ok((self, SmallVec::new()));
                    }
                }
                crate::input::TouchState::End => {
                    let hover = Self::hover_event(0, self.hover);
                    if let Some((last_pos, drag)) = self.lastdown.remove(&(device_id, index)) {
                        if area.contains(pos.xy()) {
                            let diff = pos.xy() - last_pos;
                            return Ok((
                                self,
                                SmallVec::from_iter([
                                    if drag {
                                        MouseAreaEvent::OnDrag(MouseButton::Left, diff / dpi)
                                    } else {
                                        MouseAreaEvent::OnClick(MouseButton::Left, pos.xy() / dpi)
                                    },
                                    hover,
                                ]),
                            ));
                        }
                    }
                }
            },
            _ => (),
        }
        Err((self, SmallVec::new()))
    }
}

#[derive_where(Clone)]
pub struct MouseArea<T: leaf::Prop + 'static> {
    pub id: Arc<SourceID>,
    props: Rc<T>,
    deadzone: f32, // A deadzone of infinity disables drag events
    slots: [Option<Slot>; MouseAreaEvent::SIZE],
}

impl<T: leaf::Prop + 'static> MouseArea<T> {
    pub fn new(
        id: Arc<SourceID>,
        props: T,
        deadzone: Option<f32>,
        slots: [Option<Slot>; MouseAreaEvent::SIZE],
    ) -> Self {
        Self {
            id,
            props: props.into(),
            deadzone: deadzone.unwrap_or(f32::INFINITY),
            slots,
        }
    }
}

impl<T: leaf::Prop + 'static> crate::StateMachineChild for MouseArea<T> {
    fn id(&self) -> Arc<SourceID> {
        self.id.clone()
    }
    fn init(
        &self,
        _: &std::sync::Weak<crate::Driver>,
    ) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        Ok(Box::new(StateMachine {
            state: Some(MouseAreaState {
                lastdown: HashMap::new(),
                hover: false,
                deadzone: f32::INFINITY,
            }),
            input_mask: RawEventKind::Mouse as u64
                | RawEventKind::MouseMove as u64
                | RawEventKind::Touch as u64
                | RawEventKind::Key as u64,
            output: self.slots.clone(),
            changed: true,
        }))
    }
}

impl<T: leaf::Prop + 'static> super::Component for MouseArea<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Prop + 'static)>,
{
    type Props = T;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        _: &crate::graphics::Driver,
        _: &Arc<SourceID>,
    ) -> Box<dyn Layout<T> + 'static> {
        manager
            .get_mut::<StateMachine<MouseAreaState, { MouseAreaEvent::SIZE }>>(&self.id)
            .and_then(|state| {
                state
                    .state
                    .as_mut()
                    .ok_or(crate::Error::InternalFailure.into())
            })
            .map(|state| {
                state.deadzone = self.deadzone;
            })
            .unwrap();

        Box::new(layout::Node::<T, dyn leaf::Prop> {
            props: self.props.clone(),
            children: Default::default(),
            id: Arc::downgrade(&self.id),
            renderable: None,
        })
    }
}
