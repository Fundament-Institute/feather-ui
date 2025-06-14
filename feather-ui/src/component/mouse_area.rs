// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::StateMachine;
use crate::component::Layout;
use crate::input::{MouseButton, MouseState, RawEvent, RawEventKind};
use crate::layout::leaf;
use crate::{Dispatchable, Slot, SourceID, layout};
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use std::collections::HashMap;
use std::rc::Rc;
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
}

#[derive_where(Clone)]
pub struct MouseArea<T: leaf::Prop + 'static> {
    pub id: Rc<SourceID>,
    props: Rc<T>,
    deadzone: f32, // A deadzone of infinity disables drag events
    slots: [Option<Slot>; MouseAreaEvent::SIZE],
}

impl<T: leaf::Prop + 'static> MouseArea<T> {
    pub fn new(
        id: Rc<SourceID>,
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

impl<T: leaf::Prop + 'static> crate::StateMachineChild for MouseArea<T> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }
    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        let deadzone = self.deadzone;
        let oninput = Box::new(
            crate::wrap_event::<RawEvent, MouseAreaEvent, MouseAreaState>(
                move |e, area, dpi, mut data| {
                    match e {
                        RawEvent::Key {
                            down,
                            logical_key: winit::keyboard::Key::Named(code),
                            ..
                        } => {
                            if (code == NamedKey::Enter || code == NamedKey::Accept) && down {
                                return Ok((
                                    data,
                                    vec![MouseAreaEvent::OnClick(
                                        crate::input::MouseButton::Left,
                                        Vec2::zero(),
                                    )],
                                ));
                            }
                        }
                        RawEvent::MouseOn { all_buttons, .. }
                        | RawEvent::MouseOff { all_buttons, .. } => {
                            data.hover = matches!(e, RawEvent::MouseOff { .. });
                            let hover = Self::hover_event(all_buttons, data.hover);
                            return Ok((data, vec![hover]));
                        }
                        RawEvent::MouseMove {
                            device_id,
                            pos,
                            all_buttons,
                            ..
                        } => {
                            let hover = Self::hover_event(all_buttons, data.hover);
                            let pos = crate::graphics::pixel_to_vec(pos);
                            for i in 0..5 {
                                if let Some((last_pos, drag)) =
                                    data.lastdown.get_mut(&(device_id, (1 << i)))
                                {
                                    let diff = pos - *last_pos;
                                    if !*drag && diff.dot(diff) > deadzone {
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
                                            data,
                                            vec![hover, MouseAreaEvent::OnDrag(b, diff / dpi)],
                                        ));
                                    }
                                }
                            }

                            return Ok((data, vec![hover]));
                        }
                        RawEvent::Mouse {
                            device_id,
                            state,
                            pos,
                            button,
                            ..
                        } => {
                            let pos = crate::graphics::pixel_to_vec(pos);
                            let hover = Self::hover_event(button as u16, data.hover);
                            match state {
                                MouseState::Down => {
                                    if area.contains(pos) {
                                        data.lastdown
                                            .insert((device_id, button as u64), (pos, false));
                                        return Ok((data, vec![hover]));
                                    }
                                }
                                MouseState::Up => {
                                    if let Some((last_pos, drag)) =
                                        data.lastdown.remove(&(device_id, button as u64))
                                    {
                                        if area.contains(pos) {
                                            return Ok((
                                                data,
                                                vec![
                                                    if drag {
                                                        let diff = pos - last_pos;
                                                        MouseAreaEvent::OnDrag(button, diff / dpi)
                                                    } else {
                                                        MouseAreaEvent::OnClick(button, pos / dpi)
                                                    },
                                                    hover,
                                                ],
                                            ));
                                        }
                                    }
                                }
                                MouseState::DblClick => {
                                    if let Some((last_pos, drag)) =
                                        data.lastdown.remove(&(device_id, button as u64))
                                    {
                                        if area.contains(pos) {
                                            return Ok((
                                                data,
                                                if drag {
                                                    vec![
                                                        MouseAreaEvent::OnClick(button, pos / dpi),
                                                        MouseAreaEvent::OnDblClick(
                                                            button,
                                                            pos / dpi,
                                                        ),
                                                        hover,
                                                    ]
                                                } else {
                                                    vec![
                                                        if drag {
                                                            let diff = pos - last_pos;
                                                            MouseAreaEvent::OnDrag(
                                                                button,
                                                                diff / dpi,
                                                            )
                                                        } else {
                                                            MouseAreaEvent::OnClick(
                                                                button,
                                                                pos / dpi,
                                                            )
                                                        },
                                                        hover,
                                                    ]
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
                                let hover = Self::hover_event(MouseButton::Left as u16, data.hover);
                                if area.contains(pos.xy()) {
                                    data.lastdown.insert((device_id, index), (pos.xy(), false));
                                    return Ok((data, vec![hover]));
                                }
                            }
                            crate::input::TouchState::Move => {
                                let hover = Self::hover_event(MouseButton::Left as u16, data.hover);
                                if let Some((last_pos, drag)) =
                                    data.lastdown.get_mut(&(device_id, index))
                                {
                                    let diff = pos.xy() - *last_pos;
                                    if !*drag && diff.dot(diff) > deadzone {
                                        *drag = true;
                                    }
                                    if *drag {
                                        return Ok((
                                            data,
                                            vec![
                                                hover,
                                                MouseAreaEvent::OnDrag(
                                                    MouseButton::Left,
                                                    diff / dpi,
                                                ),
                                            ],
                                        ));
                                    }
                                }
                            }
                            crate::input::TouchState::End => {
                                let hover = Self::hover_event(0, data.hover);
                                if let Some((_, drag)) = data.lastdown.remove(&(device_id, index)) {
                                    if area.contains(pos.xy()) {
                                        return Ok((
                                            data,
                                            vec![
                                                if drag {
                                                    MouseAreaEvent::OnDrag(
                                                        MouseButton::Left,
                                                        pos.xy() / dpi,
                                                    )
                                                } else {
                                                    MouseAreaEvent::OnClick(
                                                        MouseButton::Left,
                                                        pos.xy() / dpi,
                                                    )
                                                },
                                                hover,
                                            ],
                                        ));
                                    }
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
                RawEventKind::Mouse as u64
                    | RawEventKind::MouseMove as u64
                    | RawEventKind::Touch as u64
                    | RawEventKind::Key as u64,
                oninput,
            )],
            output: self.slots.clone(),
        }))
    }
}

impl<T: leaf::Prop + 'static> super::Component<T> for MouseArea<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Prop + 'static)>,
{
    fn layout(
        &self,
        _: &crate::StateManager,
        _: &crate::graphics::Driver,
        _: &Rc<SourceID>,
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

crate::gen_component_wrap!(MouseArea, leaf::Prop);
