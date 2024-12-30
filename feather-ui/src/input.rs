// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2024 Fundament Software SPC <https://fundament.software>

use crate::AbsRect;
use crate::Dispatchable;
use crate::Error;
use enum_variant_type::EnumVariantType;
use std::any::Any;
use std::any::TypeId;
use ultraviolet::Vec2;
use ultraviolet::Vec3;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum TouchState {
    Start = 0,
    Move = 1,
    End = 2,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum MouseState {
    Down = 0,
    Up = 1,
    DblClick = 2,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum MouseMoveState {
    Move = 0,
    On = 1,
    Off = 2,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum MouseButton {
    L = 1,
    R = 2,
    M = 4,
    X1 = 8,
    X2 = 16,
    X3 = 32,
    X4 = 64,
    X5 = 128,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum ModifierKeys {
    Shift = 1,
    Control = 2,
    Alt = 4,
    Super = 8,
    Capslock = 16,
    Numlock = 32,
    Held = 64,
}

#[derive(Debug, EnumVariantType, Clone)]
#[evt(derive(Clone), module = "raw_event")]
pub enum RawEvent {
    Draw(AbsRect),
    Drop,
    Focus {
        acquired: bool,
    },
    JoyAxis {
        device_id: winit::event::DeviceId,
        value: f64,
        axis: u32,
    },
    JoyButton {
        device_id: winit::event::DeviceId,
        down: bool,
        button: u32,
    },
    JoyOrientation {
        device_id: winit::event::DeviceId,
        velocity: Vec3,
        rotation: Vec3,
    },
    Key {
        device_id: winit::event::DeviceId,
        physical_key: winit::keyboard::PhysicalKey,
        location: winit::keyboard::KeyLocation,
        down: bool,
        logical_key: winit::keyboard::Key,
        modifiers: u8,
    },
    Mouse {
        device_id: winit::event::DeviceId,
        state: MouseState,
        pos: Vec2,
        button: MouseButton,
        all_buttons: u8,
        modifiers: u8,
    },
    MouseMove {
        device_id: winit::event::DeviceId,
        state: MouseMoveState,
        pos: Vec2,
        all_buttons: u8,
        modifiers: u8,
    },
    MouseScroll {
        device_id: winit::event::DeviceId,
        state: TouchState,
        pos: Vec2,
        delta: Vec2,
        pixels: bool, // If true, delta is expressed in pixels
    },
    Touch {
        device_id: winit::event::DeviceId,
        index: u64,
        state: TouchState,
        pos: Vec3,
        angle: Vec2,
        pressure: f64,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum RawEventKind {
    Draw = 1,
    Drop = 2,
    Focus = 4,
    JoyAxis = 8,
    JoyButton = 16,
    JoyOrientation = 32,
    Key = 64,
    Mouse = 128,
    MouseMove = 256,
    MouseScroll = 512,
    Touch = 1024,
}

use crate::DispatchPair;

impl Dispatchable for RawEvent {
    fn restore(pair: DispatchPair) -> Result<Self, Error> {
        const KIND_DRAW: u64 = RawEventKind::Draw as u64;
        const KIND_DROP: u64 = RawEventKind::Drop as u64;
        const KIND_FOCUS: u64 = RawEventKind::Focus as u64;
        const KIND_JOYAXIS: u64 = RawEventKind::JoyAxis as u64;
        const KIND_JOYBUTTON: u64 = RawEventKind::JoyButton as u64;
        const KIND_JOYORIENTATION: u64 = RawEventKind::JoyOrientation as u64;
        const KIND_KEY: u64 = RawEventKind::Key as u64;
        const KIND_MOUSE: u64 = RawEventKind::Mouse as u64;
        const KIND_MOUSEMOVE: u64 = RawEventKind::MouseMove as u64;
        const KIND_MOUSESCROLL: u64 = RawEventKind::MouseScroll as u64;
        const KIND_TOUCH: u64 = RawEventKind::Touch as u64;
        let type_id = pair.1.type_id();

        match pair.0 {
            KIND_DRAW => Ok(RawEvent::from(
                *pair.1.downcast::<raw_event::Draw>().map_err(|_| {
                    Error::MismatchedEnumTag(pair.0, TypeId::of::<raw_event::Draw>(), type_id)
                })?,
            )),
            KIND_DROP => Ok(RawEvent::from(
                *pair.1.downcast::<raw_event::Drop>().map_err(|_| {
                    Error::MismatchedEnumTag(pair.0, TypeId::of::<raw_event::Drop>(), type_id)
                })?,
            )),
            KIND_FOCUS => Ok(RawEvent::from(
                *pair.1.downcast::<raw_event::Focus>().map_err(|_| {
                    Error::MismatchedEnumTag(pair.0, TypeId::of::<raw_event::Focus>(), type_id)
                })?,
            )),
            KIND_JOYAXIS => Ok(RawEvent::from(
                *pair.1.downcast::<raw_event::JoyAxis>().map_err(|_| {
                    Error::MismatchedEnumTag(pair.0, TypeId::of::<raw_event::JoyAxis>(), type_id)
                })?,
            )),
            KIND_JOYBUTTON => Ok(RawEvent::from(
                *pair.1.downcast::<raw_event::JoyButton>().map_err(|_| {
                    Error::MismatchedEnumTag(pair.0, TypeId::of::<raw_event::JoyButton>(), type_id)
                })?,
            )),
            KIND_JOYORIENTATION => Ok(RawEvent::from(
                *pair
                    .1
                    .downcast::<raw_event::JoyOrientation>()
                    .map_err(|_| {
                        Error::MismatchedEnumTag(
                            pair.0,
                            TypeId::of::<raw_event::JoyOrientation>(),
                            type_id,
                        )
                    })?,
            )),
            KIND_KEY => Ok(RawEvent::from(
                *pair.1.downcast::<raw_event::Key>().map_err(|_| {
                    Error::MismatchedEnumTag(pair.0, TypeId::of::<raw_event::Key>(), type_id)
                })?,
            )),
            KIND_MOUSE => Ok(RawEvent::from(
                *pair.1.downcast::<raw_event::Mouse>().map_err(|_| {
                    Error::MismatchedEnumTag(pair.0, TypeId::of::<raw_event::Mouse>(), type_id)
                })?,
            )),
            KIND_MOUSEMOVE => Ok(RawEvent::from(
                *pair.1.downcast::<raw_event::MouseMove>().map_err(|_| {
                    Error::MismatchedEnumTag(pair.0, TypeId::of::<raw_event::MouseMove>(), type_id)
                })?,
            )),
            KIND_MOUSESCROLL => Ok(RawEvent::from(
                *pair.1.downcast::<raw_event::MouseScroll>().map_err(|_| {
                    Error::MismatchedEnumTag(
                        pair.0,
                        TypeId::of::<raw_event::MouseScroll>(),
                        type_id,
                    )
                })?,
            )),
            KIND_TOUCH => Ok(RawEvent::from(
                *pair.1.downcast::<raw_event::Touch>().map_err(|_| {
                    Error::MismatchedEnumTag(pair.0, TypeId::of::<raw_event::Touch>(), type_id)
                })?,
            )),
            _ => Err(Error::InvalidEnumTag(pair.0)),
        }
    }

    const SIZE: usize = 11;

    fn extract(self) -> DispatchPair {
        match self {
            RawEvent::Draw(_) => (
                RawEventKind::Draw as u64,
                Box::new(raw_event::Draw::try_from(self).unwrap()),
            ),
            RawEvent::Drop => (
                RawEventKind::Drop as u64,
                Box::new(raw_event::Drop::try_from(self).unwrap()),
            ),
            RawEvent::Focus { .. } => (
                RawEventKind::Focus as u64,
                Box::new(raw_event::Focus::try_from(self).unwrap()),
            ),
            RawEvent::JoyAxis { .. } => (
                RawEventKind::JoyAxis as u64,
                Box::new(raw_event::JoyAxis::try_from(self).unwrap()),
            ),
            RawEvent::JoyButton { .. } => (
                RawEventKind::JoyButton as u64,
                Box::new(raw_event::JoyButton::try_from(self).unwrap()),
            ),
            RawEvent::JoyOrientation { .. } => (
                RawEventKind::JoyOrientation as u64,
                Box::new(raw_event::JoyOrientation::try_from(self).unwrap()),
            ),
            RawEvent::Key { .. } => (
                RawEventKind::Key as u64,
                Box::new(raw_event::Key::try_from(self).unwrap()),
            ),
            RawEvent::Mouse { .. } => (
                RawEventKind::Mouse as u64,
                Box::new(raw_event::Mouse::try_from(self).unwrap()),
            ),
            RawEvent::MouseMove { .. } => (
                RawEventKind::MouseMove as u64,
                Box::new(raw_event::MouseMove::try_from(self).unwrap()),
            ),
            RawEvent::MouseScroll { .. } => (
                RawEventKind::MouseScroll as u64,
                Box::new(raw_event::MouseScroll::try_from(self).unwrap()),
            ),
            RawEvent::Touch { .. } => (
                RawEventKind::Touch as u64,
                Box::new(raw_event::Touch::try_from(self).unwrap()),
            ),
        }
    }
}

impl From<winit::event::TouchPhase> for TouchState {
    fn from(value: winit::event::TouchPhase) -> Self {
        match value {
            winit::event::TouchPhase::Started => TouchState::Start,
            winit::event::TouchPhase::Moved => TouchState::Move,
            winit::event::TouchPhase::Ended => TouchState::End,
            winit::event::TouchPhase::Cancelled => TouchState::End,
        }
    }
}

impl From<winit::event::MouseButton> for MouseButton {
    fn from(value: winit::event::MouseButton) -> Self {
        match value {
            winit::event::MouseButton::Left => MouseButton::L,
            winit::event::MouseButton::Right => MouseButton::R,
            winit::event::MouseButton::Middle => MouseButton::M,
            winit::event::MouseButton::Back => MouseButton::X1,
            winit::event::MouseButton::Forward => MouseButton::X2,
            winit::event::MouseButton::Other(5) => MouseButton::X3,
            winit::event::MouseButton::Other(6) => MouseButton::X4,
            winit::event::MouseButton::Other(7) => MouseButton::X5,
            winit::event::MouseButton::Other(_) => panic!("Mouse button out of range"),
        }
    }
}
