// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use ultraviolet::{Vec2, Vec3};
use winit::dpi::PhysicalPosition;

use crate::DriverState;

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

#[derive(Debug, Dispatch, EnumVariantType, Clone)]
#[evt(derive(Clone), module = "raw_event")]
pub enum RawEvent {
    Drag, // TBD, must be included here so RawEvent matches RawEventKind
    Drop {
        pos: PhysicalPosition<f32>,
    },
    Focus {
        acquired: bool,
        window: std::sync::Arc<winit::window::Window>, // Allows setting IME mode for textboxes
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
        pos: PhysicalPosition<f32>,
        button: MouseButton,
        all_buttons: u8,
        modifiers: u8,
    },
    MouseMove {
        device_id: winit::event::DeviceId,
        state: MouseMoveState,
        pos: PhysicalPosition<f32>,
        all_buttons: u8,
        modifiers: u8,
        driver: std::sync::Weak<DriverState>, // Allows setting our global cursor tracker
    },
    MouseScroll {
        device_id: winit::event::DeviceId,
        state: TouchState,
        pos: PhysicalPosition<f32>,
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

impl RawEvent {
    pub fn kind(&self) -> RawEventKind {
        self.into()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum RawEventKind {
    Drag = 1, // This must start from 1 and perfectly match RawEvent to ensure the dispatch works correctly
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

impl From<&RawEvent> for RawEventKind {
    fn from(value: &RawEvent) -> Self {
        match value {
            RawEvent::Drag => RawEventKind::Drag,
            RawEvent::Drop { .. } => RawEventKind::Drop,
            RawEvent::Focus { .. } => RawEventKind::Focus,
            RawEvent::JoyAxis { .. } => RawEventKind::JoyAxis,
            RawEvent::JoyButton { .. } => RawEventKind::JoyButton,
            RawEvent::JoyOrientation { .. } => RawEventKind::JoyOrientation,
            RawEvent::Key { .. } => RawEventKind::Key,
            RawEvent::Mouse { .. } => RawEventKind::Mouse,
            RawEvent::MouseMove { .. } => RawEventKind::MouseMove,
            RawEvent::MouseScroll { .. } => RawEventKind::MouseScroll,
            RawEvent::Touch { .. } => RawEventKind::Touch,
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
