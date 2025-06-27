// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use ultraviolet::{Vec2, Vec3};
use winit::dpi::PhysicalPosition;

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
#[repr(u16)]
pub enum MouseButton {
    Left = (1 << 0),
    Right = (1 << 1),
    Middle = (1 << 2),
    Back = (1 << 3),
    Forward = (1 << 4),
    X1 = (1 << 5),
    X2 = (1 << 6),
    X3 = (1 << 7),
    X4 = (1 << 8),
    X5 = (1 << 9),
    X6 = (1 << 10),
    X7 = (1 << 11),
    X8 = (1 << 12),
    X9 = (1 << 13),
    X10 = (1 << 14),
    X11 = (1 << 15),
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
        device_id: winit::event::DeviceId,
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
        // 32 bytes
        device_id: winit::event::DeviceId,
        velocity: Vec3,
        rotation: Vec3,
    },
    Key {
        // 48 bytes
        device_id: winit::event::DeviceId,
        physical_key: winit::keyboard::PhysicalKey,
        location: winit::keyboard::KeyLocation,
        down: bool,
        logical_key: winit::keyboard::Key,
        modifiers: u8,
    },
    Mouse {
        // 24 bytes
        device_id: winit::event::DeviceId,
        state: MouseState,
        pos: PhysicalPosition<f32>,
        button: MouseButton,
        all_buttons: u16,
        modifiers: u8,
    },
    MouseOn {
        device_id: winit::event::DeviceId,
        pos: PhysicalPosition<f32>,
        modifiers: u8,
        all_buttons: u16,
    },
    MouseMove {
        device_id: winit::event::DeviceId,
        pos: PhysicalPosition<f32>,
        modifiers: u8,
        all_buttons: u16,
    },
    MouseOff {
        device_id: winit::event::DeviceId,
        modifiers: u8,
        all_buttons: u16,
    },
    MouseScroll {
        device_id: winit::event::DeviceId,
        state: TouchState,
        pos: PhysicalPosition<f32>,
        delta: Vec2,
        pixels: bool, // If true, delta is expressed in pixels
    },
    Touch {
        // 48 bytes
        device_id: winit::event::DeviceId,
        index: u64,
        state: TouchState,
        pos: Vec3,
        angle: Vec2,
        pressure: f64,
    },
}

static_assertions::const_assert!(size_of::<RawEvent>() == 48);

impl RawEvent {
    pub fn kind(&self) -> RawEventKind {
        self.into()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum RawEventKind {
    Drag = (1 << 0), // This must start from 1 and perfectly match RawEvent to ensure the dispatch works correctly
    Drop = (1 << 1),
    Focus = (1 << 2),
    JoyAxis = (1 << 3),
    JoyButton = (1 << 4),
    JoyOrientation = (1 << 5),
    Key = (1 << 6),
    Mouse = (1 << 7),
    MouseOn = (1 << 8),
    MouseMove = (1 << 9),
    MouseOff = (1 << 10),
    MouseScroll = (1 << 11),
    Touch = (1 << 12),
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
            RawEvent::MouseOn { .. } => RawEventKind::MouseOn,
            RawEvent::MouseMove { .. } => RawEventKind::MouseMove,
            RawEvent::MouseOff { .. } => RawEventKind::MouseOff,
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
            winit::event::MouseButton::Left => MouseButton::Left,
            winit::event::MouseButton::Right => MouseButton::Right,
            winit::event::MouseButton::Middle => MouseButton::Middle,
            winit::event::MouseButton::Back => MouseButton::Back,
            winit::event::MouseButton::Forward => MouseButton::Forward,
            winit::event::MouseButton::Other(5) => MouseButton::X1,
            winit::event::MouseButton::Other(6) => MouseButton::X2,
            winit::event::MouseButton::Other(7) => MouseButton::X3,
            winit::event::MouseButton::Other(8) => MouseButton::X4,
            winit::event::MouseButton::Other(9) => MouseButton::X5,
            winit::event::MouseButton::Other(10) => MouseButton::X6,
            winit::event::MouseButton::Other(11) => MouseButton::X7,
            winit::event::MouseButton::Other(12) => MouseButton::X8,
            winit::event::MouseButton::Other(13) => MouseButton::X9,
            winit::event::MouseButton::Other(14) => MouseButton::X10,
            winit::event::MouseButton::Other(15) => MouseButton::X11,
            winit::event::MouseButton::Other(_) => panic!("Mouse button out of range"),
        }
    }
}
