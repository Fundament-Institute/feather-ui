use crate::AbsRect;
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

pub type ButtonState = u8;
pub type Angle = f32;
pub type Modifiers = u8;
pub type ScanCode = u16;
pub type DeviceIndex = u16;

#[derive(Debug, Copy, Clone)]
pub enum Event {
    Draw(AbsRect),
    Drop,
    GotFocus,
    JoyAxis(DeviceIndex, f32, u16, ModifierKeys),
    JoyButton(bool, DeviceIndex, u16, ModifierKeys),
    JoyOrientation(DeviceIndex, Vec3, Vec3),
    KeyChar(i32, ModifierKeys),
    Key(bool, u8, ModifierKeys, ScanCode),
    LostFocus,
    Mouse(MouseState, Vec2, ButtonState, ModifierKeys, MouseButton),
    MouseMove(MouseMoveState, Vec2, ButtonState, ModifierKeys),
    MouseScroll(Vec2, f32, f32),
    Touch(TouchState, Vec3, Angle, f32, DeviceIndex, ModifierKeys),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum EventKind {
    Mouse = 1,
    MouseMove = 2,
    MouseScroll = 4,
    Touch = 8,
}
