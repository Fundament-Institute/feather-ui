// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::compositor::{Compositor, Data};
use std::cell::RefCell;
use std::u16;
use ultraviolet::Vec2;

pub struct Instance {
    pub start: RefCell<Vec2>,
    pub end: RefCell<Vec2>,
    pub color: u32,
}

impl super::Renderable for Instance {
    fn render(&self, _: crate::AbsRect, _: &crate::graphics::Driver, compositor: &mut Compositor) {
        let p = *self.end.borrow() - *self.start.borrow();
        compositor.append(&Data {
            pos: *self.start.borrow().as_array(),
            dim: [p.mag(), 1.0],
            uv: [0, 0],
            uvdim: [0, 0],
            color: self.color,
            rotation: p.y.atan2(p.x),
            tex: u16::MAX,
            clip: 0,
        });
    }
}
