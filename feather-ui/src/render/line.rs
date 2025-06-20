// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;

use super::compositor::{Compositor, Data};
use std::cell::RefCell;
use std::u16;
use ultraviolet::Vec2;

pub struct Instance {
    pub start: RefCell<Vec2>,
    pub end: RefCell<Vec2>,
    pub color: sRGB,
}

impl super::Renderable for Instance {
    fn render(
        &self,
        _: crate::AbsRect,
        _: &crate::graphics::Driver,
        compositor: &mut Compositor,
    ) -> Result<(), crate::Error> {
        let p = *self.end.borrow() - *self.start.borrow();
        compositor.append(&Data {
            pos: self.start.borrow().as_array().into(),
            dim: [p.mag(), 1.0].into(),
            uv: [0.0, 0.0].into(),
            uvdim: [0.0, 0.0].into(),
            color: self.color.as_32bit().rgba,
            rotation: p.y.atan2(p.x),
            texclip: ((u16::MAX as u32) << 16) | 0,
            ..Default::default()
        });
        Ok(())
    }
}
