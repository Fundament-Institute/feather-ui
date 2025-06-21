// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;

use super::compositor::{Compositor, Data};
use std::cell::RefCell;
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
        compositor.append(&Data::new(
            self.start.borrow().as_array().into(),
            [p.mag(), 1.0].into(),
            [0.0, 0.0].into(),
            [0.0, 0.0].into(),
            self.color.as_32bit().rgba,
            0.1, //p.y.atan2(p.x),
            u16::MAX,
            0,
        ));
        Ok(())
    }
}
