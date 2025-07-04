// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;

use super::compositor::CompositorView;
use ultraviolet::Vec2;

pub struct Instance {
    pub start: Vec2,
    pub end: Vec2,
    pub color: sRGB,
}

impl super::Renderable for Instance {
    fn render(
        &self,
        _: crate::AbsRect,
        _: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), crate::Error> {
        let p1 = self.start;
        let p2 = self.end;

        let p = p2 - p1;
        compositor.append_data(
            (((p1 + p2) * 0.5) - (Vec2::new(p.mag() * 0.5, 0.0)))
                .as_array()
                .into(),
            [p.mag(), 1.0].into(),
            [0.0, 0.0].into(),
            [0.0, 0.0].into(),
            self.color.as_32bit().rgba,
            p.y.atan2(p.x) % std::f32::consts::TAU,
            u16::MAX,
            false,
        );
        Ok(())
    }
}
