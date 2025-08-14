// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::sRGB;

use super::compositor::CompositorView;

pub struct Instance {
    pub start: crate::PxPoint,
    pub end: crate::PxPoint,
    pub color: sRGB,
}

impl super::Renderable for Instance {
    fn render(
        &self,
        _: crate::PxRect,
        _: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), crate::Error> {
        let p1 = self.start;
        let p2 = self.end;

        let p = p2 - p1;
        compositor.append_data(
            ((p1 + p2.to_vector()) * 0.5) - (crate::PxVector::new(p.length() * 0.5, 0.0)),
            [p.length(), 1.0].into(),
            [0.0, 0.0].into(),
            [0.0, 0.0].into(),
            self.color.as_32bit().rgba,
            p.y.atan2(p.x) % std::f32::consts::TAU,
            u8::MAX,
            false,
        );
        Ok(())
    }
}
