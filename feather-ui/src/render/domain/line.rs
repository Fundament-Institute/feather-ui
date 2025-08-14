// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::sRGB;
use crate::render::compositor::{self, DataFlags};
use crate::{CrossReferenceDomain, SourceID};

use std::sync::Arc;

pub struct Instance {
    pub domain: Arc<CrossReferenceDomain>,
    pub start: Arc<SourceID>,
    pub end: Arc<SourceID>,
    pub color: sRGB,
}

impl super::Renderable for Instance {
    fn render(
        &self,
        _: crate::AnyRect,
        _: &crate::graphics::Driver,
        compositor: &mut compositor::CompositorView<'_>,
    ) -> Result<(), crate::Error> {
        let domain = self.domain.clone();
        let start_id = self.start.clone();
        let end_id = self.end.clone();
        let color = self.color.as_32bit();

        compositor.defer(move |_, data| {
            let start = domain.get_area(&start_id).unwrap_or_default();
            let end = domain.get_area(&end_id).unwrap_or_default();

            let p1 = (start.topleft() + start.bottomright().to_vector()) * 0.5;
            let p2 = (end.topleft() + end.bottomright().to_vector()) * 0.5;
            let p = p2 - p1;

            *data = compositor::Data {
                pos: (((p1 + p2.to_vector()) * 0.5)
                    - (crate::AnyVector::new(p.length() * 0.5, 0.0)))
                .to_array()
                .into(),
                dim: [p.length(), 1.0].into(),
                uv: [0.0, 0.0].into(),
                uvdim: [0.0, 0.0].into(),
                color: color.rgba,
                rotation: p.y.atan2(p.x) % std::f32::consts::TAU,
                flags: DataFlags::new().with_tex(u8::MAX).into(),
                ..Default::default()
            };
        });

        Ok(())
    }
}
