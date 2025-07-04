// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;
use crate::render::compositor;
use crate::{CrossReferenceDomain, SourceID};

use std::sync::Arc;
use ultraviolet::Vec2;

pub struct Instance {
    pub domain: Arc<CrossReferenceDomain>,
    pub start: Arc<SourceID>,
    pub end: Arc<SourceID>,
    pub color: sRGB,
}

impl super::Renderable for Instance {
    fn render(
        &self,
        _: crate::AbsRect,
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

            let p1: Vec2 = (start.topleft() + start.bottomright()) * 0.5;
            let p2: Vec2 = (end.topleft() + end.bottomright()) * 0.5;
            let p = p2 - p1;

            *data = compositor::Data {
                pos: (((p1 + p2) * 0.5) - (Vec2::new(p.mag() * 0.5, 0.0)))
                    .as_array()
                    .into(),
                dim: [p.mag(), 1.0].into(),
                uv: [0.0, 0.0].into(),
                uvdim: [0.0, 0.0].into(),
                color: color.rgba,
                rotation: p.y.atan2(p.x) % std::f32::consts::TAU,
                texclip: 0x3FFF0000,
                ..Default::default()
            };
        });

        Ok(())
    }
}
