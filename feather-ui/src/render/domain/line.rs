// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;
use crate::render::compositor;
use crate::{CrossReferenceDomain, SourceID};

use std::{rc::Rc, u16};
use ultraviolet::Vec2;

pub struct Instance {
    pub domain: Rc<CrossReferenceDomain>,
    pub start: Rc<SourceID>,
    pub end: Rc<SourceID>,
    pub color: sRGB,
}

impl super::Renderable for Instance {
    fn render(
        &self,
        _: crate::AbsRect,
        _: &crate::graphics::Driver,
        compositor: &mut compositor::Compositor,
    ) {
        let domain = self.domain.clone();
        let start_id = self.start.clone();
        let end_id = self.end.clone();
        let color = self.color.as_32bit();

        compositor.defer(move |_, data| {
            let start = domain.get_area(&start_id).unwrap_or_default();
            let end = domain.get_area(&end_id).unwrap_or_default();

            let p1: Vec2 = ((start.topleft() + start.bottomright()) * 0.5).into();
            let p2: Vec2 = ((end.topleft() + end.bottomright()) * 0.5).into();
            let p = p2 - p1;

            *data = compositor::Data {
                pos: p1.as_array().into(),
                dim: [p.mag(), 1.0].into(),
                uv: [0, 0].into(),
                uvdim: [0, 0].into(),
                color: color.rgba,
                rotation: p.y.atan2(p.x),
                texclip: ((u16::MAX as u32) << 16) | 0,
                ..Default::default()
            };
        })
    }
}
