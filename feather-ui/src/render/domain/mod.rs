// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::Renderable;
use crate::{AbsRect, CrossReferenceDomain, SourceID};
use std::rc::Rc;

pub mod line;

pub struct Write {
    pub(crate) id: std::rc::Weak<SourceID>,
    pub(crate) domain: Rc<CrossReferenceDomain>,
    pub(crate) base: Option<Rc<dyn Renderable>>,
}

impl Renderable for Write {
    fn render(
        &self,
        area: AbsRect,
        graphics: &crate::graphics::Driver,
        compositor: &mut crate::render::Compositor,
    ) {
        if let Some(idref) = self.id.upgrade() {
            self.domain.write_area(idref, area);
        }

        if let Some(x) = &self.base {
            x.render(area, graphics, compositor);
        }
    }
}
