// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::Renderable;
use crate::{AnyRect, CrossReferenceDomain, SourceID};
use std::rc::Rc;
use std::sync::Arc;

pub mod line;

pub struct Write {
    pub(crate) id: std::sync::Weak<SourceID>,
    pub(crate) domain: Arc<CrossReferenceDomain>,
    pub(crate) base: Option<Rc<dyn Renderable>>,
}

impl Renderable for Write {
    fn render(
        &self,
        area: AnyRect,
        driver: &crate::graphics::Driver,
        compositor: &mut crate::render::CompositorView<'_>,
    ) -> Result<(), crate::Error> {
        if let Some(idref) = self.id.upgrade() {
            self.domain.write_area(idref, area);
        }

        self.base
            .as_ref()
            .map(|x| x.render(area, driver, compositor))
            .unwrap_or(Ok(()))
    }
}
