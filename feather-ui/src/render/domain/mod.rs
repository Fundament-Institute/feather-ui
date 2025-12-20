// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::Renderable;
use crate::component::ComponentMarker;
use crate::reactive::Identity;
use crate::{CrossReferenceDomain, PxPoint, PxRect, sample};
use std::rc::Rc;
use std::sync::Arc;

pub mod line;

pub struct Write<T: Renderable> {
    pub(crate) id: std::sync::Weak<dyn ComponentMarker + Send + Sync>,
    pub(crate) domain: Arc<CrossReferenceDomain>,
    pub(crate) base: Option<T>,
    pub(crate) area: crate::DynSignal<PxRect>,
}

impl<T: Renderable> Renderable for Write<T> {
    fn render(
        &self,
        parent_pos: PxPoint,
        driver: &crate::graphics::Driver,
        compositor: &mut crate::render::CompositorView<'_>,
    ) -> Result<(), crate::Error> {
        if let Some(idref) = self.id.upgrade() {
            self.domain
                .write_area(idref, *sample(&self.area) + parent_pos);
        }

        self.base
            .as_ref()
            .map(|x| x.render(parent_pos, driver, compositor))
            .unwrap_or(Ok(()))
    }
}
