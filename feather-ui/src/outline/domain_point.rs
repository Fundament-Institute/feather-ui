// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::CrossReferenceDomain;
use crate::layout::domain_write::DomainWrite;
use crate::SourceID;
use derive_where::derive_where;
use std::rc::Rc;

// This simply writes it's area to the given cross-reference domain during the layout phase
#[derive_where(Clone)]
pub struct DomainPoint<Parent: Clone> {
    pub id: Rc<SourceID>,
    domain: Rc<CrossReferenceDomain>,
    props: Parent,
}

impl<Parent: Clone + 'static> DomainPoint<Parent> {
    pub fn new(id: Rc<SourceID>, props: Parent, domain: Rc<CrossReferenceDomain>) -> Self {
        Self { id, props, domain }
    }
}
impl<Parent: Clone + 'static> super::Outline<Parent> for DomainPoint<Parent> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, _: &mut crate::StateManager) -> eyre::Result<()> {
        Ok(())
    }

    fn layout(
        &self,
        _: &crate::StateManager,
        _: &crate::DriverState,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn crate::layout::Layout<Parent>> {
        Box::new(crate::layout::Node::<DomainWrite, Parent> {
            props: DomainWrite {
                domain: self.domain.clone(),
            },
            imposed: self.props.clone(),
            children: Default::default(),
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
