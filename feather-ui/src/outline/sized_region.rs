// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout;
use crate::layout::simple;
use crate::layout::Desc;
use crate::layout::Layout;
use crate::outline::OutlineFrom;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::Outline;
use crate::SourceID;
use derive_where::derive_where;
use std::rc::Rc;

#[derive_where(Clone)]
pub struct SizedRegion<T: simple::Prop + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub children: im::Vector<Option<Box<dyn Outline<()>>>>,
}

impl<Parent: Clone + 'static> super::Outline<Parent> for SizedRegion<Parent> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, manager: &mut crate::StateManager) -> eyre::Result<()> {
        for child in self.children.iter() {
            manager.init_outline(child.as_ref().unwrap().as_ref())?;
        }
        Ok(())
    }

    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &crate::DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Rc<dyn simple::Prop>>> {
        let map = VectorMap::new(
            |child: &Option<Box<dyn Outline<()>>>| -> Option<Box<dyn Layout<()>>> {
                Some(child.as_ref().unwrap().layout(state, driver, config))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<Rc<dyn simple::Prop>> {
            props: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
