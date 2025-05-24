// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Desc, Layout, LayoutWrap, list};
use crate::persist::{FnPersist, VectorMap};
use crate::{SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;

use super::ComponentFrom;

#[derive_where(Clone)]
pub struct ListBox<T: list::Prop + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub children: im::Vector<Option<Box<ComponentFrom<dyn list::Prop>>>>,
}

impl<T: list::Prop + 'static> super::Component<T> for ListBox<T> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, manager: &mut crate::StateManager) -> eyre::Result<()> {
        for child in self.children.iter() {
            manager.init_component(child.as_ref().unwrap().as_ref())?;
        }
        Ok(())
    }

    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &crate::DriverState,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let map = VectorMap::new(
            |child: &Option<Box<ComponentFrom<dyn list::Prop>>>| -> Option<Box<dyn LayoutWrap<<dyn list::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(state, driver,window, config))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<T, dyn list::Prop> {
            props: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}

crate::gen_component_wrap!(ListBox, list::Prop);
