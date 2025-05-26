// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Desc, Layout, list};
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

/*
use std::ops::Deref;

impl<T> Layout<dyn list::Prop> for Box<dyn Layout<T>>
where
    T: list::Prop + 'static,
{
    fn get_props(&self) -> &(dyn list::Prop + 'static) {
        let a = Box::<(dyn layout::Layout<T> + 'static)>::deref(&self);
        a.get_props()
    }

    fn stage<'a>(
        &self,
        area: crate::AbsRect,
        limits: crate::AbsLimits,
        dpi: ultraviolet::Vec2,
        driver: &crate::DriverState,
    ) -> Box<dyn layout::Staged + 'a> {
        let a = Box::<(dyn layout::Layout<T> + 'static)>::deref(&self);
        a.stage(area, limits, dpi, driver)
    }
}*/

impl<T: list::Prop + 'static> super::Component<dyn list::Prop> for ListBox<T> {
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
    ) -> Box<dyn Layout<dyn list::Prop>> {
        let map = VectorMap::new(
            |child: &Option<Box<ComponentFrom<dyn list::Prop>>>| -> Option<Box<dyn Layout<<dyn list::Prop as Desc>::Child>>> {
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
