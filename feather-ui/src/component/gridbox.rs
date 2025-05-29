// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Desc, Layout, grid};
use crate::persist::{FnPersist, VectorMap};
use crate::{SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;

use super::ComponentFrom;

#[derive_where(Clone)]
pub struct GridBox<T: grid::Prop + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub children: im::Vector<Option<Box<ComponentFrom<dyn grid::Prop>>>>,
}

impl<T: grid::Prop + 'static> crate::StateMachineChild for GridBox<T> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn apply_children(
        &self,
        f: &mut dyn FnMut(&dyn crate::StateMachineChild) -> eyre::Result<()>,
    ) -> eyre::Result<()> {
        self.children
            .iter()
            .try_for_each(|x| f(x.as_ref().unwrap().as_ref()))
    }
}

impl<T: grid::Prop + 'static> super::Component<T> for GridBox<T> {
    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &crate::DriverState,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let map = VectorMap::new(
            |child: &Option<Box<ComponentFrom<dyn grid::Prop>>>| -> Option<Box<dyn Layout<<dyn grid::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(state, driver,window, config))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<T, dyn grid::Prop> {
            props: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}

crate::gen_component_wrap!(GridBox, grid::Prop);
