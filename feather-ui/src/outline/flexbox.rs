// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{flex, Desc, Layout, LayoutWrap};
use crate::persist::{FnPersist, VectorMap};
use crate::{layout, SourceID};
use derive_where::derive_where;
use std::rc::Rc;

use super::OutlineFrom;

#[derive_where(Clone)]
pub struct FlexBox<T: flex::Prop + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub children: im::Vector<Option<Box<OutlineFrom<dyn flex::Prop>>>>,
}

impl<T: flex::Prop + 'static> super::Outline<T> for FlexBox<T> {
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
        dpi: crate::Vec2,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let map = VectorMap::new(
            |child: &Option<Box<OutlineFrom<dyn flex::Prop>>>| -> Option<Box<dyn LayoutWrap<<dyn flex::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(state, driver,dpi, config))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<T, dyn flex::Prop> {
            props: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}

crate::gen_outline_wrap!(FlexBox, flex::Prop);
