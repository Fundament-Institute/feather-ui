// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout;
use crate::layout::flex;
use crate::layout::Desc;
use crate::layout::Layout;
use crate::layout::LayoutWrap;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::SourceID;
use derive_where::derive_where;
use std::rc::Rc;

use super::OutlineWrap;

#[derive_where(Clone)]
pub struct FlexBox<T: flex::Prop + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub children: im::Vector<Option<Box<dyn OutlineWrap<<dyn flex::Prop as Desc>::Child>>>>,
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
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let map = VectorMap::new(
            |child: &Option<Box<dyn OutlineWrap<<dyn flex::Prop as Desc>::Child>>>| -> Option<Box<dyn LayoutWrap<<dyn flex::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(state, driver, config))
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
