// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout;
use crate::layout::fixed;
use crate::layout::Desc;
use crate::layout::Layout;
use crate::layout::LayoutWrap;
use crate::outline::OutlineFrom;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::SourceID;
use derive_where::derive_where;
use std::rc::Rc;

#[derive_where(Clone, Default)]
pub struct Region<T: fixed::Prop + Default + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub children: im::Vector<Option<Box<OutlineFrom<dyn fixed::Prop>>>>,
}

impl<T: fixed::Prop + Default + 'static> super::Outline<T> for Region<T>
where
    for<'a> &'a T: Into<&'a (dyn fixed::Prop + 'static)>,
{
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
            |child: &Option<Box<OutlineFrom<dyn fixed::Prop>>>| -> Option<Box<dyn LayoutWrap<<dyn fixed::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(state, driver, config))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<T, dyn fixed::Prop> {
            props: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}

crate::gen_outline_wrap!(Region, fixed::Prop, Default);
