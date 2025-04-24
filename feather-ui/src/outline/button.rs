// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::mouse_area::MouseArea;
use crate::layout::fixed;
use crate::layout::Layout;
use crate::layout::LayoutWrap;
use crate::outline::Desc;
use crate::outline::OutlineFrom;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::Outline;
use crate::SourceID;
use crate::URect;
use crate::{layout, Slot};
use derive_where::derive_where;
use std::rc::Rc;

// A button component that contains a mousearea alongside it's children
#[derive_where(Clone)]
pub struct Button<T: fixed::Prop + 'static> {
    pub id: Rc<SourceID>,
    props: Rc<T>,
    marea: MouseArea<URect>,
    children: im::Vector<Option<Box<OutlineFrom<dyn fixed::Prop>>>>,
}

impl<T: fixed::Prop + 'static> Button<T> {
    pub fn new(
        id: Rc<SourceID>,
        props: T,
        onclick: Slot,
        children: im::Vector<Option<Box<OutlineFrom<dyn fixed::Prop>>>>,
    ) -> Self {
        Self {
            id: id.clone(),
            props: props.into(),
            marea: MouseArea::new(
                SourceID {
                    parent: Some(id.clone()),
                    id: crate::DataID::Named("__marea_internal__"),
                }
                .into(),
                crate::FILL_URECT,
                [Some(onclick), None, None],
            ),
            children,
        }
    }
}

impl<T: fixed::Prop + 'static> Outline<T> for Button<T>
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
        let blah: &dyn Outline<URect> = &self.marea;
        manager.init_outline::<URect>(&blah)?;
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

        let (_, mut children) = map.call(Default::default(), &self.children);
        let test = self.marea.layout(state, driver, config);
        children.push_back(Some(Box::new(test)));

        Box::new(layout::Node::<T, dyn fixed::Prop> {
            props: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}

crate::gen_outline_wrap!(Button, fixed::Prop);
