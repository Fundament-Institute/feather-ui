// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::mouse_area::MouseArea;
use crate::layout::prop;
use crate::layout::simple;
use crate::layout::Layout;
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
pub struct Button<T: simple::Prop + 'static> {
    pub id: Rc<SourceID>,
    props: Rc<T>,
    marea: MouseArea<URect>,
    children: im::Vector<Option<Box<dyn Outline<()>>>>,
}

impl<T: simple::Prop + 'static> Button<T> {
    pub fn new(
        id: Rc<SourceID>,
        props: T,
        onclick: Slot,
        children: im::Vector<Option<Box<dyn Outline<()>>>>,
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

impl<T: simple::Prop + 'static> Outline<Rc<dyn simple::Prop>> for Button<T> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, manager: &mut crate::StateManager) -> eyre::Result<()> {
        for child in self.children.iter() {
            manager.init_outline(child.as_ref().unwrap().as_ref())?;
        }
        manager.init_outline(&self.marea)?;
        Ok(())
    }

    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &crate::DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Rc<dyn simple::Prop>>> {
        let map = VectorMap::new(
            |child: &Option<Box<dyn Outline<()>>>| -> Option<Box<dyn Layout<Rc<dyn prop::Empty>>>> {
                Some(child.as_ref().unwrap().layout(state, driver, config))
            },
        );

        let (_, mut children) = map.call(Default::default(), &self.children);
        children.push_back(Some(self.marea.layout(state, driver, config)));

        Box::new(layout::Node::<Rc<dyn simple::Prop>> {
            props: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
