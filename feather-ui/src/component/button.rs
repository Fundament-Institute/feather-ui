// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::mouse_area::MouseArea;

use crate::component::{ChildOf, Desc};
use crate::layout::{Layout, fixed};
use crate::persist::{FnPersist, VectorMap};
use crate::{Component, DRect, Slot, SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

// A button component that contains a mousearea alongside it's children
#[derive_where(Clone)]
pub struct Button<T: fixed::Prop + 'static> {
    pub id: Arc<SourceID>,
    props: Rc<T>,
    marea: MouseArea<DRect>,
    children: im::Vector<Option<Box<ChildOf<dyn fixed::Prop>>>>,
}

impl<T: fixed::Prop + 'static> Button<T> {
    pub fn new(
        id: Arc<SourceID>,
        props: T,
        onclick: Slot,
        children: im::Vector<Option<Box<ChildOf<dyn fixed::Prop>>>>,
    ) -> Self {
        super::set_children(Self {
            id: id.clone(),
            props: props.into(),
            marea: MouseArea::new(
                SourceID {
                    parent: id.clone().into(),
                    id: crate::DataID::Named("__marea_internal__"),
                }
                .into(),
                crate::FILL_DRECT,
                None,
                [Some(onclick), None, None, None, None, None],
            ),
            children,
        })
    }
}

impl<T: fixed::Prop + 'static> crate::StateMachineChild for Button<T> {
    fn id(&self) -> Arc<SourceID> {
        self.id.clone()
    }

    fn apply_children(
        &self,
        f: &mut dyn FnMut(&dyn crate::StateMachineChild) -> eyre::Result<()>,
    ) -> eyre::Result<()> {
        for child in self.children.iter() {
            f(child.as_ref().unwrap().as_ref())?;
        }
        f(&self.marea)
    }
}

impl<T: fixed::Prop + 'static> Component for Button<T>
where
    for<'a> &'a T: Into<&'a (dyn fixed::Prop + 'static)>,
{
    type Props = T;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        driver: &crate::graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        let mut map = VectorMap::new(
            |child: &Option<Box<ChildOf<dyn fixed::Prop>>>| -> Option<Box<dyn Layout<<dyn fixed::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(manager, driver, window))
            },
        );

        let (_, mut children) = map.call(Default::default(), &self.children);
        children.push_back(Some(Box::new(self.marea.layout(manager, driver, window))));

        Box::new(layout::Node::<T, dyn fixed::Prop> {
            props: self.props.clone(),
            children,
            id: Arc::downgrade(&self.id),
            renderable: None,
        })
    }
}
