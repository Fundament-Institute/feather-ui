// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Desc, Layout, list};
use crate::persist::{FnPersist, VectorMap};
use crate::{SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;

use super::ComponentFrom;

#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct ListBox<T: list::Prop + 'static> {
    pub id: Rc<SourceID>,
    props: Rc<T>,
    children: im::Vector<Option<Box<ComponentFrom<dyn list::Prop>>>>,
}

impl<T: list::Prop + 'static> ListBox<T> {
    pub fn new(
        id: Rc<SourceID>,
        props: Rc<T>,
        children: im::Vector<Option<Box<ComponentFrom<dyn list::Prop>>>>,
    ) -> Self {
        super::set_children(Self {
            id,
            props,
            children,
        })
    }
}

impl<T: list::Prop + 'static> super::Component<T> for ListBox<T> {
    fn layout(
        &self,
        state: &mut crate::StateManager,
        driver: &crate::graphics::Driver,
        window: &Rc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        let mut map = VectorMap::new(
            |child: &Option<Box<ComponentFrom<dyn list::Prop>>>| -> Option<Box<dyn Layout<<dyn list::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(state, driver,window))
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
