// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::component::ComponentFrom;
use crate::layout::{Desc, Layout, fixed};
use crate::persist::{FnPersist, VectorMap};
use crate::{SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;

#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone, Default)]
pub struct Region<T: fixed::Prop + Default + 'static> {
    pub id: Rc<SourceID>,
    props: Rc<T>,
    children: im::Vector<Option<Box<ComponentFrom<dyn fixed::Prop>>>>,
}

impl<T: fixed::Prop + Default + 'static> Region<T> {
    pub fn new(
        id: Rc<SourceID>,
        props: Rc<T>,
        children: im::Vector<Option<Box<ComponentFrom<dyn fixed::Prop>>>>,
    ) -> Self {
        super::set_children(Self {
            id,
            props,
            children,
        })
    }
}

impl<T: fixed::Prop + Default + 'static> super::Component<T> for Region<T>
where
    for<'a> &'a T: Into<&'a (dyn fixed::Prop + 'static)>,
{
    fn layout(
        &self,
        state: &mut crate::StateManager,
        driver: &crate::graphics::Driver,
        window: &Rc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        let mut map = VectorMap::new(
            |child: &Option<Box<ComponentFrom<dyn fixed::Prop>>>| -> Option<Box<dyn Layout<<dyn fixed::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(state, driver, window))
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

crate::gen_component_wrap!(Region, fixed::Prop, Default);
