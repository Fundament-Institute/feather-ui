// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Desc, Layout, flex};
use crate::persist::{FnPersist, VectorMap};
use crate::{SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

use super::ChildOf;

#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct FlexBox<T: flex::Prop + 'static> {
    pub id: Arc<SourceID>,
    props: Rc<T>,
    children: im::Vector<Option<Box<ChildOf<dyn flex::Prop>>>>,
}

impl<T: flex::Prop + 'static> FlexBox<T> {
    pub fn new(
        id: Arc<SourceID>,
        props: Rc<T>,
        children: im::Vector<Option<Box<ChildOf<dyn flex::Prop>>>>,
    ) -> Self {
        super::set_children(Self {
            id,
            props,
            children,
        })
    }
}

impl<T: flex::Prop + 'static> super::Component for FlexBox<T> {
    type Props = T;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        driver: &crate::graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        let mut map = VectorMap::new(
            |child: &Option<Box<ChildOf<dyn flex::Prop>>>| -> Option<Box<dyn Layout<<dyn flex::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(manager, driver,window))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<T, dyn flex::Prop> {
            props: self.props.clone(),
            children,
            id: Arc::downgrade(&self.id),
            renderable: None,
            layer: None,
        })
    }
}
