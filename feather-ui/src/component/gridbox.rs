// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Desc, Layout, grid};
use crate::persist::{FnPersist, VectorMap};
use crate::{SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

use super::ChildOf;

#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct GridBox<T: grid::Prop + 'static> {
    pub id: Arc<SourceID>,
    props: Rc<T>,
    children: im::Vector<Option<Box<ChildOf<dyn grid::Prop>>>>,
}

impl<T: grid::Prop + 'static> GridBox<T> {
    pub fn new(
        id: Arc<SourceID>,
        props: Rc<T>,
        children: im::Vector<Option<Box<ChildOf<dyn grid::Prop>>>>,
    ) -> Self {
        super::set_children(Self {
            id,
            props,
            children,
        })
    }
}

impl<T: grid::Prop + 'static> super::Component for GridBox<T> {
    type Props = T;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        driver: &crate::graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        let mut map = VectorMap::new(
            |child: &Option<Box<ChildOf<dyn grid::Prop>>>| -> Option<Box<dyn Layout<<dyn grid::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(manager, driver,window))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<T, dyn grid::Prop> {
            props: self.props.clone(),
            children,
            id: Arc::downgrade(&self.id),
            renderable: None,
            layer: None,
        })
    }
}
