// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::layout::{Desc, Layout, flex};
use crate::persist::{FnPersist, VectorMap};
use crate::{SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

use super::ChildOf;

pub struct FlexBox<T> {
    pub id: Arc<SourceID>,
    props: Rc<T>,
    children: imbl::Vector<Rc<ChildOf<dyn flex::Prop>>>,
}

impl<T: flex::Prop + 'static> FlexBox<T> {
    pub fn new(
        id: Arc<SourceID>,
        props: T,
        children: imbl::Vector<Rc<ChildOf<dyn flex::Prop>>>,
    ) -> Self {
        Self {
            id,
            props: props.into(),
            children,
        }
    }
}

impl<T: flex::Prop + 'static> super::Component for FlexBox<T> {
    type Props = T;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        driver: &crate::graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Rc<dyn Layout<T>> {
        #[allow(clippy::borrowed_box)]
        let mut map = VectorMap::new(crate::persist::Persist::new(
            |child: &Box<ChildOf<dyn flex::Prop>>| -> Rc<dyn Layout<<dyn flex::Prop as Desc>::Child>> {
                child.layout(manager, driver,window)
            })
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
