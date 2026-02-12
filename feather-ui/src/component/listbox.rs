// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::component::ChildOf;
use crate::layout;
use crate::layout::{Layout, list};
use crate::reactive::{MutableSignal, map_vec};
use std::rc::Rc;
use std::sync::Arc;

// Doesn't need to be clonable because we store it as an Rc<dyn Component>
pub struct ListBox<T: list::Prop + 'static> {
    props: Rc<T>,
    children: MutableSignal<imbl::Vector<Rc<ChildOf<dyn list::Prop>>>>,
}

impl<T: list::Prop + 'static> ListBox<T> {
    pub fn new(
        props: T,
        children: MutableSignal<imbl::Vector<Rc<ChildOf<dyn list::Prop>>>>,
    ) -> Self {
        let props = Rc::new(props);
        Self {
            props: props.clone(),
            children,
        }
    }
}

impl<T: list::Prop + 'static> super::Component for ListBox<T> {
    type Props = T;

    fn layout(
        &self,
        driver: &Arc<crate::graphics::Driver>,
        dpi: MutableSignal<crate::RelDim>,
    ) -> Rc<dyn Layout<T>> {
        let children = map_vec(
            move |child: &Rc<ChildOf<dyn list::Prop>>| child.layout(driver.clone(), dpi.clone()),
            |x| crate::reactive::Identity(x.clone()),
            &self.children,
        );

        Box::new(layout::Node::<T, dyn list::Prop> {
            props: self.props.clone(),
            children: children.into(),
            renderable: None,
            layer: None,
        })
    }
}
