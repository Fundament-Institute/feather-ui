// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Desc, Layout, flex};
use crate::persist::{FnPersist, VectorMap};
use crate::{SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;

use super::ComponentFrom;

#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct FlexBox<T: flex::Prop + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub children: im::Vector<Option<Box<ComponentFrom<dyn flex::Prop>>>>,
}

impl<T: flex::Prop + 'static> super::Component<T> for FlexBox<T> {
    fn layout(
        &self,
        state: &crate::StateManager,
        graphics: &crate::graphics::State,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let map = VectorMap::new(
            |child: &Option<Box<ComponentFrom<dyn flex::Prop>>>| -> Option<Box<dyn Layout<<dyn flex::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(state, graphics,window, config))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<T, dyn flex::Prop> {
            props: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}

crate::gen_component_wrap!(FlexBox, flex::Prop);
