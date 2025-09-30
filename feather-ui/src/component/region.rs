// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::sRGB32;
use crate::component::ChildOf;
use crate::layout::{Desc, Layout, fixed};
use crate::persist::{FnPersist, VectorMap};
use crate::{SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone, Default)]
pub struct Region<T: Default> {
    pub id: Arc<SourceID>,
    pub color: Option<sRGB32>,
    pub rotation: Option<f32>,
    props: Rc<T>,
    children: im::Vector<Box<ChildOf<dyn fixed::Prop>>>,
}

impl<T: fixed::Prop + Default + 'static> Region<T> {
    pub fn new(
        id: Arc<SourceID>,
        props: T,
        children: im::Vector<Box<ChildOf<dyn fixed::Prop>>>,
    ) -> Self {
        Self {
            id,
            props: props.into(),
            children,
            ..Default::default()
        }
    }

    pub fn new_layer(
        id: Arc<SourceID>,
        props: T,
        color: sRGB32,
        rotation: f32,
        children: im::Vector<Box<ChildOf<dyn fixed::Prop>>>,
    ) -> Self {
        Self {
            id,
            props: props.into(),
            color: Some(color),
            rotation: Some(rotation),
            children,
        }
    }
}

impl<T: fixed::Prop + Default + 'static> super::Component for Region<T>
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
        #[allow(clippy::borrowed_box)]
        let mut map = VectorMap::new(crate::persist::Persist::new(
            |child: &Box<ChildOf<dyn fixed::Prop>>| -> Box<dyn Layout<<dyn fixed::Prop as Desc>::Child>> {
                child.layout(manager, driver, window)
            }));

        let layer = if self.color.is_some() || self.rotation.is_some() {
            Some((
                self.color.unwrap_or(sRGB32::white()),
                self.rotation.unwrap_or_default(),
            ))
        } else {
            None
        };

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<T, dyn fixed::Prop> {
            props: self.props.clone(),
            children,
            id: Arc::downgrade(&self.id),
            renderable: None,
            layer,
        })
    }
}
