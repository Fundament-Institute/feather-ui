// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::sRGB32;
use crate::component::ChildOf;
use crate::layout;
use crate::layout::fixed;
use crate::reactive::{DynSignal, MutableSignal, map_vec};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

#[derive_where(Clone)]
pub struct Region<T: Default> {
    pub color: Option<sRGB32>,
    pub rotation: Option<f32>,
    props: Rc<T>,
    children: DynSignal<imbl::Vector<Rc<ChildOf<dyn fixed::Prop>>>>,
}

impl<T: fixed::Prop + Default + 'static> Region<T> {
    pub fn new(props: T, children: DynSignal<imbl::Vector<Rc<ChildOf<dyn fixed::Prop>>>>) -> Self {
        Self {
            props: props.into(),
            children,
            color: None,
            rotation: None,
        }
    }

    pub fn new_layer(
        props: T,
        color: sRGB32,
        rotation: f32,
        children: DynSignal<imbl::Vector<Rc<ChildOf<dyn fixed::Prop>>>>,
    ) -> Self {
        Self {
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
    type R = layout::Node<T, dyn fixed::Prop>;

    fn layout(
        &self,
        driver: Arc<crate::graphics::Driver>,
        dpi: MutableSignal<crate::RelDim>,
    ) -> Self::R {
        let children = map_vec(
            move |child: &Rc<ChildOf<dyn fixed::Prop>>| child.layout(driver.clone(), dpi.clone()),
            |x| crate::reactive::Identity(x.clone()),
            &self.children,
        );

        let layer = if self.color.is_some() || self.rotation.is_some() {
            Some((
                self.color.unwrap_or(sRGB32::white()),
                self.rotation.unwrap_or_default(),
            ))
        } else {
            None
        };

        Self::R {
            props: self.props.clone(),
            children: children.into(),
            renderable: None,
            layer,
        }
    }
}
