// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::sRGB32;
use crate::component::ChildOf;
use crate::layout;
use crate::layout::fixed;
use crate::reactive::{DynSignal, MutableSignal, SignalTupleZip, map_vec, zip_pair};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

#[derive_where(Clone)]
pub struct Region<T: Default> {
    pub layer: Option<DynSignal<(sRGB32, f32)>>,
    props: Rc<T>,
    children: DynSignal<imbl::Vector<Arc<ChildOf<dyn fixed::Prop>>>>,
}

impl<T: fixed::Prop + Default + 'static> Region<T> {
    pub fn new(props: T, children: DynSignal<imbl::Vector<Arc<ChildOf<dyn fixed::Prop>>>>) -> Self {
        Self {
            props: props.into(),
            children,
            layer: None,
        }
    }

    pub fn new_layer(
        props: T,
        color: DynSignal<sRGB32>,
        rotation: DynSignal<f32>,
        children: DynSignal<imbl::Vector<Arc<ChildOf<dyn fixed::Prop>>>>,
    ) -> Self {
        Self {
            props: props.into(),
            layer: Some((color, rotation).zip().into_dyn_signal()),
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
            move |child: &Arc<ChildOf<dyn fixed::Prop>>| child.layout(driver.clone(), dpi.clone()),
            |x| crate::reactive::Identity(x.clone()),
            self.children.clone(),
        );

        Self::R {
            props: self.props.clone(),
            children: children.into(),
            renderable: None,
            layer: self.layer.clone(),
        }
    }
}
