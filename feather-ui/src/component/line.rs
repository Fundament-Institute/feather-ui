// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::sRGB;
use crate::layout::{Layout, base};
use crate::reactive::{DynSignal, MutableSignal};
use crate::{PxPoint, SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

// This draws a line between two points relative to the parent
#[derive_where(Clone)]
pub struct Line<T> {
    pub start: DynSignal<PxPoint>,
    pub end: DynSignal<PxPoint>,
    pub props: Rc<T>,
    pub fill: DynSignal<sRGB>,
}

impl<T: base::Empty + 'static> super::Component for Line<T>
where
    for<'a> &'a T: Into<&'a (dyn base::Empty + 'static)>,
{
    type Props = T;
    type R = layout::Node<T, dyn base::Empty>;

    fn layout(&self, _: Arc<crate::graphics::Driver>, _: MutableSignal<crate::RelDim>) -> Self::R {
        use crate::reactive::AsSignal;
        layout::Node::<T, dyn base::Empty> {
            props: self.props.clone(),
            children: ().to_signal().into(),
            renderable: Some(Rc::new(crate::render::line::Instance::new(
                self.start.clone(),
                self.end.clone(),
                self.fill.clone(),
            ))),
            layer: None,
        }
    }
}
