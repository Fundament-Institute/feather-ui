// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::sRGB;
use crate::layout::base;
use crate::reactive::{self, DynSignal, MutableSignal};
use crate::{PxPoint, layout};
use std::rc::Rc;
use std::sync::Arc;

// This draws a line between two points relative to the parent
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
    type R = layout::Node<T, dyn base::Empty, crate::render::line::Instance>;

    fn layout(&self, _: &Arc<crate::graphics::Driver>, _: MutableSignal<crate::RelDim>) -> Self::R {
        layout::Node::<T, dyn base::Empty, crate::render::line::Instance> {
            props: self.props.clone(),
            children: reactive::empty_signal().into(),
            renderable: Some(crate::render::line::Instance::new(
                self.start.clone(),
                self.end.clone(),
                self.fill.clone(),
            )),
            machine: None,
        }
    }
}
