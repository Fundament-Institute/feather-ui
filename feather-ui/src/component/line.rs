// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;
use crate::layout::{Layout, base};
use crate::{SourceID, layout};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;
use ultraviolet::Vec2;

// This draws a line between two points relative to the parent
#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct Line<T: base::Empty + 'static> {
    pub id: Arc<SourceID>,
    pub start: Vec2,
    pub end: Vec2,
    pub props: Rc<T>,
    pub fill: sRGB,
}

impl<T: base::Empty + 'static> super::Component for Line<T>
where
    for<'a> &'a T: Into<&'a (dyn base::Empty + 'static)>,
{
    type Props = T;

    fn layout(
        &self,
        _: &mut crate::StateManager,
        _: &crate::graphics::Driver,
        _window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        Box::new(layout::Node::<T, dyn base::Empty> {
            props: self.props.clone(),
            children: Default::default(),
            id: Arc::downgrade(&self.id),
            renderable: Some(Rc::new(crate::render::line::Instance {
                start: self.start,
                end: self.end,
                color: self.fill,
            })),
            layer: None,
        })
    }
}
