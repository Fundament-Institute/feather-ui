// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;
use crate::layout::{Layout, base};
use crate::{CrossReferenceDomain, SourceID, layout, render};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

// This draws a line between two points that were previously stored in a Cross-reference Domain
#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct DomainLine<T: base::Empty + 'static> {
    pub id: Arc<SourceID>,
    pub domain: Arc<CrossReferenceDomain>,
    pub start: Arc<SourceID>,
    pub end: Arc<SourceID>,
    pub props: Rc<T>,
    pub fill: sRGB,
}

impl<T: base::Empty + 'static> super::Component for DomainLine<T>
where
    for<'a> &'a T: Into<&'a (dyn base::Empty + 'static)>,
{
    type Props = T;

    fn layout(
        &self,
        _: &mut crate::StateManager,
        _: &crate::graphics::Driver,
        _: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        Box::new(layout::Node::<T, dyn base::Empty> {
            props: self.props.clone(),
            children: Default::default(),
            id: Arc::downgrade(&self.id),
            renderable: Some(Rc::new(render::domain::line::Instance {
                domain: self.domain.clone(),
                start: self.start.clone(),
                end: self.end.clone(),
                color: self.fill,
            })),
        })
    }
}
