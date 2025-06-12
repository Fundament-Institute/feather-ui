// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Layout, base};
use crate::shaders::gen_uniform;
use crate::{CrossReferenceDomain, SourceID, layout, render, vec4_to_u32};
use derive_where::derive_where;
use std::rc::Rc;
use ultraviolet::Vec4;
use wgpu::util::DeviceExt;

// This draws a line between two points that were previously stored in a Cross-reference Domain
#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct DomainLine<T: base::Empty + 'static> {
    pub id: Rc<SourceID>,
    pub domain: Rc<CrossReferenceDomain>,
    pub start: Rc<SourceID>,
    pub end: Rc<SourceID>,
    pub props: Rc<T>,
    pub fill: Vec4,
}

impl<T: base::Empty + 'static> super::Component<T> for DomainLine<T>
where
    for<'a> &'a T: Into<&'a (dyn base::Empty + 'static)>,
{
    fn layout(
        &self,
        _: &crate::StateManager,
        _: &crate::graphics::State,
        _window: &Rc<SourceID>,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        Box::new(layout::Node::<T, dyn base::Empty> {
            props: self.props.clone(),
            children: Default::default(),
            id: Rc::downgrade(&self.id),
            renderable: Some(Rc::new(render::domain::line::Instance {
                domain: self.domain.clone(),
                start: self.start.clone(),
                end: self.end.clone(),
                color: vec4_to_u32(&self.fill),
            })),
        })
    }
}

crate::gen_component_wrap!(DomainLine, base::Empty);
