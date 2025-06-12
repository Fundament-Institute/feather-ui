// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Layout, leaf};
use crate::render::atlas;
use crate::{BASE_DPI, SourceID, WindowStateMachine, layout};
use derive_where::derive_where;
use std::borrow::Cow;
use std::rc::Rc;
use ultraviolet::Vec4;

#[derive_where(Clone)]
pub struct Shape<T: leaf::Padded + 'static> {
    pub id: std::rc::Rc<SourceID>,
    pub props: Rc<T>,
    border: f32,
    blur: f32,
    dim: ultraviolet::Vec2,
    pub corners: Vec4,
    pub fill: Vec4,
    pub outline: Vec4,
}

impl<T: leaf::Padded + 'static> Shape<T> {
    pub fn round_rect(
        id: std::rc::Rc<SourceID>,
        props: Rc<T>,
        dim: ultraviolet::Vec2,
        border: f32,
        blur: f32,
        corners: Vec4,
        fill: Vec4,
        outline: Vec4,
    ) -> Self {
        Self {
            id,
            props,
            uniforms: [Vec4::new(0.0, 0.0, border, blur), corners, fill, outline],
            fragment: Cow::Borrowed(include_str!("../shaders/RoundRect.wgsl")),
            label: "RoundRect FS",
        }
    }

    pub fn arc(
        id: std::rc::Rc<SourceID>,
        props: Rc<T>,
        border: f32,
        blur: f32,
        arcs: Vec4,
        fill: Vec4,
        outline: Vec4,
    ) -> Self {
        Self {
            id,
            props,
            uniforms: [Vec4::new(0.0, 0.0, border, blur), arcs, fill, outline],
            fragment: Cow::Borrowed(include_str!("../shaders/Arc.wgsl")),
            label: "Arc FS",
        }
    }

    pub fn circle(
        id: std::rc::Rc<SourceID>,
        props: Rc<T>,
        border: f32,
        blur: f32,
        radii: crate::Vec2,
        fill: Vec4,
        outline: Vec4,
    ) -> Self {
        Self {
            id,
            props,
            uniforms: [
                Vec4::new(0.0, 0.0, border, blur),
                Vec4::new(radii.x, radii.y, 0.0, 0.0),
                fill,
                outline,
            ],
            fragment: Cow::Borrowed(include_str!("../shaders/Circle.wgsl")),
            label: "Circle FS",
        }
    }
}

impl<T: leaf::Padded + 'static> crate::StateMachineChild for Shape<T> {
    fn id(&self) -> std::rc::Rc<SourceID> {
        self.id.clone()
    }
}

impl<T: leaf::Padded + 'static> super::Component<T> for Shape<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    fn layout(
        &self,
        state: &crate::StateManager,
        graphics: &crate::graphics::State,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let winstate: &WindowStateMachine = state.get(window).unwrap();
        let dpi = winstate.state.as_ref().map(|x| x.dpi).unwrap_or(BASE_DPI);

        Box::new(layout::Node::<T, dyn leaf::Prop> {
            props: self.props.clone(),
            children: Default::default(),
            id: Rc::downgrade(&self.id),
            renderable: Some(Rc::new(crate::render::shape::Instance {
                padding: self.props.padding().resolve(dpi),
                border: self.border,
                blur: self.blur,
                fill: self.fill,
                outline: self.outline,
                corners: self.corners,
                id: self.id.clone(),
            })),
        })
    }
}

crate::gen_component_wrap!(Shape, leaf::Padded);
