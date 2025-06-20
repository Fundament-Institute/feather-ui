// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;
use crate::layout::{Layout, leaf};
use crate::{BASE_DPI, SourceID, WindowStateMachine, layout};
use derive_where::derive_where;
use std::marker::PhantomData;
use std::rc::Rc;
use ultraviolet::{Vec2, Vec4};

#[repr(u8)]
pub enum ShapeKind {
    RoundRect,
    Triangle,
    Circle,
    Arc,
}

#[derive_where(Clone)]
pub struct Shape<T: leaf::Padded + 'static, const KIND: u8> {
    pub id: std::rc::Rc<SourceID>,
    pub props: Rc<T>,
    border: f32,
    blur: f32,
    dim: Vec2,
    pub corners: Vec4,
    pub fill: sRGB,
    pub outline: sRGB,
}
impl<T: leaf::Padded + 'static> Shape<T, { ShapeKind::RoundRect as u8 }> {
    pub fn new(
        id: std::rc::Rc<SourceID>,
        props: Rc<T>,
        dim: Vec2,
        border: f32,
        blur: f32,
        corners: Vec4,
        fill: sRGB,
        outline: sRGB,
    ) -> Self {
        Self {
            id,
            props,
            border,
            blur,
            dim,
            corners,
            fill,
            outline,
        }
    }
}

impl<T: leaf::Padded + 'static> Shape<T, 1> {
    pub fn new(
        id: std::rc::Rc<SourceID>,
        props: Rc<T>,
        dim: Vec2,
        border: f32,
        blur: f32,
        corners: ultraviolet::Vec3,
        offset: f32,
        fill: sRGB,
        outline: sRGB,
    ) -> Self {
        Self {
            id,
            props,
            border,
            blur,
            dim,
            corners: Vec4::new(corners.x, corners.y, corners.z, offset),
            fill,
            outline,
        }
    }
}

impl<T: leaf::Padded + 'static> Shape<T, 2> {
    pub fn new(
        id: std::rc::Rc<SourceID>,
        props: Rc<T>,
        dim: Vec2,
        border: f32,
        blur: f32,
        radii: Vec2,
        fill: sRGB,
        outline: sRGB,
    ) -> Self {
        Self {
            id,
            props,
            border,
            blur,
            dim,
            corners: Vec4::new(radii.x, radii.y, 0.0, 0.0),
            fill,
            outline,
        }
    }
}

impl<T: leaf::Padded + 'static> Shape<T, 3> {
    pub fn new(
        id: std::rc::Rc<SourceID>,
        props: Rc<T>,
        dim: Vec2,
        border: f32,
        blur: f32,
        arcs: Vec4,
        fill: sRGB,
        outline: sRGB,
    ) -> Self {
        Self {
            id,
            props,
            border,
            blur,
            dim,
            corners: arcs,
            fill,
            outline,
        }
    }
}

impl<T: leaf::Padded + 'static, const KIND: u8> crate::StateMachineChild for Shape<T, KIND> {
    fn id(&self) -> std::rc::Rc<SourceID> {
        self.id.clone()
    }
}

impl<T: leaf::Padded + 'static, const KIND: u8> super::Component<T> for Shape<T, KIND>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    fn layout(
        &self,
        state: &crate::StateManager,
        _: &crate::graphics::Driver,
        window: &Rc<SourceID>,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let winstate: &WindowStateMachine = state.get(window).unwrap();
        let dpi = winstate.state.as_ref().map(|x| x.dpi).unwrap_or(BASE_DPI);

        Box::new(layout::Node::<T, dyn leaf::Prop> {
            props: self.props.clone(),
            children: Default::default(),
            id: Rc::downgrade(&self.id),
            renderable: Some(Rc::new(crate::render::shape::Instance::<
                crate::render::shape::Shape<KIND>,
            > {
                padding: self.props.padding().resolve(dpi),
                border: self.border,
                blur: self.blur,
                fill: self.fill,
                outline: self.outline,
                corners: self.corners,
                id: self.id.clone(),
                phantom: PhantomData,
            })),
        })
    }
}

impl<U: ?Sized, T: leaf::Padded + 'static, const KIND: u8> crate::component::ComponentWrap<U>
    for Shape<T, KIND>
where
    Shape<T, KIND>: crate::component::Component<T>,
    for<'a> &'a T: Into<&'a U>,
{
    crate::gen_component_wrap_inner!();
}
