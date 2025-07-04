// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;
use crate::layout::{Layout, leaf};
use crate::{BASE_DPI, SourceID, WindowStateMachine, layout};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use ultraviolet::{Vec2, Vec4};

#[repr(u8)]
pub enum ShapeKind {
    RoundRect,
    Triangle,
    Circle,
    Arc,
}

pub struct Shape<T: leaf::Padded + 'static, const KIND: u8> {
    pub id: std::sync::Arc<SourceID>,
    pub props: Rc<T>,
    border: f32,
    blur: f32,
    pub corners: Vec4,
    pub fill: sRGB,
    pub outline: sRGB,
}

pub fn round_rect<T: leaf::Padded + 'static>(
    id: std::sync::Arc<SourceID>,
    props: Rc<T>,
    border: f32,
    blur: f32,
    corners: Vec4,
    fill: sRGB,
    outline: sRGB,
) -> Shape<T, { ShapeKind::RoundRect as u8 }> {
    Shape {
        id,
        props,
        border,
        blur,
        corners,
        fill,
        outline,
    }
}

pub fn triangle<T: leaf::Padded + 'static>(
    id: std::sync::Arc<SourceID>,
    props: Rc<T>,
    border: f32,
    blur: f32,
    corners: ultraviolet::Vec3,
    offset: f32,
    fill: sRGB,
    outline: sRGB,
) -> Shape<T, { ShapeKind::Triangle as u8 }> {
    Shape {
        id,
        props,
        border,
        blur,
        corners: Vec4::new(corners.x, corners.y, corners.z, offset),
        fill,
        outline,
    }
}

pub fn circle<T: leaf::Padded + 'static>(
    id: std::sync::Arc<SourceID>,
    props: Rc<T>,
    border: f32,
    blur: f32,
    radii: Vec2,
    fill: sRGB,
    outline: sRGB,
) -> Shape<T, { ShapeKind::Circle as u8 }> {
    Shape {
        id,
        props,
        border,
        blur,
        corners: Vec4::new(radii.x, radii.y, 0.0, 0.0),
        fill,
        outline,
    }
}

pub fn arcs<T: leaf::Padded + 'static>(
    id: std::sync::Arc<SourceID>,
    props: Rc<T>,
    border: f32,
    blur: f32,
    arcs: Vec4,
    fill: sRGB,
    outline: sRGB,
) -> Shape<T, { ShapeKind::Arc as u8 }> {
    Shape {
        id,
        props,
        border,
        blur,
        corners: arcs,
        fill,
        outline,
    }
}

impl<T: leaf::Padded + 'static, const KIND: u8> Clone for Shape<T, KIND> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.duplicate(),
            props: self.props.clone(),
            border: self.border,
            blur: self.blur,
            corners: self.corners,
            fill: self.fill,
            outline: self.outline,
        }
    }
}

impl<T: leaf::Padded + 'static> Shape<T, { ShapeKind::RoundRect as u8 }> {
    pub fn new(
        id: std::sync::Arc<SourceID>,
        props: Rc<T>,
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
            corners,
            fill,
            outline,
        }
    }
}

impl<T: leaf::Padded + 'static> Shape<T, { ShapeKind::Triangle as u8 }> {
    pub fn new(
        id: std::sync::Arc<SourceID>,
        props: Rc<T>,
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
            corners: Vec4::new(corners.x, corners.y, corners.z, offset),
            fill,
            outline,
        }
    }
}

impl<T: leaf::Padded + 'static> Shape<T, { ShapeKind::Circle as u8 }> {
    pub fn new(
        id: std::sync::Arc<SourceID>,
        props: Rc<T>,
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
            corners: Vec4::new(radii.x, radii.y, 0.0, 0.0),
            fill,
            outline,
        }
    }
}

impl<T: leaf::Padded + 'static> Shape<T, { ShapeKind::Arc as u8 }> {
    pub fn new(
        id: std::sync::Arc<SourceID>,
        props: Rc<T>,
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
            corners: arcs,
            fill,
            outline,
        }
    }
}

impl<T: leaf::Padded + 'static, const KIND: u8> crate::StateMachineChild for Shape<T, KIND> {
    fn id(&self) -> std::sync::Arc<SourceID> {
        self.id.clone()
    }
}

impl<T: leaf::Padded + 'static, const KIND: u8> super::Component for Shape<T, KIND>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    type Props = T;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        _: &crate::graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        let winstate: &WindowStateMachine = manager.get(window).unwrap();
        let dpi = winstate.state.as_ref().map(|x| x.dpi).unwrap_or(BASE_DPI);

        let mut corners = self.corners;
        if KIND == ShapeKind::RoundRect as u8 {
            corners *= Vec4::new(dpi.x, dpi.y, dpi.x, dpi.y);
        }

        Box::new(layout::Node::<T, dyn leaf::Prop> {
            props: self.props.clone(),
            children: Default::default(),
            id: Arc::downgrade(&self.id),
            renderable: Some(Rc::new(crate::render::shape::Instance::<
                crate::render::shape::Shape<KIND>,
            > {
                padding: self.props.padding().resolve(dpi),
                border: self.border,
                blur: self.blur,
                fill: self.fill,
                outline: self.outline,
                corners,
                id: self.id.clone(),
                phantom: PhantomData,
            })),
            layer: None,
        })
    }
}
