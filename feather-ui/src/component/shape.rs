// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::sRGB;
use crate::layout::leaf;
use crate::reactive::{DynSignal, MutableSignal, SignalMap, zip_pair};
use std::rc::Rc;
use std::sync::Arc;

#[repr(u8)]
pub enum ShapeKind {
    RoundRect,
    Triangle,
    Circle,
    Arc,
}

pub struct Shape<T, const KIND: u8> {
    props: Rc<T>,
    border: DynSignal<f32>,
    blur: DynSignal<f32>,
    corners: DynSignal<[f32; 4]>,
    fill: DynSignal<sRGB>,
    outline: DynSignal<sRGB>,
    size: DynSignal<crate::DAbsPoint>,
}

pub fn round_rect<T: leaf::Padded + 'static>(
    props: T,
    border: DynSignal<f32>,
    blur: DynSignal<f32>,
    corners: DynSignal<wide::f32x4>,
    fill: DynSignal<sRGB>,
    outline: DynSignal<sRGB>,
    size: DynSignal<crate::DAbsPoint>,
) -> Shape<T, { ShapeKind::RoundRect as u8 }> {
    Shape {
        props: props.into(),
        border,
        blur,
        corners: corners.map(|x| x.to_array()).into(),
        fill,
        outline,
        size,
    }
}

pub fn triangle<T: leaf::Padded + 'static>(
    props: T,
    border: DynSignal<f32>,
    blur: DynSignal<f32>,
    corners: DynSignal<[f32; 3]>,
    offset: DynSignal<f32>,
    fill: DynSignal<sRGB>,
    outline: DynSignal<sRGB>,
    size: DynSignal<crate::DAbsPoint>,
) -> Shape<T, { ShapeKind::Triangle as u8 }> {
    Shape {
        props: props.into(),
        border,
        blur,
        corners: zip_pair(corners, offset, |c, o| [c[0], c[1], c[2], *o]).into(),
        fill,
        outline,
        size,
    }
}

pub fn circle<T: leaf::Padded + 'static>(
    props: T,
    border: DynSignal<f32>,
    blur: DynSignal<f32>,
    radii: DynSignal<[f32; 2]>,
    fill: DynSignal<sRGB>,
    outline: DynSignal<sRGB>,
    size: DynSignal<crate::DAbsPoint>,
) -> Shape<T, { ShapeKind::Circle as u8 }> {
    Shape {
        props: props.into(),
        border,
        blur,
        corners: radii.map(|r| [r[0], r[1], 0.0, 0.0]).into(),
        fill,
        outline,
        size,
    }
}

pub fn arcs<T: leaf::Padded + 'static>(
    props: T,
    border: DynSignal<f32>,
    blur: DynSignal<f32>,
    inner_radius: DynSignal<f32>,
    arcs: DynSignal<[f32; 2]>,
    fill: DynSignal<sRGB>,
    outline: DynSignal<sRGB>,
    size: DynSignal<crate::DAbsPoint>,
) -> Shape<T, { ShapeKind::Arc as u8 }> {
    Shape {
        props: props.into(),
        border,
        blur,
        corners: zip_pair(arcs, inner_radius, |arcs, r| {
            [arcs[0] + arcs[1] * 0.5, arcs[1] * 0.5, *r, 0.0]
        })
        .into(),
        fill,
        outline,
        size,
    }
}

impl<T: leaf::Padded + 'static, const KIND: u8> super::Component for Shape<T, KIND>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    type Props = T;
    type R = leaf::Sized<T, crate::render::shape::PreInstance<KIND>>;

    fn layout(
        &self,
        driver: Arc<crate::graphics::Driver>,
        dpi: MutableSignal<crate::RelDim>,
    ) -> Self::R {
        let corners = if KIND == ShapeKind::RoundRect as u8 {
            zip_pair(self.corners.clone(), dpi.clone(), |c, dpi| {
                [
                    c[0] * dpi.width,
                    c[1] * dpi.height,
                    c[2] * dpi.width,
                    c[3] * dpi.height,
                ]
            })
            .into_dyn_signal()
        } else {
            self.corners.clone()
        };

        leaf::Sized::<T, _> {
            props: self.props.clone(),
            size: zip_pair(self.size.clone(), dpi.clone(), |s, dpi| {
                s.resolve(*dpi).to_vector().to_size().cast_unit()
            })
            .into_dyn_signal(),
            renderable: Some(crate::render::shape::PreInstance::<KIND> {
                padding: zip_pair(self.props.padding(), dpi, |x, dpi| x.as_perimeter(*dpi)).into(),
                border: self.border.clone(),
                blur: self.blur.clone(),
                fill: self.fill.clone(),
                outline: self.outline.clone(),
                corners,
                driver,
            }),
            machine: None,
        }
        .into()
    }
}
