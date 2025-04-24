// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

pub mod base;
pub mod domain_write;
pub mod fixed;
pub mod flex;
pub mod leaf;
pub mod list;
pub mod root;

use dyn_clone::DynClone;
use ultraviolet::Vec2;

use crate::outline::Renderable;
use crate::persist::FnPersist2;
use crate::persist::VectorFold;
use crate::rtree;
use crate::AbsDim;
use crate::AbsRect;
use crate::DriverState;
use crate::RelRect;
use crate::RenderInstruction;
use crate::SourceID;
use derive_where::derive_where;
use std::rc::{Rc, Weak};

pub trait Layout<Props>: DynClone {
    fn get_props(&self) -> &Props;
    fn inner_stage<'a>(&self, area: AbsRect, driver: &DriverState) -> Box<dyn Staged + 'a>;
}

dyn_clone::clone_trait_object!(<Imposed> Layout<Imposed> where Imposed:Sized);

pub trait LayoutWrap<Imposed: ?Sized>: DynClone {
    fn get_imposed(&self) -> &Imposed;
    fn stage<'a>(&self, area: AbsRect, driver: &DriverState) -> Box<dyn Staged + 'a>;
}

dyn_clone::clone_trait_object!(<Imposed> LayoutWrap<Imposed> where Imposed:?Sized);

impl<U: ?Sized, T> LayoutWrap<U> for Box<dyn Layout<T>>
where
    for<'a> &'a T: Into<&'a U>,
{
    fn get_imposed(&self) -> &U {
        self.get_props().into()
    }

    fn stage<'a>(&self, area: AbsRect, driver: &DriverState) -> Box<dyn Staged + 'a> {
        self.inner_stage(area, driver)
    }
}

impl<U: ?Sized, T> LayoutWrap<U> for &dyn Layout<T>
where
    for<'a> &'a T: Into<&'a U>,
{
    fn get_imposed(&self) -> &U {
        self.get_props().into()
    }

    fn stage<'a>(&self, area: AbsRect, driver: &DriverState) -> Box<dyn Staged + 'a> {
        self.inner_stage(area, driver)
    }
}

impl<T: 'static> From<Box<dyn Layout<T>>> for Box<dyn LayoutWrap<T>> {
    fn from(value: Box<dyn Layout<T>>) -> Self {
        Box::new(value)
    }
}

pub trait Desc {
    type Props: ?Sized;
    type Child: ?Sized;
    type Children: Clone;

    /// Resolves a pending layout into a resolved node, which contains a pointer to the R-tree
    fn stage<'a>(
        props: &Self::Props,
        outer_area: AbsRect,
        children: &Self::Children,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a>;
}

#[derive_where(Clone)]
pub struct Node<T, D: Desc + ?Sized> {
    pub props: Rc<T>,
    pub id: std::rc::Weak<SourceID>,
    pub children: D::Children,
    pub renderable: Option<Rc<dyn Renderable>>,
}

impl<T, D: Desc + ?Sized> Layout<T> for Node<T, D>
where
    for<'a> &'a T: Into<&'a D::Props>,
{
    fn get_props(&self) -> &T {
        self.props.as_ref()
    }
    fn inner_stage<'a>(&self, area: AbsRect, driver: &DriverState) -> Box<dyn Staged + 'a> {
        D::stage(
            self.props.as_ref().into(),
            area,
            &self.children,
            self.id.clone(),
            self.renderable.as_ref().map(|x| x.clone()),
            driver,
        )
    }
}

pub trait Staged: DynClone {
    fn render(&self, parent_pos: Vec2, driver: &DriverState) -> im::Vector<RenderInstruction>;
    fn get_rtree(&self) -> Weak<rtree::Node>;
    fn get_area(&self) -> AbsRect;
}

dyn_clone::clone_trait_object!(Staged);

#[derive(Clone)]
pub(crate) struct Concrete {
    pub render: Option<Rc<dyn Renderable>>,
    pub area: AbsRect,
    pub rtree: Rc<rtree::Node>,
    pub children: im::Vector<Option<Box<dyn Staged>>>,
}

impl Staged for Concrete {
    fn render(&self, parent_pos: Vec2, driver: &DriverState) -> im::Vector<RenderInstruction> {
        let instructions = self
            .render
            .as_ref()
            .map(|x| x.render(self.area + parent_pos, driver))
            .unwrap_or_default();

        let fold = VectorFold::new(
            |list: &im::Vector<RenderInstruction>,
             n: &Option<Box<dyn Staged>>|
             -> im::Vector<RenderInstruction> {
                let mut a = n
                    .as_ref()
                    .unwrap()
                    .render(parent_pos + self.area.topleft, driver);
                a.append(list.clone());
                a
            },
        );

        let (_, result) = fold.call(fold.init(), &instructions, &self.children);
        result
    }

    fn get_rtree(&self) -> Weak<rtree::Node> {
        Rc::downgrade(&self.rtree)
    }

    fn get_area(&self) -> AbsRect {
        self.area
    }
}

#[inline]
pub(crate) fn zero_infinity(mut v: AbsDim) -> AbsDim {
    if v.0.x.is_infinite() {
        v.0.x = 0.0
    }
    if v.0.y.is_infinite() {
        v.0.y = 0.0
    }
    v
}

#[inline]
pub(crate) fn nuetralize_infinity(mut v: AbsRect) -> AbsRect {
    if v.bottomright.x.is_infinite() {
        v.bottomright.x = v.topleft.x
    }
    if v.bottomright.y.is_infinite() {
        v.bottomright.y = v.topleft.y
    }
    v
}

#[inline]
pub(crate) fn limit_area(mut v: AbsRect, limits: AbsRect) -> AbsRect {
    v.bottomright = v.topleft.max_by_component(v.topleft + limits.topleft);
    v.bottomright = v.topleft.min_by_component(v.topleft + limits.bottomright);
    v
}

#[inline]
pub(crate) fn merge_limits(l: AbsRect, r: AbsRect) -> AbsRect {
    AbsRect {
        topleft: l.topleft.max_by_component(r.topleft),
        bottomright: l.bottomright.min_by_component(r.bottomright),
    }
}

#[inline]
pub(crate) fn limit_dim(v: AbsDim, limits: AbsRect) -> AbsDim {
    AbsDim(Vec2 {
        x: if v.0.x.is_finite() {
            v.0.x.max(limits.topleft.x).min(limits.bottomright.x)
        } else {
            v.0.x
        },
        y: if v.0.y.is_finite() {
            v.0.y.max(limits.topleft.y).min(limits.bottomright.y)
        } else {
            v.0.y
        },
    })
}

#[inline]
pub(crate) fn eval_dim(area: crate::URect, dim: AbsDim) -> AbsDim {
    AbsDim(Vec2 {
        x: if area.bottomright.abs.x.is_finite() {
            let top = area.topleft.abs.x + (area.topleft.rel.0.x * dim.0.x);
            let bottom = area.bottomright.abs.x + (area.bottomright.rel.0.x * dim.0.x);
            bottom - top
        } else {
            area.bottomright.abs.x
        },
        y: if area.bottomright.abs.y.is_finite() {
            let top = area.topleft.abs.y + (area.topleft.rel.0.y * dim.0.y);
            let bottom = area.bottomright.abs.y + (area.bottomright.rel.0.y * dim.0.y);
            bottom - top
        } else {
            area.bottomright.abs.y
        },
    })
}

#[inline]
pub(crate) fn eval_limits(limits: RelRect, dim: AbsDim) -> AbsRect {
    AbsRect {
        topleft: Vec2 {
            x: if dim.0.x.is_finite() {
                limits.topleft.0.x * dim.0.x
            } else {
                limits.topleft.0.x
            },
            y: if dim.0.y.is_finite() {
                limits.topleft.0.y * dim.0.y
            } else {
                limits.topleft.0.y
            },
        },
        bottomright: Vec2 {
            x: if dim.0.x.is_finite() {
                limits.bottomright.0.x * dim.0.x
            } else {
                limits.bottomright.0.x
            },
            y: if dim.0.y.is_finite() {
                limits.bottomright.0.y * dim.0.y
            } else {
                limits.bottomright.0.y
            },
        },
    }
}
