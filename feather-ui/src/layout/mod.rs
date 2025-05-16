// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

pub mod base;
pub mod domain_write;
pub mod fixed;
pub mod flex;
pub mod grid;
pub mod leaf;
pub mod list;
pub mod root;

use dyn_clone::DynClone;
use ultraviolet::Vec2;
use wide::f32x4;

use crate::outline::Renderable;
use crate::persist::{FnPersist2, VectorFold};
use crate::{
    rtree, AbsDim, AbsLimits, AbsRect, DriverState, RelLimits, RenderInstruction, SourceID, URect,
    UNSIZED_AXIS,
};
use derive_where::derive_where;
use std::rc::{Rc, Weak};

pub trait Layout<Props>: DynClone {
    fn get_props(&self) -> &Props;
    fn inner_stage<'a>(
        &self,
        area: AbsRect,
        limits: AbsLimits,
        dpi: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a>;
}

dyn_clone::clone_trait_object!(<Imposed> Layout<Imposed> where Imposed:Sized);

pub trait LayoutWrap<Imposed: ?Sized>: DynClone {
    fn get_imposed(&self) -> &Imposed;
    fn stage<'a>(
        &self,
        area: AbsRect,
        limits: AbsLimits,
        dpi: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a>;
}

dyn_clone::clone_trait_object!(<Imposed> LayoutWrap<Imposed> where Imposed:?Sized);

impl<U: ?Sized, T> LayoutWrap<U> for Box<dyn Layout<T>>
where
    for<'a> &'a T: Into<&'a U>,
{
    fn get_imposed(&self) -> &U {
        self.get_props().into()
    }

    fn stage<'a>(
        &self,
        area: AbsRect,
        limits: AbsLimits,
        dpi: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a> {
        self.inner_stage(area, limits, dpi, driver)
    }
}

impl<U: ?Sized, T> LayoutWrap<U> for &dyn Layout<T>
where
    for<'a> &'a T: Into<&'a U>,
{
    fn get_imposed(&self) -> &U {
        self.get_props().into()
    }

    fn stage<'a>(
        &self,
        area: AbsRect,
        limits: AbsLimits,
        dpi: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a> {
        self.inner_stage(area, limits, dpi, driver)
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
        limits: AbsLimits,
        children: &Self::Children,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        dpi: Vec2,
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
    fn inner_stage<'a>(
        &self,
        area: AbsRect,
        limits: AbsLimits,
        dpi: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a> {
        D::stage(
            self.props.as_ref().into(),
            area,
            limits,
            &self.children,
            self.id.clone(),
            self.renderable.as_ref().map(|x| x.clone()),
            dpi,
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
    render: Option<Rc<dyn Renderable>>,
    area: AbsRect,
    rtree: Rc<rtree::Node>,
    children: im::Vector<Option<Box<dyn Staged>>>,
}

impl Concrete {
    pub fn new(
        render: Option<Rc<dyn Renderable>>,
        area: AbsRect,
        rtree: Rc<rtree::Node>,
        children: im::Vector<Option<Box<dyn Staged>>>,
    ) -> Self {
        let (unsized_x, unsized_y) = check_unsized_abs(area.bottomright());
        assert!(
            !unsized_x && !unsized_y,
            "concrete area must always be sized!"
        );
        Self {
            render,
            area,
            rtree,
            children,
        }
    }
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
                    .render(parent_pos + self.area.topleft(), driver);
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

#[must_use]
#[inline]
pub(crate) fn zero_unsized(mut v: AbsDim) -> AbsDim {
    let (unsized_x, unsized_y) = check_unsized_dim(v);
    if unsized_x {
        v.0.x = 0.0
    }
    if unsized_y {
        v.0.y = 0.0
    }
    v
}

#[must_use]
#[inline]
pub(crate) fn map_unsized_area(mut area: URect, adjust: Vec2) -> URect {
    let (unsized_x, unsized_y) = check_unsized(area);
    let abs = area.abs.0.as_array_mut();
    let rel = area.rel.0.as_array_mut();
    // Unsized objects must always have a single anchor point to make sense, so we copy over from topleft.
    if unsized_x {
        rel[2] = rel[0];
        // Fix the bottomright abs area in unsized scenarios, because it is now relative to the topleft instead of being independent.
        abs[2] += abs[0] + adjust.x;
    }
    if unsized_y {
        rel[3] = rel[1];
        abs[3] += abs[1] + adjust.y;
    }
    area
}

#[must_use]
#[inline]
pub(crate) fn nuetralize_unsized(v: AbsRect) -> AbsRect {
    let (unsized_x, unsized_y) = check_unsized_abs(v.bottomright());
    let ltrb = v.0.to_array();
    AbsRect(f32x4::new([
        ltrb[0],
        ltrb[1],
        if unsized_x { ltrb[0] } else { ltrb[2] },
        if unsized_y { ltrb[1] } else { ltrb[3] },
    ]))
}

#[must_use]
#[inline]
pub(crate) fn limit_area(mut v: AbsRect, limits: AbsLimits) -> AbsRect {
    // We do this by checking clamp(topleft + limit) instead of clamp(bottomright - topleft)
    // because this avoids floating point precision issues.
    v.set_bottomright(
        v.bottomright()
            .max_by_component(v.topleft() + limits.min())
            .min_by_component(v.topleft() + limits.max()),
    );
    v
}

#[must_use]
#[inline]
pub(crate) fn limit_dim(v: AbsDim, limits: AbsLimits) -> AbsDim {
    let (unsized_x, unsized_y) = check_unsized_dim(v);
    AbsDim(Vec2 {
        x: if unsized_x {
            v.0.x
        } else {
            v.0.x.max(limits.min().x).min(limits.max().x)
        },
        y: if unsized_y {
            v.0.y
        } else {
            v.0.y.max(limits.min().y).min(limits.max().y)
        },
    })
}

#[must_use]
#[inline]
pub(crate) fn eval_dim(area: URect, dim: AbsDim) -> AbsDim {
    let (unsized_x, unsized_y) = check_unsized(area);
    AbsDim(Vec2 {
        x: if unsized_x {
            area.bottomright().rel().0.x
        } else {
            let top = area.topleft().abs().x + (area.topleft().rel().0.x * dim.0.x);
            let bottom = area.bottomright().abs().x + (area.bottomright().rel().0.x * dim.0.x);
            bottom - top
        },
        y: if unsized_y {
            area.bottomright().rel().0.y
        } else {
            let top = area.topleft().abs().y + (area.topleft().rel().0.y * dim.0.y);
            let bottom = area.bottomright().abs().y + (area.bottomright().rel().0.y * dim.0.y);
            bottom - top
        },
    })
}

#[must_use]
#[inline]
pub(crate) fn apply_limit(dim: AbsDim, limits: AbsLimits, rlimits: RelLimits) -> AbsLimits {
    let (unsized_x, unsized_y) = check_unsized_dim(dim);
    AbsLimits(f32x4::new([
        if unsized_x {
            limits.min().x
        } else {
            limits.min().x.max(rlimits.min().0.x * dim.0.x)
        },
        if unsized_y {
            limits.min().y
        } else {
            limits.min().y.max(rlimits.min().0.y * dim.0.y)
        },
        if unsized_x {
            limits.max().x
        } else {
            limits.max().x.min(rlimits.max().0.x * dim.0.x)
        },
        if unsized_y {
            limits.max().y
        } else {
            limits.max().y.min(rlimits.max().0.y * dim.0.y)
        },
    ]))
}

// Returns true if an axis is unsized, which means it is defined as the size of it's children's maximum extent.
#[must_use]
#[inline]
pub(crate) fn check_unsized(area: URect) -> (bool, bool) {
    (
        area.bottomright().rel().0.x == UNSIZED_AXIS,
        area.bottomright().rel().0.y == UNSIZED_AXIS,
    )
}

// Returns true if an axis is unsized, which means it is defined as the size of it's children's maximum extent.
#[must_use]
#[inline]
pub(crate) fn check_unsized_abs(bottomright: Vec2) -> (bool, bool) {
    (bottomright.x == UNSIZED_AXIS, bottomright.y == UNSIZED_AXIS)
}

// Returns true if an axis is unsized, which means it is defined as the size of it's children's maximum extent.
#[must_use]
#[inline]
pub(crate) fn check_unsized_dim(dim: crate::AbsDim) -> (bool, bool) {
    check_unsized_abs(dim.0)
}

#[must_use]
#[inline]
pub(crate) fn cap_unsized(area: crate::AbsRect) -> crate::AbsRect {
    let ltrb = area.0.to_array();
    crate::AbsRect(f32x4::new(ltrb.map(|x| {
        if x.is_finite() {
            x
        } else {
            crate::UNSIZED_AXIS
        }
    })))
}

#[must_use]
#[inline]
pub(crate) fn apply_anchor(area: AbsRect, outer_area: AbsRect, mut anchor: Vec2) -> AbsRect {
    let (unsized_outer_x, unsized_outer_y) = check_unsized_abs(outer_area.bottomright());
    if unsized_outer_x {
        anchor.x = 0.0;
    }
    if unsized_outer_y {
        anchor.y = 0.0;
    }
    area - anchor
}

#[must_use]
#[inline]
fn swap_axis(xaxis: bool, v: Vec2) -> (f32, f32) {
    if xaxis {
        (v.x, v.y)
    } else {
        (v.y, v.x)
    }
}

/// If prev is NAN, always returns zero, which is the correct action for margin edges.
#[must_use]
#[inline]
fn merge_margin(prev: f32, margin: f32) -> f32 {
    if prev.is_nan() {
        0.0
    } else {
        margin.max(prev)
    }
}
