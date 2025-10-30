// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

pub mod base;
//pub mod domain_write;
pub mod fixed;
//pub mod flex;
//pub mod grid;
//pub mod leaf;
//pub mod list;
pub mod root;
//pub mod text;

use guillotiere::euclid::{Point2D, Vector2D};
use wide::f32x4;

use crate::color::sRGB32;
use crate::reactive::sample;
use crate::render::Renderable;
use crate::render::compositor::CompositorView;
use crate::{
    DynSignal, Error, PxDim, PxLimits, PxPoint, PxRect, RelLimits, SourceID, UNSIZED_AXIS, URect,
    UnsizedDim, rtree,
};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

type StageThunk<'a> =
    Box<dyn FnMut(DynSignal<PxPoint>, DynSignal<PxDim>, DynSignal<PxLimits>) -> rtree::Node>;

/// Represents an arbitrary layout node that hasn't been staged yet. The vast
/// majority of the time, components should simply use the standard [`Node`]
/// implementation of this trait, which handles most common layout cases.
/// However, some components, like the text component, have complex layout logic
/// or special cases that [`Node`] can't cover.
pub trait Layout {
    type Props: ?Sized;

    fn get_props(&self) -> &Self::Props;
    fn stage<'a>(
        &self,
        dim: DynSignal<crate::UnsizedDim>,
        limits: DynSignal<PxLimits>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, StageThunk<'a>);
}

pub trait DynLayout<DynProps: ?Sized> {
    fn get_props(&self) -> &DynProps;
    fn stage<'a>(
        &self,
        dim: DynSignal<crate::UnsizedDim>,
        limits: DynSignal<PxLimits>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, StageThunk<'a>);
}

impl<T: ?Sized + 'static, U: Layout<Props = T>, P: ?Sized> DynLayout<P> for U
where
    for<'a> &'a T: Into<&'a P>,
{
    fn get_props(&self) -> &P {
        &Layout::get_props(self).into()
    }
    fn stage<'a>(
        &self,
        dim: DynSignal<crate::UnsizedDim>,
        limits: DynSignal<PxLimits>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, StageThunk<'a>) {
        Layout::stage(self, dim, limits, dpi)
    }
}

pub trait Desc {
    type Props: ?Sized;
    type Child: ?Sized;
    type Children;

    fn stage<'a>(
        props: &Self::Props,
        outer: DynSignal<crate::UnsizedDim>,
        limits: DynSignal<PxLimits>,
        children: DynSignal<Self::Children>,
        renderable: Option<Rc<dyn Renderable>>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, StageThunk<'a>);
}

/// The standard layout node. Expects the layout properties, which must be
/// compatible with the layout description `D` provided, which also determines
/// the type that contains the children. A unique ID must be provided, and a
/// renderable is optional - it will be passed to staging if provided. The
/// layer, if provided will create a new layer operation with the given color
/// and rotation. This is normally used to do correct transparency.
///
/// # Examples
/// See [`super::component::Component`]
pub struct Node<T, D: Desc + ?Sized> {
    pub props: Rc<T>,
    pub children: DynSignal<D::Children>,
    pub renderable: Option<Rc<dyn Renderable>>,
    pub layer: Option<(sRGB32, f32)>,
}

impl<T, D: Desc + ?Sized> Layout for Node<T, D>
where
    for<'a> &'a T: Into<&'a D::Props>,
{
    type Props = T;

    fn get_props(&self) -> &T {
        self.props.as_ref()
    }
    fn stage<'a>(
        &self,
        dim: DynSignal<crate::UnsizedDim>,
        limits: DynSignal<PxLimits>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, StageThunk<'a>) {
        let mut staged = D::stage(
            self.props.as_ref().into(),
            dim,
            limits,
            self.children.clone(),
            self.renderable.as_ref().map(|x| x.clone()),
            dpi,
        );
        /*if let Some((color, rotation)) = self.layer {
            window.driver.shared.create_layer(
                &window.driver.device,
                self.id.upgrade().unwrap(),
                staged.get_area().to_untyped(),
                None,
                color,
                rotation,
                false,
            );
            staged.set_layer(self.id.clone());
        }*/
        staged
    }
}

pub trait Staged {
    fn render(
        &self,
        parent_pos: PxPoint,
        area: PxRect,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
        children: Option<&imbl::Vector<Rc<rtree::Node>>>,
        dependents: &mut Vec<std::sync::Weak<SourceID>>,
    ) -> Result<(), Error>;
    fn set_layer(&mut self, _id: std::sync::Weak<SourceID>) {
        panic!("This staged object doesn't support layers!");
    }
}

pub(crate) struct Concrete {
    renderable: Option<Rc<dyn Renderable>>,
    layer: Option<std::sync::Weak<SourceID>>,
}

impl Concrete {
    pub fn new(renderable: Option<Rc<dyn Renderable>>) -> Self {
        Self {
            renderable,
            layer: None,
        }
    }

    fn render_self(
        &self,
        area: PxRect,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), Error> {
        if let Some(r) = &self.renderable {
            r.render(area.to_untyped(), driver, compositor)?;
        }
        Ok(())
    }

    fn render_children(
        &self,
        parent_pos: PxPoint,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
        children: &imbl::Vector<Rc<rtree::Node>>,
        dependents: &mut Vec<std::sync::Weak<SourceID>>,
    ) -> Result<(), Error> {
        for child in children {
            // The r-tree node will determine whether it should be culled in its render function
            child.render(parent_pos, driver, compositor, dependents)?;
        }
        Ok(())
    }
}

impl Staged for Concrete {
    fn render(
        &self,
        parent_pos: PxPoint,
        area: PxRect,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
        children: Option<&imbl::Vector<Rc<rtree::Node>>>,
        dependents: &mut Vec<std::sync::Weak<SourceID>>,
    ) -> Result<(), Error> {
        if let Some(id) = self.layer.as_ref().and_then(|x| x.upgrade()) {
            let layers = driver.shared.access_layers();
            let layer = layers.get(&id).expect("Missing layer in render call!");
            let mut deps = Vec::new();
            let mut region_uv = None;

            let (mut view, depview) = if layer.target.is_some() {
                // If this is a "real" layer with a texture target, mark it as a dependency of
                // our parent
                dependents.push(Arc::downgrade(&id));

                // Acquire a region if we don't have one already. This is done carefully so that
                // the layer can be moved to a different dependency layer and
                // therefore switched to a different atlas without the user render functions
                // needing to care.
                let index = match compositor.index {
                    0 => 1,
                    1 => 2,
                    2 => 1,
                    _ => panic!("Invalid index!"),
                };

                let mut atlas = driver.layer_atlas[index - 1].write();
                let region = atlas.cache_region(
                    &driver.device,
                    &id,
                    layer.area.dim().ceil().to_i32(),
                    None,
                    None,
                )?;
                region_uv = Some(region.uv);

                // Make sure we aren't cached in the opposite atlas
                driver.layer_atlas[index % 2].write().remove_cache(&id);
                assert!(compositor.pass < 0b111111);

                let mut v = CompositorView {
                    index: index as u8,
                    window: compositor.window,
                    layer0: compositor.layer0,
                    layer1: compositor.layer1,
                    clipstack: compositor.clipstack,
                    offset: region.uv.min.to_f32() - layer.area.topleft() - parent_pos.to_vector(),
                    surface_dim: compositor.surface_dim,
                    pass: compositor.pass + 1,
                    slice: region.index,
                };

                v.reserve(driver);
                // And return a reference to a new dependency vector
                (v, &mut deps)
            } else {
                // Otherwise, we don't create a new compositor view, instead copying our
                // previous one, and passing in the parent's dependency tracker.
                (
                    CompositorView {
                        index: compositor.index,
                        window: compositor.window,
                        layer0: compositor.layer0,
                        layer1: compositor.layer1,
                        clipstack: compositor.clipstack,
                        offset: compositor.offset,
                        surface_dim: compositor.surface_dim,
                        pass: compositor.pass,
                        slice: compositor.slice,
                    },
                    dependents,
                )
            };

            // Always push a new clipping area, but remember that a layer can only store
            // it's relative area.
            view.with_clip(layer.area + parent_pos, |refview| {
                self.render_self(area + parent_pos, driver, refview)?;
                if let Some(c) = children {
                    self.render_children(
                        parent_pos + area.topleft().to_vector(),
                        driver,
                        refview,
                        c,
                        depview,
                    )?;
                }
                Ok(())
            })?;

            if let Some(target) = layer.target.as_ref() {
                // If this was a real layer, now we need to actually assign the result of our
                // dependencies, and append ourselves to the parent layer. We
                // must be very careful not to use the wrong view here.
                target.write().dependents = deps;
                compositor.append_layer(layer, parent_pos, region_uv.unwrap());
            }
        } else {
            self.render_self(area + parent_pos, driver, compositor)?;
            if let Some(c) = children {
                self.render_children(
                    parent_pos + area.topleft().to_vector(),
                    driver,
                    compositor,
                    c,
                    dependents,
                )?;
            }
        };

        Ok(())
    }

    fn set_layer(&mut self, id: std::sync::Weak<SourceID>) {
        self.layer = Some(id)
    }
}

#[must_use]
#[inline]
pub(crate) fn map_unsized_area(mut area: URect, adjust: PxDim) -> URect {
    let (unsized_x, unsized_y) = check_unsized(area);
    let abs = area.abs.v.as_array_mut();
    let rel = area.rel.v.as_array_mut();
    // Unsized objects must always have a single anchor point to make sense, so we
    // copy over from topleft.
    if unsized_x {
        rel[2] = rel[0];
        // Fix the bottomright abs area in unsized scenarios, because it was relative to
        // the topleft instead of being independent.
        abs[2] += abs[0] + adjust.width;
    }
    if unsized_y {
        rel[3] = rel[1];
        abs[3] += abs[1] + adjust.height;
    }
    area
}

#[must_use]
#[inline]
pub(crate) fn zero_unsized(v: UnsizedDim) -> PxDim {
    let (unsized_x, unsized_y) = check_unsized_dim(v);
    PxDim {
        width: if unsized_x { 0.0 } else { v.width },
        height: if unsized_y { 0.0 } else { v.height },
        _unit: PhantomData,
    }
}

#[must_use]
#[inline]
pub(crate) fn limit_area(mut v: PxRect, limits: PxLimits) -> PxRect {
    // We do this by checking clamp(topleft + limit) instead of clamp(bottomright -
    // topleft) because this avoids floating point precision issues.
    v.set_bottomright(
        v.bottomright()
            .max(v.topleft() + limits.min())
            .min(v.topleft() + limits.max()),
    );
    v
}

#[must_use]
#[inline]
pub(crate) fn limit_dim(v: crate::UnsizedDim, limits: PxLimits) -> PxDim {
    let (unsized_x, unsized_y) = check_unsized_dim(v);
    PxDim::new(
        if unsized_x {
            v.width
        } else {
            v.width.max(limits.min().width).min(limits.max().width)
        },
        if unsized_y {
            v.height
        } else {
            v.height.max(limits.min().height).min(limits.max().height)
        },
    )
}

#[must_use]
#[inline]
pub(crate) fn eval_dim(area: URect, dim: PxDim, limits: PxLimits) -> UnsizedDim {
    let (unsized_x, unsized_y) = check_unsized(area);
    UnsizedDim::new(
        if unsized_x {
            area.bottomright().rel().x
        } else {
            let left = area.topleft().abs().x + (area.topleft().rel().x * dim.width);
            let right = area.bottomright().abs().x + (area.bottomright().rel().x * dim.width);
            (right - left)
                .max(limits.min().width)
                .min(limits.max().width)
        },
        if unsized_y {
            area.bottomright().rel().y
        } else {
            let top = area.topleft().abs().y + (area.topleft().rel().y * dim.height);
            let bottom = area.bottomright().abs().y + (area.bottomright().rel().y * dim.height);
            (bottom - top)
                .max(limits.min().height)
                .min(limits.max().height)
        },
    )
}

#[must_use]
#[inline]
pub(crate) fn apply_limit(dim: UnsizedDim, limits: PxLimits, rlimits: RelLimits) -> PxLimits {
    let (unsized_x, unsized_y) = check_unsized_dim(dim);
    let sign = limits.v.sign_bit() | rlimits.v.sign_bit();

    let px = f32x4::new([
        if unsized_x {
            limits.min().width
        } else {
            limits.min().width.max(dim.width)
        },
        if unsized_y {
            limits.min().height
        } else {
            limits.min().height.max(dim.height)
        },
        if unsized_x {
            limits.max().width
        } else {
            limits.max().width.min(dim.width)
        },
        if unsized_y {
            limits.max().height
        } else {
            limits.max().height.min(dim.height)
        },
    ]);

    PxLimits {
        v: (rlimits.v.is_finite().blend(px, f32x4::ONE) * rlimits.v).copysign(sign),
        _unit: PhantomData,
    }
}

// Returns true if an axis is unsized, which means it is defined as the size of
// it's children's maximum extent.
#[must_use]
#[inline]
pub(crate) fn check_unsized(area: URect) -> (bool, bool) {
    (
        area.bottomright().rel().x == UNSIZED_AXIS,
        area.bottomright().rel().y == UNSIZED_AXIS,
    )
}

// Returns true if an axis is unsized, which means it is defined as the size of
// it's children's maximum extent.
#[must_use]
#[inline]
pub(crate) fn check_unsized_dim(dim: UnsizedDim) -> (bool, bool) {
    (dim.width == UNSIZED_AXIS, dim.height == UNSIZED_AXIS)
}

pub(crate) fn assert_sized(area: PxRect) {
    let ltrb = area.v.as_array_ref();

    for v in ltrb {
        assert_ne!(*v, UNSIZED_AXIS);
        assert!(v.is_finite());
    }
}

#[must_use]
#[inline]
pub(crate) fn cap_unsized(area: PxRect) -> PxRect {
    let ltrb = area.v.to_array();
    PxRect {
        v: f32x4::new(ltrb.map(|x| {
            if x.is_finite() {
                x
            } else {
                crate::UNSIZED_AXIS
            }
        })),
        _unit: PhantomData,
    }
}

#[must_use]
#[inline]
fn swap_pair<T>(xaxis: bool, v: (T, T)) -> (T, T) {
    if xaxis { (v.0, v.1) } else { (v.1, v.0) }
}

trait Swappable<T> {
    fn swap_axis(self, xaxis: bool) -> (T, T);
}

impl<T, U> Swappable<T> for Point2D<T, U> {
    #[inline]
    fn swap_axis(self, xaxis: bool) -> (T, T) {
        swap_pair(xaxis, (self.x, self.y))
    }
}

impl<T, U> Swappable<T> for guillotiere::euclid::Size2D<T, U> {
    #[inline]
    fn swap_axis(self, xaxis: bool) -> (T, T) {
        swap_pair(xaxis, (self.width, self.height))
    }
}

impl<T, U> Swappable<T> for Vector2D<T, U> {
    #[inline]
    fn swap_axis(self, xaxis: bool) -> (T, T) {
        swap_pair(xaxis, (self.x, self.y))
    }
}

/// If prev is NAN, always returns zero, which is the correct action for margin
/// edges.
#[must_use]
#[inline]
fn merge_margin(prev: f32, margin: f32) -> f32 {
    if prev.is_nan() { 0.0 } else { margin.max(prev) }
}
