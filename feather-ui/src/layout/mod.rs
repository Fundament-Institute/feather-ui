// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

pub mod base;
pub mod domain_write;
pub mod fixed;
//pub mod flex;
//pub mod grid;
pub mod leaf;
//pub mod list;
pub mod root;
pub mod text;

use guillotiere::euclid::{Point2D, Vector2D};
use wide::f32x4;

use crate::reactive::{self, MutableSignal, SignalDebug};
use crate::render::compositor::{CompositorView, Layer};
use crate::render::{Prerender, Renderable};
use crate::{
    DynSignal, Limited, PxDim, PxLimits, PxPoint, PxRect, RelDim, RelLimits, RenderError, Resolve,
    UNSIZED_AXIS, UPerimeter, URect, Unsizable, Unsized, UnsizedDim, rtree,
};
use std::any::Any;
use std::marker::PhantomData;
use std::rc::Rc;

/// Represents an arbitrary layout node that hasn't been staged yet. The vast
/// majority of the time, components should simply use the standard [`Node`]
/// implementation of this trait, which handles most common layout cases.
/// However, some components, like the text component, have complex layout logic
/// or special cases that [`Node`] can't cover.
pub trait Layout: reactive::SignalDebug {
    type Props: ?Sized;
    type Staging: 'static;

    fn get_props(&self) -> &Self::Props;

    /// Returns the intrinsic size (sets relative coordinates to 0) of this node
    /// based on the provided bounds
    fn presize(
        &self,
        bounds: DynSignal<PxLimits>,
        dpi: MutableSignal<RelDim>,
    ) -> (DynSignal<PxRect>, Self::Staging);
    /// Returns either a partial area (if `dim` is partially unsized) or a final true area (if `dim` is sized)
    fn size(
        &self,
        dim: DynSignal<UnsizedDim>,
        bounds: DynSignal<PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<PxRect>, Self::Staging);
    /// Stages the layout using a previously calculated true area returned from [`Self::size()`]
    fn stage(
        &self,
        offset: DynSignal<PxPoint>,
        area: DynSignal<PxRect>,
        data: Self::Staging,
    ) -> Rc<rtree::Node>;
}

pub trait DynLayout<DynProps: ?Sized>: reactive::SignalDebug {
    fn get_props(&self) -> &DynProps;
    /// Returns the intrinsic size (sets relative coordinates to 0) of this node
    /// based on the provided bounds
    fn presize(
        &self,
        bounds: DynSignal<PxLimits>,
        dpi: MutableSignal<RelDim>,
    ) -> (DynSignal<PxRect>, Box<dyn Any>);
    /// Returns either a partial area (if `dim` is partially unsized) or a final true area (if `dim` is sized)
    fn size(
        &self,
        dim: DynSignal<UnsizedDim>,
        bounds: DynSignal<PxLimits>,
        data: &dyn Any,
    ) -> (DynSignal<PxRect>, Box<dyn Any>);
    /// Stages the layout using a previously calculated true area returned from [`Self::size()`]
    fn stage(
        &self,
        offset: DynSignal<PxPoint>,
        area: DynSignal<PxRect>,
        data: &dyn Any,
    ) -> Rc<rtree::Node>;
}

impl<T: ?Sized + 'static, U: Layout<Props = T> + reactive::SignalDebug, P: ?Sized> DynLayout<P>
    for U
where
    for<'a> &'a T: Into<&'a P>,
    U::Staging: Clone,
{
    fn get_props(&self) -> &P {
        Layout::get_props(self).into()
    }

    fn presize(
        &self,
        bounds: DynSignal<PxLimits>,
        dpi: MutableSignal<RelDim>,
    ) -> (DynSignal<PxRect>, Box<dyn Any>) {
        let (presize, data) = Layout::presize(self, bounds, dpi);
        (presize, Box::new(data))
    }

    fn size(
        &self,
        dim: DynSignal<UnsizedDim>,
        bounds: DynSignal<PxLimits>,
        data: &dyn Any,
    ) -> (DynSignal<PxRect>, Box<dyn Any>) {
        let (area, data) = Layout::size(
            self,
            dim,
            bounds,
            data.downcast_ref::<U::Staging>()
                .expect("Wrong Staging type passed to Layout")
                .clone(),
        );
        (area, Box::new(data))
    }

    fn stage(
        &self,
        offset: DynSignal<PxPoint>,
        area: DynSignal<PxRect>,
        data: &dyn Any,
    ) -> Rc<rtree::Node> {
        Layout::stage(
            self,
            offset,
            area,
            data.downcast_ref::<U::Staging>()
                .expect("Wrong Staging type passed to Layout")
                .clone(),
        )
    }
}

pub type DeferMachine<P> = (
    crate::event::AntiStream<'static, crate::input::RawEvent, Rc<rtree::Node>>,
    reactive::Signal<reactive::DeferProvider<P>>,
);

#[inline]
pub fn resolve_defer_machine<P: reactive::SignalProvider + ?Sized>(
    node: rtree::Node,
    defer: &Option<DeferMachine<P>>,
    target: reactive::Signal<P>,
) -> Rc<rtree::Node> {
    let n = Rc::new(node);
    if let Some((machine, state)) = defer {
        machine.connect(n.clone()).unwrap();
        state
            .set(target)
            .map_err(|_| ())
            .expect("State already resolved!");
    }
    n
}

pub trait Desc {
    type Props: ?Sized;
    type Child: ?Sized;
    type Children;
    type Provider: reactive::SignalProvider + ?Sized;
    type Staging: 'static;

    fn presize(
        props: &Self::Props,
        bounds: DynSignal<PxLimits>,
        dpi: MutableSignal<RelDim>,
        children: DynSignal<Self::Children>,
    ) -> (DynSignal<PxRect>, Self::Staging);

    fn size(
        props: &Self::Props,
        dim: DynSignal<UnsizedDim>,
        bounds: DynSignal<PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<PxRect>, Self::Staging);

    fn stage<T: Prerender + 'static>(
        props: &Self::Props,
        offset: DynSignal<PxPoint>,
        area: DynSignal<PxRect>,
        renderable: Option<T>,
        defer: Option<DeferMachine<Self::Provider>>,
        data: Self::Staging,
    ) -> Rc<rtree::Node>;
}

/// The standard layout node. Expects the layout properties, which must be
/// compatible with the layout description `D` provided, which also determines
/// the type that contains the children. A unique ID must be provided, and a
/// renderable is optional - it will be passed to staging if provided.
///
/// # Examples
/// See [`super::component::Component`]
pub struct Node<T, D: Desc + ?Sized, R> {
    pub props: Rc<T>,
    pub children: DynSignal<D::Children>,
    pub renderable: Option<R>,
    pub machine: Option<DeferMachine<D::Provider>>,
}

impl<T, D: Desc + ?Sized, R> std::fmt::Debug for Node<T, D, R>
where
    T: std::fmt::Debug,
    R: std::fmt::Debug,
    D::Children: std::fmt::Debug,
    <D::Provider as crate::reactive::SignalProvider>::Item: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("props", &self.props)
            .field("children", &self.children)
            .field("renderable", &self.renderable)
            .field("machine", &self.machine)
            .finish()
    }
}

impl<T: SignalDebug, D: Desc + ?Sized, R: Prerender + Clone + SignalDebug + 'static> Layout
    for Node<T, D, R>
where
    for<'a> &'a T: Into<&'a D::Props>,
    <D as Desc>::Children: SignalDebug,
    <D::Provider as crate::reactive::SignalProvider>::Item: SignalDebug,
{
    type Props = T;
    type Staging = D::Staging;

    fn get_props(&self) -> &T {
        self.props.as_ref()
    }

    fn presize(
        &self,
        bounds: DynSignal<PxLimits>,
        dpi: MutableSignal<RelDim>,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        D::presize(
            self.props.as_ref().into(),
            bounds,
            dpi,
            self.children.clone(),
        )
    }

    fn size(
        &self,
        dim: DynSignal<UnsizedDim>,
        bounds: DynSignal<PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        D::size(self.props.as_ref().into(), dim, bounds, data)
    }

    fn stage(
        &self,
        offset: DynSignal<PxPoint>,
        area: DynSignal<PxRect>,
        data: Self::Staging,
    ) -> Rc<rtree::Node> {
        D::stage(
            self.props.as_ref().into(),
            offset,
            area,
            self.renderable.clone(),
            self.machine.clone(),
            data,
        )
    }
}

pub trait Staged {
    fn render(
        &self,
        parent_pos: PxPoint,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
        children: Option<&imbl::Vector<Rc<rtree::Node>>>,
        dependents: &mut Vec<std::rc::Weak<Layer>>,
    ) -> Result<(), RenderError>;
}

pub(crate) struct Concrete<T: Renderable> {
    renderable: Option<T>,
    layer: Option<Rc<Layer>>,
    area: DynSignal<PxRect>,
}

impl<T: Renderable> Concrete<T> {
    pub fn new<R: Prerender<R = T>>(prerender: Option<&R>, area: DynSignal<PxRect>) -> Self {
        Self {
            renderable: prerender.map(|x| x.prerender(area.clone())),
            layer: None,
            area,
        }
    }

    fn render_self(
        &self,
        parent_pos: PxPoint,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), RenderError> {
        if let Some(r) = &self.renderable {
            r.render(parent_pos, driver, compositor)?;
        }
        Ok(())
    }

    fn render_children(
        &self,
        parent_pos: PxPoint,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
        children: &imbl::Vector<Rc<rtree::Node>>,
        dependents: &mut Vec<std::rc::Weak<Layer>>,
    ) -> Result<(), RenderError> {
        for child in children {
            // The r-tree node will determine whether it should be culled in its render function
            child.render(parent_pos, driver, compositor, dependents)?;
        }
        Ok(())
    }
}

impl<T: Renderable> Staged for Concrete<T> {
    fn render(
        &self,
        parent_pos: PxPoint,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
        children: Option<&imbl::Vector<Rc<rtree::Node>>>,
        dependents: &mut Vec<std::rc::Weak<Layer>>,
    ) -> Result<(), RenderError> {
        let area = *reactive::sample(&self.area);
        compositor.redraw.add_parent(&self.area); // Added because we get parent_pos from this

        if let Some(layer) = &self.layer {
            let mut deps = Vec::new();

            let layer_area = *reactive::sample(&layer.area);
            compositor.redraw.add_parent(&layer.area); // Added because we calculate the cliprect from this
            compositor.redraw.add_parent(&layer.color);
            compositor.redraw.add_parent(&layer.rotation);

            let target = reactive::sample(&layer.target);
            let (mut view, depview) = if let Some(target) = &*target {
                // If this is a "real" layer with a texture target, mark it as a dependency of
                // our parent
                dependents.push(Rc::downgrade(layer));

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

                target
                    .write()
                    .update(layer_area.dim().ceil().to_i32(), index - 1, driver)?;

                let region_uv = target.read().region.uv;
                let region_index = target.read().region.index;
                assert!(compositor.pass < 0b111111);

                let mut v = CompositorView {
                    index: index as u8,
                    window: compositor.window,
                    layer0: compositor.layer0,
                    layer1: compositor.layer1,
                    clipstack: compositor.clipstack,
                    offset: region_uv.min.to_f32() - layer_area.topleft() - parent_pos.to_vector(),
                    surface_dim: compositor.surface_dim,
                    pass: compositor.pass + 1,
                    slice: region_index,
                    redraw: compositor.redraw,
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
                        redraw: compositor.redraw,
                    },
                    dependents,
                )
            };

            // Always push a new clipping area, but remember that a layer can only store
            // it's relative area.
            view.with_clip(layer_area + parent_pos, |refview| {
                self.render_self(parent_pos, driver, refview)?;
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

            if let Some(target) = &*target {
                // If this was a real layer, now we need to actually assign the result of our
                // dependencies, and append ourselves to the parent layer. We
                // must be very careful not to use the wrong view here.
                target.write().dependents = deps;
                compositor.append_layer(layer, parent_pos, target.read().region.uv);
            }
        } else {
            self.render_self(parent_pos, driver, compositor)?;
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
}

/// Resolves a potentially unsized area using a potentially unsized parent dim
/// and returns the result as a potentially unsized dim for use in child area
/// calculations
#[must_use]
#[inline]
pub fn resolve_dim(dim: UnsizedDim, area: URect<Unsized>, intrinsic_size: PxDim) -> UnsizedDim {
    // If the dim is unsized AND the area is unsized on an axis, this must be carried
    // forward into the result

    let (dim_unsized_x, dim_unsized_y) = dim.is_unsized();
    let (unsized_x, unsized_y) = area.is_unsized();
    let v_abs = area.abs.v.as_array();
    let v_rel = area.rel.v.as_array();
    // Unsized objects must always have a single anchor point to make sense, so we
    // copy over from topleft.
    let x = if unsized_x {
        if dim_unsized_x {
            dim.width
        } else {
            // when unsized, the relative coordinate never contributes to the dimension
            v_abs[2] + intrinsic_size.width
        }
    } else {
        ((v_rel[2] - v_rel[0]) * dim.width) + (v_abs[2] - v_abs[0])
    };
    let y = if unsized_y {
        if dim_unsized_y {
            dim.height
        } else {
            // when unsized, the relative coordinate never contributes to the dimension
            v_abs[3] + intrinsic_size.height
        }
    } else {
        ((v_rel[3] - v_rel[1]) * dim.height) + (v_abs[3] - v_abs[1])
    };
    UnsizedDim::new(x, y)
}

#[must_use]
#[inline]
pub(crate) fn apply_limit(dim: UnsizedDim, limits: PxLimits, rlimits: RelLimits) -> PxLimits {
    let (unsized_x, unsized_y) = dim.is_unsized();
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

pub(crate) fn assert_sized(area: PxRect) {
    let ltrb = area.v.as_array();

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

struct ReferenceProps {
    area: crate::DRect,
    limits: crate::DLimits,
    anchor: crate::DPoint,
    padding: UPerimeter,
    intrinsic: either::Either<PxDim, Vec<ReferenceProps>>,
    prev_size: PxDim,
}

impl Default for ReferenceProps {
    fn default() -> Self {
        Self {
            area: Default::default(),
            limits: Default::default(),
            anchor: Default::default(),
            padding: Default::default(),
            intrinsic: either::Either::Left(PxDim::default()),
            prev_size: Default::default(),
        }
    }
}

fn reference_presize(
    bounds: PxLimits,
    dpi: RelDim,
    area: crate::DRect,
    dlimits: crate::DLimits,
    anchor: crate::DPoint,
    padding: UPerimeter,
    intrinsic: &mut either::Either<PxDim, Vec<ReferenceProps>>,
    prev_size: &mut PxDim,
) -> PxRect {
    let area = area.resolve(dpi);
    let limits = area.to_bounds(dlimits.resolve(dpi).preresolve(bounds));
    *prev_size = match intrinsic {
        either::Either::Left(size) => *size,
        either::Either::Right(children) => children
            .iter_mut()
            .fold(PxRect::zero(), |size, child| {
                size.extend(reference_presize(
                    limits,
                    dpi,
                    child.area,
                    child.limits,
                    child.anchor,
                    child.padding,
                    &mut child.intrinsic,
                    &mut child.prev_size,
                ))
            })
            .bottomright()
            .to_vector()
            .to_size(),
    };

    area.resolve(*prev_size + padding.resolve(dpi).total())
        .preresolve()
        .limit(limits)
        .anchored(anchor.resolve(dpi))
}

fn reference_size(
    dim: UnsizedDim,
    bounds: PxLimits,
    dpi: RelDim,
    area: crate::DRect,
    dlimits: crate::DLimits,
    anchor: crate::DPoint,
    padding: UPerimeter,
    intrinsic: &mut either::Either<PxDim, Vec<ReferenceProps>>,
    prev_size: &mut PxDim,
) -> PxRect {
    let limits = dlimits.resolve(dpi).resolve(bounds.to_bounds(dim));
    let child_dim = area
        .resolve(dpi)
        .resolve(*prev_size + padding.resolve(dpi).total())
        .partial_resolve(dim)
        .limit(limits);

    let size = match intrinsic {
        either::Either::Left(size) => *size,
        either::Either::Right(children) => children
            .iter_mut()
            .fold(PxRect::zero(), |size, child| {
                size.extend(reference_size(
                    child_dim,
                    // This is redundant in a standard fixed-size layout, but necessary because children
                    // control how they utilize their own bounds, so we must provide a maximally correct
                    // bounds even if a normal fixed-size layout doesn't need it.
                    limits.to_bounds(child_dim),
                    dpi,
                    child.area,
                    child.limits,
                    child.anchor,
                    child.padding,
                    &mut child.intrinsic,
                    &mut child.prev_size,
                ))
            })
            .bottomright()
            .to_vector()
            .to_size(),
    };
    *prev_size = size;

    area.resolve(dpi)
        .resolve(size + padding.resolve(dpi).total())
        .resolve(dim.zero_unsized())
        .limit(limits)
        .anchored(anchor.resolve(dpi))
}

fn reference_stage(offset: PxPoint, area: PxRect) -> PxRect {
    area + offset
}

#[test]
fn test_counterexamples() {
    // anchored counter-example to using bottomright instead of dim

    // limits counterexample for text

    // limits counterexample for why fully seperate RLimits doesn't work

    // Nested RLimit example
    {
        let child_auto = ReferenceProps {
            area: AUTO_DRECT,
            intrinsic: either::Either::Left(PxDim::new(150.0, 150.0)),
            ..Default::default()
        };
        let child_fill = ReferenceProps {
            area: FILL_DRECT,
            ..Default::default()
        };
        let parent = ReferenceProps {
            area: PxRect::new(0.0, 0.0, 300.0, 300.0).into(),
            limits: PxLimits::new(..200.0, ..).into(),
            intrinsic: either::Either::Right(vec![child_auto, child_fill]),
            ..Default::default()
        };
    }
}
