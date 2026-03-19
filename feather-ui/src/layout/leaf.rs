// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::base::Empty;
use super::{Concrete, Desc, Layout, base};
use crate::layout::DeferMachine;
use crate::reactive::{DynSignal, MutableSignal, SignalDebug, SignalZip, zip_pair};
use crate::{
    DRect, Limited, PxDim, PxLimits, PxPerimeter, PxPoint, PxRect, RelDim, Resolve, Unsizable,
    Unsized, UnsizedDim, rtree,
};
use std::rc::Rc;

pub trait Prop: base::Area + base::Limits + base::Anchor {}

crate::gen_dyn_prop!(Prop);

impl Prop for DRect {}

// Actual leaves do not require padding, but a lot of raw elements do (text,
// shape, images, etc.) This inherits Prop to allow elements to "extract" the
// padding for the rendering system for when it doesn't affect layouts.
pub trait Padded: Prop + base::Padding {}

crate::gen_dyn_prop!(Padded);

impl Padded for DRect {}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Empty;
    type Children = ();
    type Provider = dyn crate::reactive::SignalProvider<Item = (PxRect, RelDim)>;
    type Staging = (DynSignal<crate::URect<Unsized>>, MutableSignal<RelDim>);

    fn presize(
        props: &Self::Props,
        bounds: DynSignal<PxLimits>,
        dpi: MutableSignal<RelDim>,
        _: DynSignal<Self::Children>,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        let area = props.area().resolve(dpi.clone());
        let limits = (props.limits(), dpi.clone(), bounds, area.clone())
            .zip()
            .flatmap(|(limits, dpi, bounds, area)| {
                area.to_bounds(limits.resolve(*dpi).preresolve(*bounds))
            });

        let evaluated_area = crate::reactive::zip_pair(area.clone(), limits, |a, l| {
            a.resolve(PxDim::zero()).preresolve().limit(*l)
        });

        (evaluated_area.into_dyn(), (area.into_dyn(), dpi))
    }

    fn size(
        props: &Self::Props,
        dim: DynSignal<UnsizedDim>,
        bounds: DynSignal<crate::PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        let (area, dpi) = data;

        let final_area = (area.clone(), dim, bounds, props.limits(), dpi.clone())
            .zip()
            .flatmap(|(a, dim, b, limits, dpi)| {
                a.resolve(PxDim::zero())
                    .resolve(dim.zero_unsized())
                    .limit(limits.resolve(*dpi).resolve(b.to_bounds(*dim)))
            });

        (
            (final_area.clone(), props.anchor(), dpi.clone())
                .zip()
                .flatmap(|(area, a, dpi)| area.anchored(a.resolve(*dpi)))
                .into_dyn(),
            (area, dpi),
        )
    }

    fn stage<T: crate::render::Prerender + 'static>(
        _: &Self::Props,
        offset: DynSignal<PxPoint>,
        area: DynSignal<PxRect>,
        renderable: Option<T>,
        defer: Option<DeferMachine<Self::Provider>>,
        data: Self::Staging,
    ) -> Rc<rtree::Node> {
        let final_area = (area + offset).into_dyn();

        super::resolve_defer_machine(
            rtree::Node::new(
                final_area.clone(),
                None,
                Default::default(),
                Some(Box::new(Concrete::new(
                    renderable.as_ref(),
                    final_area.clone(),
                ))),
            ),
            &defer,
            (final_area, data.1).zip().value().into_dyn(),
        )
    }
}

/// A sized leaf is one with inherent size, like an image. This is used to
/// preserve aspect ratio when encounting an unsized axis. If both axis are
/// unsized, the inherent size is used as the intrinsic size of the node.
/// This must be provided in pixels.
#[derive_where::derive_where(Clone)]
#[derive(Debug)]
pub struct Sized<T, R: Clone> {
    pub props: Rc<T>,
    pub size: DynSignal<crate::PxDim>,
    pub renderable: Option<R>,
    pub machine: Option<DeferMachine<<dyn Prop as Desc>::Provider>>,
}

#[inline]
fn calc_sized_area(
    padding: PxPerimeter,
    area: crate::URect<Unsized>,
    size: PxDim,
    outer: PxDim,
) -> PxRect {
    let (unsized_x, unsized_y) = area.is_unsized();
    let aspect_ratio = size.width / size.height; // Will be NAN if both are 0, which disables any attempt to preserve aspect ratio
    match (unsized_x, unsized_y, aspect_ratio.is_finite()) {
        (true, false, false) => {
            let mut presize = area.resolve(PxDim::zero()).resolve(outer);
            let adjust = presize.dim().height * aspect_ratio;
            let v = presize.v.as_mut_array();
            v[2] += adjust;
            presize
        }
        (false, true, false) => {
            let mut presize = area.resolve(PxDim::zero()).resolve(outer);
            // Be careful, the aspect ratio here is being divided instead of multiplied
            let adjust = presize.dim().width / aspect_ratio;
            let v = presize.v.as_mut_array();
            v[3] += adjust;
            presize
        }
        _ => area.resolve(size + padding.total()).resolve(outer),
    }
}

impl<T: Padded + SignalDebug, R: crate::render::Prerender + Clone + SignalDebug + 'static> Layout
    for Sized<T, R>
{
    type Props = T;
    type Staging = (
        DynSignal<PxPerimeter>,
        DynSignal<crate::URect<Unsized>>,
        MutableSignal<RelDim>,
    );

    fn get_props(&self) -> &T {
        &self.props
    }

    fn presize(
        &self,
        bounds: DynSignal<PxLimits>,
        dpi: MutableSignal<RelDim>,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        let area = self.props.area().resolve(dpi.clone());
        let limits = (self.props.limits(), bounds, area.clone(), dpi.clone())
            .zip()
            .flatmap(|(limits, bounds, area, dpi)| {
                area.to_bounds(limits.resolve(*dpi).preresolve(*bounds))
            });

        let padding = self.props.padding().resolve(dpi.clone());

        // The way we handle unsized here is different from how we normally handle it.
        // If both axes are unsized, we simply set the area to the internal
        // size. If only one axis is unsized, we stretch it to maintain an aspect
        // ratio relative to the size of the other axis.
        let mapped_area = (padding.clone(), area.clone(), self.size.clone())
            .zip()
            .flatmap(|(padding, area, size)| {
                calc_sized_area(*padding, *area, *size, PxDim::zero())
            });

        let evaluated_area = mapped_area.limit(limits);

        (
            evaluated_area.into_dyn(),
            (padding.into_dyn(), area.into_dyn(), dpi.clone()),
        )
    }

    fn size(
        &self,
        dim: DynSignal<UnsizedDim>,
        bounds: DynSignal<crate::PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        let (padding, area, dpi) = data;

        let limits = self.props.limits().resolve(dpi.clone()).resolve(zip_pair(
            bounds,
            dim.clone(),
            |b, d| b.to_bounds(*d),
        ));

        let final_area = (
            padding.clone(),
            area.clone(),
            self.size.clone(),
            dim,
            limits,
        )
            .zip()
            .flatmap(move |(padding, area, size, outer, limits)| {
                calc_sized_area(*padding, *area, *size, outer.zero_unsized()).limit(*limits)
            });

        (
            (final_area.clone(), self.props.anchor().clone(), dpi.clone())
                .zip()
                .flatmap(|(area, a, d)| area.anchored(a.resolve(*d)))
                .into_dyn(),
            (padding, area, dpi),
        )
    }

    fn stage(
        &self,
        offset: DynSignal<PxPoint>,
        area: DynSignal<PxRect>,
        data: Self::Staging,
    ) -> Rc<rtree::Node> {
        let (_, _, dpi) = data;

        let final_area = (area + offset).into_dyn();

        super::resolve_defer_machine(
            rtree::Node::new(
                final_area.clone(),
                None,
                Default::default(),
                Some(Box::new(Concrete::new(
                    self.renderable.as_ref(),
                    final_area.clone(),
                ))),
            ),
            &self.machine,
            (final_area, dpi).zip().value().into_dyn(),
        )
    }
}
