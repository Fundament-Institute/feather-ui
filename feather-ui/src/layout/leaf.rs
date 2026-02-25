// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::base::Empty;
use super::{Concrete, Desc, Layout, base};
use crate::layout::DeferMachine;
use crate::reactive::{DynSignal, SignalZip, zip_pair};
use crate::{DRect, PxDim, PxPerimeter, PxRect, RelDim, Unsizable, rtree};
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
    type Provider = dyn crate::reactive::SignalProvider<Item = (PxRect, crate::RelDim)>;

    fn stage<'a, T: crate::render::Prerender + 'static>(
        props: &Self::Props,
        prelimits: DynSignal<crate::PxLimits>,
        _: DynSignal<Self::Children>,
        renderable: Option<T>,
        dpi: crate::reactive::MutableSignal<RelDim>,
        defer: Option<super::DeferMachine<Self::Provider>>,
    ) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
        let limits = (props.limits(), dpi.clone(), prelimits.clone())
            .zip()
            .flatmap(|(limits, dpi, prelimits)| limits.resolve(*dpi) + *prelimits);

        let myarea = zip_pair(props.area(), dpi.clone(), |p, dpi| p.resolve(*dpi));

        let evaluated_area = crate::reactive::zip((myarea.clone(), limits.clone()))
            .flatmap(|(a, l)| super::limit_area(a.preresolve(PxDim::zero()), *l));

        let anchor = props.anchor();
        (
            evaluated_area.into(),
            Box::new(move |offset, final_dim, final_limits| {
                let final_area = (
                    myarea.clone(),
                    offset,
                    final_dim,
                    limits.clone(),
                    final_limits.clone(),
                )
                    .zip()
                    .flatmap(|(a, o, dim, limits, l2)| {
                        super::limit_area(a.resolve(*dim, PxDim::zero()), *limits + *l2) + *o
                    });

                let anchored_area = (final_area.clone(), anchor.clone(), dpi.clone())
                    .zip()
                    .flatmap(|(area, a, d)| *area - (a.resolve(*d) * area.dim()))
                    .into_dyn();

                super::resolve_defer_machine(
                    rtree::Node::new(
                        anchored_area.clone(),
                        None,
                        Default::default(),
                        Some(Box::new(Concrete::new(
                            renderable.as_ref(),
                            anchored_area.clone(),
                        ))),
                    ),
                    &defer,
                    (anchored_area, dpi.clone()).zip().value().into_dyn(),
                )
            }),
        )
    }
}

/// A sized leaf is one with inherent size, like an image. This is used to
/// preserve aspect ratio when encounting an unsized axis. If both axis are
/// unsized, the inherent size is used as the intrinsic size of the node.
/// This must be provided in pixels.
#[derive_where::derive_where(Clone)]
pub struct Sized<T, R: Clone> {
    pub props: Rc<T>,
    pub size: DynSignal<crate::PxDim>,
    pub renderable: Option<R>,
    pub machine: Option<DeferMachine<<dyn Prop as Desc>::Provider>>,
}

#[inline]
fn calc_sized_area(padding: PxPerimeter, area: crate::URect, size: PxDim, outer: PxDim) -> PxRect {
    let (unsized_x, unsized_y) = area.is_unsized();
    let aspect_ratio = size.width / size.height; // Will be NAN if both are 0, which disables any attempt to preserve aspect ratio
    match (unsized_x, unsized_y, aspect_ratio.is_finite()) {
        (true, false, false) => {
            let mut presize = area.resolve(outer, PxDim::zero());
            let adjust = presize.dim().height * aspect_ratio;
            let v = presize.v.as_array_mut();
            v[2] += adjust;
            presize
        }
        (false, true, false) => {
            let mut presize = area.resolve(outer, PxDim::zero());
            // Be careful, the aspect ratio here is being divided instead of multiplied
            let adjust = presize.dim().width / aspect_ratio;
            let v = presize.v.as_array_mut();
            v[3] += adjust;
            presize
        }
        _ => area.resolve(outer, size + padding.total()),
    }
}

impl<T: Padded, R: crate::render::Prerender + Clone + 'static> Layout for Sized<T, R> {
    type Props = T;

    fn get_props(&self) -> &T {
        &self.props
    }
    fn stage<'a>(
        &self,
        prelimits: DynSignal<crate::PxLimits>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, super::StageThunk<'a>) {
        let limits = (self.props.limits(), dpi.clone(), prelimits.clone())
            .zip()
            .flatmap(|(limits, dpi, prelimits)| limits.resolve(*dpi) + *prelimits);

        let padding = zip_pair(self.props.padding(), dpi.clone(), |p, dpi| {
            p.as_perimeter(*dpi)
        });
        let area = zip_pair(self.props.area(), dpi.clone(), |p, dpi| p.resolve(*dpi));

        // The way we handle unsized here is different from how we normally handle it.
        // If both axes are unsized, we simply set the area to the internal
        // size. If only one axis is unsized, we stretch it to maintain an aspect
        // ratio relative to the size of the other axis.
        let mapped_area = (padding.clone(), area.clone(), self.size.clone())
            .zip()
            .flatmap(|(padding, area, size)| {
                calc_sized_area(*padding, *area, *size, PxDim::zero())
            });

        let size = self.size.clone();
        let evaluated_area = zip_pair(mapped_area, limits.clone(), |a, l| {
            super::limit_area(*a, *l)
        });
        let renderable = self.renderable.clone();
        let anchor = self.props.anchor();
        let defer = self.machine.clone();

        (
            evaluated_area.into(),
            Box::new(move |offset, final_dim, final_limits| {
                let final_area = (
                    padding.clone(),
                    area.clone(),
                    size.clone(),
                    offset,
                    final_dim,
                    limits.clone(),
                    final_limits.clone(),
                )
                    .zip()
                    .flatmap(move |(padding, area, size, o, outer, limits, l2)| {
                        super::limit_area(
                            calc_sized_area(*padding, *area, *size, *outer),
                            *limits + *l2,
                        ) + *o
                    });

                // debug_assert!(anchored_area.v.is_finite().all());
                let anchored_area = (final_area.clone(), anchor.clone(), dpi.clone())
                    .zip()
                    .flatmap(|(area, a, d)| *area - (a.resolve(*d) * area.dim()))
                    .into_dyn();

                super::resolve_defer_machine(
                    rtree::Node::new(
                        anchored_area.clone(),
                        None,
                        Default::default(),
                        Some(Box::new(Concrete::new(renderable.as_ref(), anchored_area))),
                    ),
                    &defer,
                    (final_area, dpi.clone()).zip().value().into_dyn(),
                )
            }),
        )
    }
}
