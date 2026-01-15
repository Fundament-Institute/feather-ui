// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::base::Empty;
use super::{Concrete, Desc, Layout, base, map_unsized_area};
use crate::layout::DeferMachine;
use crate::reactive::{DynSignal, SignalMap, SignalTupleZip, zip_pair};
use crate::{DRect, PxDim, PxLimits, PxPerimeter, PxRect, RelDim, rtree};
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
        predim: DynSignal<crate::PxDim>,
        _: DynSignal<Self::Children>,
        renderable: Option<T>,
        dpi: crate::reactive::MutableSignal<RelDim>,
        defer: Option<super::DeferMachine<Self::Provider>>,
    ) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
        let limits = zip_pair(props.limits(), dpi.clone(), |limits, dpi| {
            limits.resolve(dpi)
        });

        let myarea = zip_pair(props.area(), dpi.clone(), |p, dpi| p.resolve(dpi));

        let evaluated_area = (myarea.clone(), predim.clone(), limits.clone())
            .zip::<(crate::URect, crate::PxDim, PxLimits)>()
            .map(|(a, dim, l)| super::limit_area(*a * *dim, *l));

        let anchor = props.anchor();
        (
            evaluated_area.into(),
            Box::new(move |offset, final_dim| {
                let final_area = (myarea.clone(), offset, final_dim, limits.clone())
                    .zip()
                    .map(|(a, o, dim, limits)| super::limit_area(*a * *dim, *limits) + *o)
                    .into_dyn_signal();

                let anchored_area = (final_area.clone(), anchor.clone(), dpi.clone())
                    .zip::<(PxRect, crate::DPoint, RelDim)>()
                    .map(|(e, a, d)| *e - (a.resolve(*d) * e.dim()))
                    .into_dyn_signal();

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
                    crate::reactive::zip(anchored_area, dpi.clone()).into_dyn_signal(),
                )
            }),
        )
    }
}

/// A sized leaf is one with inherent size, like an image. This is used to
/// preserve aspect ratio when encounting an unsized axis. This must be provided
/// in pixels.
#[derive_where::derive_where(Clone)]
pub struct Sized<T, R: Clone> {
    pub props: Rc<T>,
    pub size: DynSignal<crate::PxDim>,
    pub renderable: Option<R>,
    pub machine: Option<DeferMachine<<dyn Prop as Desc>::Provider>>,
}

#[inline]
fn calc_sized_area(padding: PxPerimeter, area: crate::URect, size: PxDim, outer: PxDim) -> PxRect {
    let (unsized_x, unsized_y) = super::check_unsized(area);
    let aspect_ratio = size.width / size.height; // Will be NAN if both are 0, which disables any attempt to preserve aspect ratio
    match (unsized_x, unsized_y, aspect_ratio.is_finite()) {
        (true, false, false) => {
            let mut presize = map_unsized_area(area, PxDim::zero()) * outer;
            let adjust = presize.dim().height * aspect_ratio;
            let v = presize.v.as_array_mut();
            v[2] += adjust;
            presize
        }
        (false, true, false) => {
            let mut presize = map_unsized_area(area, PxDim::zero()) * outer;
            // Be careful, the aspect ratio here is being divided instead of multiplied
            let adjust = presize.dim().width / aspect_ratio;
            let v = presize.v.as_array_mut();
            v[3] += adjust;
            presize
        }
        _ => map_unsized_area(area, size + padding.topleft() + padding.bottomright()) * outer,
    }
}

impl<T: Padded, R: crate::render::Prerender + Clone + 'static> Layout for Sized<T, R> {
    type Props = T;

    fn get_props(&self) -> &T {
        &self.props
    }
    fn stage<'a>(
        &self,
        predim: DynSignal<crate::PxDim>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
        //_: crate::reactive::ConstSignal<std::sync::Weak<Driver>>,
    ) -> (DynSignal<PxRect>, super::StageThunk<'a>) {
        let limits = zip_pair(self.props.limits(), dpi.clone(), |limits, dpi| {
            limits.resolve(dpi)
        });

        let padding = zip_pair(self.props.padding(), dpi.clone(), |p, dpi| {
            p.as_perimeter(dpi)
        });
        let area = zip_pair(self.props.area(), dpi.clone(), |p, dpi| p.resolve(dpi));

        // The way we handle unsized here is different from how we normally handle it.
        // If both axes are unsized, we simply set the area to the internal
        // size. If only one axis is unsized, we stretch it to maintain an aspect
        // ratio relative to the size of the other axis.
        let mapped_area = (padding.clone(), area.clone(), self.size.clone(), predim)
            .zip()
            .map(|(padding, area, size, outer)| calc_sized_area(*padding, *area, *size, *outer));

        let size = self.size.clone();
        let evaluated_area = zip_pair(mapped_area, limits.clone(), |a, l| super::limit_area(a, l));
        let renderable = self.renderable.clone();
        let anchor = self.props.anchor();
        let defer = self.machine.clone();

        (
            evaluated_area.into(),
            Box::new(move |offset, final_dim| {
                let final_area = (
                    padding.clone(),
                    area.clone(),
                    size.clone(),
                    offset,
                    final_dim,
                    limits.clone(),
                )
                    .zip()
                    .map(move |(padding, area, size, o, outer, limits)| {
                        calc_sized_area(
                            *padding,
                            *area,
                            *size,
                            super::limit_dim_sized(*outer, *limits),
                        ) + *o
                    })
                    .into_dyn_signal();

                // debug_assert!(anchored_area.v.is_finite().all());
                let anchored_area = (final_area.clone(), anchor.clone(), dpi.clone())
                    .zip::<(PxRect, crate::DPoint, RelDim)>()
                    .map(|(area, a, d)| *area - (a.resolve(*d) * area.dim()))
                    .into_dyn_signal();

                super::resolve_defer_machine(
                    rtree::Node::new(
                        anchored_area.clone(),
                        None,
                        Default::default(),
                        Some(Box::new(Concrete::new(renderable.as_ref(), anchored_area))),
                    ),
                    &defer,
                    crate::reactive::zip(final_area, dpi.clone()).into_dyn_signal(),
                )
            }),
        )
    }
}
