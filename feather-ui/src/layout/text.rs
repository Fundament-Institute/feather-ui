// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::rc::Rc;

use crate::Unsizable;
use crate::layout::resolve_dim;
use crate::reactive::DynDeferSignal;
use crate::render::Prerender;
use crate::{
    PxRect,
    reactive::{DynSignal, Signal, SignalProvider, SignalZip, zip_pair},
    rtree,
};

use super::{Layout, leaf};

//#[derive_where::derive_where(Clone)]
#[derive(Clone)]
pub struct Node<
    T,
    R: Clone + 'static,
    P1: SignalProvider<Item = cosmic_text::Buffer> + ?Sized = dyn SignalProvider<
        Item = cosmic_text::Buffer,
    >,
> {
    pub props: Rc<T>,
    pub buffer: Signal<P1>,
    pub renderable: R,
    //pub realign: bool,
    pub driver: std::sync::Weak<crate::graphics::Driver>,
    pub machine: Option<super::DeferMachine<<dyn leaf::Prop as super::Desc>::Provider>>,
    pub inner_dim: DynDeferSignal<crate::UnsizedDim>,
    pub inner_limits: DynDeferSignal<crate::PxLimits>,
}

impl<
    T: leaf::Padded,
    R: Prerender + Clone + 'static,
    P1: SignalProvider<Item = cosmic_text::Buffer> + ?Sized + 'static,
> Layout for Node<T, R, P1>
{
    type Props = T;

    fn get_props(&self) -> &T {
        &self.props
    }
    fn stage<'a>(
        &self,
        predim: DynSignal<crate::PxDim>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, super::StageThunk<'a>) {
        let myarea = zip_pair(self.props.area(), dpi.clone(), |p, dpi| p.resolve(*dpi));
        let padding = zip_pair(self.props.padding(), dpi.clone(), |p, dpi| {
            p.as_perimeter(*dpi)
        });

        let inner_limits = (
            self.props.limits(),
            dpi.clone(),
            myarea.clone(),
            padding.clone(),
        )
            .zip()
            .flatmap(|(limits, dpi, area, padding)| {
                let (unsized_x, unsized_y) = area.is_unsized();
                let mut l = limits.resolve(*dpi);
                let minmax = l.v.as_array_mut();
                let allpadding =
                    area.bottomright().abs().to_vector().to_size().cast_unit() + padding.total();
                if unsized_x {
                    minmax[2] -= allpadding.width;
                    minmax[0] -= allpadding.width;
                }
                if unsized_y {
                    minmax[3] -= allpadding.height;
                    minmax[1] -= allpadding.height;
                }

                l
            });

        let inner_dim = (myarea.clone(), predim.clone(), inner_limits.clone())
            .zip()
            .flatmap(|(a, o, l)| super::eval_dim(*a, *o, *l));

        //let sized_area = (myarea.clone(), predim.clone(), limits.clone())
        //   .zip::<(crate::URect, crate::PxDim, PxLimits)>()
        //    .map(|(a, o, l)| super::limit_area(*a * *o, *l));

        // Resolve the defered inner_dim and limits
        self.inner_dim
            .resolve(inner_dim.clone().into())
            .expect("Unexpected defer failure");
        self.inner_limits
            .resolve(inner_limits.clone().into())
            .expect("Unexpected defer failure");

        // Now we can operate on self.buffer, as any use of it will trigger a recalculation
        let presize = (self.buffer.clone(), inner_dim.clone(), padding.clone())
            .zip()
            .flatmap(|(buffer, dim, padding)| {
                resolve_dim(
                    *dim,
                    padding.total()
                        + crate::PxDim::new(
                            buffer.size().0.unwrap_or_default(),
                            buffer.size().1.unwrap_or_default(),
                        ),
                )
            });

        let anchor = self.props.anchor();
        let defer = self.machine.clone();

        let unsized_area = (
            myarea.clone(),
            presize.clone(),
            predim.clone(),
            inner_limits.clone(),
        )
            .zip()
            .flatmap(|(a, p, o, l)| super::limit_area(a.resolve(*o, *p), *l));

        let renderable = self.renderable.clone();
        (
            unsized_area.into(),
            Box::new(move |offset, final_dim| {
                let unsized_final = (
                    padding.clone(),
                    myarea.clone(),
                    presize.clone(),
                    final_dim.clone(),
                    inner_limits.clone(),
                    offset.clone(),
                )
                    .zip()
                    .flatmap(|(padding, a, p, dim, l, o)| {
                        super::limit_area(a.resolve(*dim, *p + padding.total()), *l) + *o
                    });

                // debug_assert!(anchored_area.v.is_finite().all());
                let anchored_area = (unsized_final.clone(), anchor.clone(), dpi.clone())
                    .zip()
                    .flatmap(|(area, a, d)| *area - (a.resolve(*d) * area.dim()))
                    .into_dyn_signal();

                super::resolve_defer_machine(
                    rtree::Node::new(
                        anchored_area.clone(),
                        None,
                        Default::default(),
                        Some(Box::new(super::Concrete::new(
                            Some(&renderable),
                            anchored_area,
                        ))),
                    ),
                    &defer,
                    (unsized_final, dpi.clone()).zip().value().into_dyn_signal(),
                )
            }),
        )
    }
}
