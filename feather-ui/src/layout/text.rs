// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::cell::RefCell;
use std::rc::Rc;

use derive_where::derive_where;

use crate::{
    PxRect,
    layout::check_unsized_dim,
    reactive::{DynSignal, MutableSignal, SignalMap, SignalTupleZip, zip, zip_pair},
    render, rtree,
};

use super::{Layout, check_unsized, leaf, limit_area};

//#[derive_where::derive_where(Clone)]
#[derive(Clone)]
pub struct Node<T> {
    pub props: Rc<T>,
    pub buffer: MutableSignal<cosmic_text::Buffer>,
    pub renderable: Rc<dyn render::Renderable>,
    //pub realign: bool,
    pub driver: std::sync::Weak<crate::graphics::Driver>,
    pub machine: Option<super::DeferMachine<<dyn leaf::Prop as super::Desc>::Provider>>,
}

impl<T: leaf::Padded> Layout for Node<T> {
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

        let limits = (
            self.props.limits(),
            dpi.clone(),
            myarea.clone(),
            padding.clone(),
        )
            .zip()
            .map(|(limits, dpi, area, padding)| {
                let (unsized_x, unsized_y) = check_unsized(*area);
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

        let mut text_buffer = self.buffer.borrow_mut();
        let driver = self.driver.clone();

        let inner_dim = (myarea.clone(), predim.clone(), limits.clone())
            .zip::<(crate::URect, crate::PxDim, crate::PxLimits)>()
            .map(|(a, o, l)| super::eval_dim(*a, *o, *l));

        //let sized_area = (myarea.clone(), predim.clone(), limits.clone())
        //   .zip::<(crate::URect, crate::PxDim, PxLimits)>()
        //    .map(|(a, o, l)| super::limit_area(*a * *o, *l));

        // Resolve the defered inner_dim and limits


        (self.buffer.clone(), )
        // Now we can operate on self.buffer, as any use of it will trigger a recalculation
        let presize = zip_pair(self.buffer, myarea.clone(), |(_, w, h), area| {
            let (unsized_x, unsized_y) = check_unsized(area);
            let mut prearea = *area;
            let ltrb = prearea.v.as_array_mut();
            if unsized_x {
                ltrb[2] = ltrb[0] + w;
            }
            if unsized_y {
                ltrb[3] = ltrb[1] + h;
            }
            prearea
        });

        let anchor = self.props.anchor();
        let defer = self.machine.clone();

        let unsized_area = (
            myarea.clone(),
            presize.clone(),
            predim.clone(),
            limits.clone(),
        )
            .zip::<(crate::URect, PxRect, crate::PxDim, crate::PxLimits)>()
            .map(|(a, p, o, l)| super::limit_area(super::map_unsized_area(*a, p.dim()) * *o, *l));

        (
            unsized_area.into(),
            Box::new(move |offset, final_dim| {
                let final_area = (
                    padding.clone(),
                    myarea.clone(),
                    offset,
                    final_dim,
                    limits.clone(),
                )
                    .zip()
                    .map(|(padding, a, o, dim, limits)| {
                        super::limit_area(
                            super::map_unsized_area(*a, padding.total()) * *dim,
                            *limits,
                        ) + *o
                    })
                    .into_dyn_signal();

                // debug_assert!(anchored_area.v.is_finite().all());
                let anchored_area = (final_area.clone(), anchor.clone(), dpi.clone())
                    .zip::<(PxRect, crate::DPoint, crate::RelDim)>()
                    .map(|(area, a, d)| *area - (a.resolve(*d) * area.dim()))
                    .into_dyn_signal();

                super::resolve_defer_machine(
                    rtree::Node::new(
                        anchored_area.clone(),
                        None,
                        Default::default(),
                        Some(Box::new(super::Concrete::new(
                            self.renderable.clone(),
                            anchored_area,
                        ))),
                    ),
                    &defer,
                    crate::reactive::zip(final_area, dpi.clone()).into_dyn_signal(),
                )
            }),
        )
    }
}
