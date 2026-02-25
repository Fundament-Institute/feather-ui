// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::rc::Rc;

use crate::layout::{limit_area, resolve_dim};
use crate::reactive::DynDeferSignal;
use crate::render::Prerender;
use crate::{PxDim, PxLimits, Unsizable};
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
    pub inner_limits: DynDeferSignal<PxLimits>,
    pub final_buffer: DynDeferSignal<cosmic_text::Buffer>,
}

fn buffer_eq(a: &cosmic_text::Buffer, b: &cosmic_text::Buffer) -> bool {
    let mut ranges = a.lines.iter();
    let mut lines = b.lines.iter();
    loop {
        match (lines.next(), ranges.next()) {
            (Some(l), Some(r)) => {
                if l.text() != r.text() {
                    return false;
                }
            }
            (None, None) => return true,
            _ => return false,
        }
    }
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
        prelimits: DynSignal<PxLimits>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, super::StageThunk<'a>) {
        let limits = (self.props.limits(), dpi.clone(), prelimits.clone())
            .zip()
            .flatmap(|(limits, dpi, prelimits)| limits.resolve(*dpi) + *prelimits);

        let padding = zip_pair(self.props.padding(), dpi.clone(), |p, dpi| {
            p.as_perimeter(*dpi)
        });
        let myarea = zip_pair(self.props.area(), dpi.clone(), |p, dpi| p.resolve(*dpi));

        let inner_limits = (limits.clone(), myarea.clone(), padding.clone())
            .zip()
            .flatmap(|(limits, area, padding)| {
                let (unsized_x, unsized_y) = area.is_unsized();
                let mut l = *limits;
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

        self.inner_limits
            .resolve(inner_limits.clone().into())
            .map_err(|_| ())
            .expect("Error resolving deferred limits");

        // Now we get the intrinsic size from the first buffer to calculate the unsized_area.
        // TODO: We can skip this dependency if the text isn't unsized
        let unsized_area = (
            self.buffer.clone(),
            myarea.clone(),
            padding.clone(),
            inner_limits.clone(),
        )
            .zip()
            .flatmap(|(buffer, area, padding, limits)| {
                super::limit_area(
                    area.preresolve(
                        PxDim::new(
                            buffer.size().0.unwrap_or_default(),
                            buffer.size().1.unwrap_or_default(),
                        ) + padding.total(),
                    ),
                    *limits,
                )
            });

        let anchor = self.props.anchor();
        let defer = self.machine.clone();
        let prev = self.buffer.clone();
        let wdriver = self.driver.clone();
        let deferred = self.final_buffer.clone();

        let renderable = self.renderable.clone();
        (
            unsized_area.into(),
            Box::new(move |offset, final_dim, final_limits| {
                let unsized_dim = (
                    padding.clone(),
                    myarea.clone(),
                    prev.clone(),
                    final_dim.clone(),
                )
                    .zip()
                    .flatmap(|(padding, area, buffer, dim)| {
                        area.resolve(
                            *dim,
                            PxDim::new(
                                buffer.size().0.unwrap_or_default(),
                                buffer.size().1.unwrap_or_default(),
                            ) + padding.total(),
                        )
                        .dim()
                    });

                let driver = wdriver.clone();
                // We must now do a second text shaping because the relative coordinates could have changed the total dimensions
                let buffer = (
                    prev.clone(),
                    unsized_dim.clone(),
                    limits.clone(),
                    final_limits.clone(),
                )
                    .zip()
                    .flatmap_mut(move |(prev, dim, l1, l2), buffer| {
                        let mut buffer = buffer.unwrap_or_else(|| prev.clone());
                        if let Some(driver) = driver.upgrade() {
                            let mut font_system = driver.font_system.write();
                            crate::text::copy_buffer(&mut buffer, &mut font_system, prev);

                            let limits = *l1 + *l2;
                            let (limitx, limity) = {
                                let max = limits.max();
                                (
                                    max.width.is_finite().then_some(max.width),
                                    max.height.is_finite().then_some(max.height),
                                )
                            };

                            // TODO: IMPLEMENT RECURSIVE LAYOUT
                            let (unsized_x, unsized_y) = (false, false);
                            buffer.set_size(
                                &mut font_system,
                                if unsized_x {
                                    limitx
                                } else {
                                    Some(dim.width.max(0.0))
                                },
                                if unsized_y {
                                    limity
                                } else {
                                    Some(dim.height.max(0.0))
                                },
                            );
                        }

                        buffer
                    });

                deferred
                    .resolve(buffer.clone().into_dyn())
                    .map_err(|_| ())
                    .expect("Failed to set final buffer??");

                let unsized_final = (
                    padding.clone(),
                    myarea.clone(),
                    buffer.clone(),
                    final_dim.clone(),
                )
                    .zip()
                    .flatmap(|(padding, area, buffer, dim)| {
                        area.resolve(
                            *dim,
                            PxDim::new(
                                buffer.size().0.unwrap_or_default(),
                                buffer.size().1.unwrap_or_default(),
                            ) + padding.total(),
                        )
                    });

                // debug_assert!(anchored_area.v.is_finite().all());
                let anchored_area = (
                    unsized_final.clone(),
                    anchor.clone(),
                    dpi.clone(),
                    offset.clone(),
                )
                    .zip()
                    .flatmap(|(area, a, d, o)| *area - (a.resolve(*d) * area.dim()) + *o)
                    .into_dyn();

                super::resolve_defer_machine(
                    rtree::Node::new(
                        anchored_area.clone(),
                        None,
                        Default::default(),
                        Some(Box::new(super::Concrete::new(
                            Some(&renderable),
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
