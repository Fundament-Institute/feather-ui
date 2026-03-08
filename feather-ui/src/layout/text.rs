// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::rc::Rc;

use crate::layout::resolve_dim;
use crate::reactive::{DynDeferSignal, DynSignal, MutableSignal, SignalDebug, SignalZip, zip_pair};
use crate::render::Prerender;
use crate::{Limited, PxDim, PxLimits, PxRect, Resolve, Unsizable, rtree};

use super::{Layout, leaf};

#[derive(Clone, Debug)]
pub struct Node<T, R: Clone + 'static> {
    pub props: Rc<T>,
    pub buffer: DynSignal<cosmic_text::Buffer>,
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

impl<T: leaf::Padded + SignalDebug, R: Prerender + Clone + SignalDebug + 'static> Layout
    for Node<T, R>
{
    type Props = T;
    type Staging = (
        DynSignal<cosmic_text::Buffer>,
        MutableSignal<crate::RelDim>,
        DynSignal<PxDim>,
    );

    fn get_props(&self) -> &T {
        &self.props
    }

    fn presize(&self, dpi: MutableSignal<crate::RelDim>) -> (DynSignal<PxRect>, Self::Staging) {
        let limits = (self.props.limits(), dpi.clone())
            .zip()
            .flatmap(|(limits, dpi)| limits.resolve(*dpi));

        let padding = zip_pair(self.props.padding(), dpi.clone(), |p, dpi| p.resolve(*dpi));
        let myarea = zip_pair(self.props.area(), dpi.clone(), |p, dpi| p.resolve(*dpi));

        let inner_limits = (limits.clone(), myarea.clone(), padding.clone())
            .zip()
            .flatmap(|(limits, area, padding)| {
                let (unsized_x, unsized_y) = area.is_unsized();
                let mut l = *limits;
                let minmax = l.v.as_array_mut();
                let allpadding =
                    padding.total() + area.bottomright().abs().to_vector().to_size().cast_unit();
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
            .set(inner_limits.clone().into())
            .map_err(|_| ())
            .expect("Error resolving deferred limits");

        // This signal assumes the text is unsized and calculates the true width/height from the layout runs,
        // adjusted based on the inner_limits.
        let buffer_dim = zip_pair(
            self.buffer.clone(),
            inner_limits.clone(),
            |buffer, limits| {
                let mut h = 0.0;
                let mut w: f32 = 0.0;

                for run in buffer.layout_runs() {
                    w = w.max(run.line_w);
                    h += run.line_height;
                }

                // Apply adjusted limits to inner size calculation
                w = w.max(limits.min().width).min(limits.max().width);
                h = h.max(limits.min().height).min(limits.max().height);

                PxDim::new(w, h)
            },
        );

        // Now we get the intrinsic size from the first buffer to calculate the unsized_area.
        // TODO: It's important to skip this and buffer_size if the area isn't actually unsized.
        let unsized_area = (
            buffer_dim.clone(),
            myarea.clone(),
            padding.clone(),
            inner_limits.clone(),
        )
            .zip()
            .flatmap(|(buffer_size, area, padding, limits)| {
                area.resolve(*buffer_size + padding.total())
                    .preresolve()
                    .limit(*limits)
            });

        let sized_area = (myarea.clone(), inner_limits.clone())
            .zip()
            .flatmap(|(area, limits)| unsafe { area.into_sized() }.preresolve().limit(*limits));

        // Check if any axis is unsized in a way that requires us to calculate baseline child sizes
        let is_sized = myarea.clone().map(|x| x.is_sized());
        let evaluated_area = is_sized
            .clone()
            .cond(sized_area.into(), unsized_area.into());

        (
            evaluated_area.into(),
            (self.buffer.clone(), dpi, buffer_dim.into()),
        )
    }

    fn size(
        &self,
        dim: DynSignal<crate::UnsizedDim>,
        limits: DynSignal<PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        let (prev, dpi, prev_dim) = data;

        let limits = (self.props.limits(), limits, dpi.clone())
            .zip()
            .flatmap(|(l1, l2, dpi)| l1.resolve(*dpi) + l2);

        let wdriver = self.driver.clone();
        let wdriver2 = self.driver.clone();

        let padding = zip_pair(self.props.padding(), dpi.clone(), |p, dpi| p.resolve(*dpi));
        let myarea = zip_pair(self.props.area(), dpi.clone(), |p, dpi| p.resolve(*dpi));

        // If we are unsized, we do a second text shaping here
        let buffer = (
            prev.clone(),
            prev_dim,
            limits.clone(),
            padding.clone(),
            myarea.clone(),
            dim.clone(),
        )
            .zip()
            .flatmap_mut(move |(prev, bufsize, limits, padding, area, dim), buffer| {
                let mut buffer = buffer.unwrap_or_else(|| prev.clone());
                if let Some(driver) = wdriver2.upgrade() {
                    let mut font_system = driver.font_system.write();
                    crate::text::copy_buffer(&mut buffer, &mut font_system, prev);

                    let (limitx, limity) = {
                        let max = limits.max();
                        (
                            max.width.is_finite().then_some(max.width),
                            max.height.is_finite().then_some(max.height),
                        )
                    };

                    let mydim = resolve_dim(*dim, *area, *bufsize + padding.total());

                    let (unsized_x, unsized_y) = mydim.is_unsized();
                    buffer.set_size(
                        &mut font_system,
                        if unsized_x {
                            limitx
                        } else {
                            Some(mydim.width.max(0.0))
                        },
                        if unsized_y {
                            limity
                        } else {
                            Some(mydim.height.max(0.0))
                        },
                    );
                }

                buffer
            });

        let buffer_dim = zip_pair(buffer.clone(), limits.clone(), |buffer, limits| {
            let mut h = 0.0;
            let mut w: f32 = 0.0;

            for run in buffer.layout_runs() {
                w = w.max(run.line_w);
                h += run.line_height;
            }

            // Apply adjusted limits to inner size calculation
            w = w.max(limits.min().width).min(limits.max().width);
            h = h.max(limits.min().height).min(limits.max().height);

            PxDim::new(w, h)
        });

        let unsized_area = (
            buffer_dim.clone(),
            myarea.clone(),
            padding.clone(),
            limits.clone(),
            dim.clone(),
        )
            .zip()
            .flatmap(|(buffer_size, area, padding, limits, d)| {
                area.resolve(*buffer_size + padding.total())
                    .resolve(d.zero_unsized())
                    .limit(*limits)
            });

        let sized_area =
            (myarea.clone(), limits.clone(), dim.clone())
                .zip()
                .flatmap(|(area, limits, d)| {
                    unsafe { area.into_sized() }
                        .resolve(d.zero_unsized())
                        .limit(*limits)
                });

        // Check if any axis is unsized in a way that requires us to calculate baseline child sizes
        let is_sized = myarea.clone().map(|x| x.is_sized());

        let evaluated_area = is_sized
            .clone()
            .cond(sized_area.clone().into(), unsized_area.into());

        let buffer_sized = (prev, sized_area)
            .zip()
            .flatmap_mut(move |(prev, area), buffer| {
                let mut buffer = buffer.unwrap_or_else(|| prev.clone());
                if let Some(driver) = wdriver.upgrade() {
                    let mut font_system = driver.font_system.write();
                    crate::text::copy_buffer(&mut buffer, &mut font_system, prev);

                    let mydim = area.dim().max(crate::PxDim::zero());

                    // TODO: Add subtract padding?
                    buffer.set_size(&mut font_system, Some(mydim.width), Some(mydim.height));
                }

                buffer
            });

        let anchored_area = (evaluated_area.clone(), self.props.anchor(), dpi.clone())
            .zip()
            .flatmap(|(area, a, d)| *area - (a.resolve(*d) * area.dim().max(crate::PxDim::zero())))
            .into_dyn();

        (
            anchored_area.into(),
            (
                is_sized
                    .clone()
                    .cond(buffer_sized.into_dyn(), buffer.into_dyn())
                    .into(),
                dpi,
                buffer_dim.into(),
            ),
        )
    }

    fn stage(
        &self,
        offset: DynSignal<crate::PxPoint>,
        area: DynSignal<PxRect>,
        data: Self::Staging,
    ) -> Rc<rtree::Node> {
        let (prev, dpi, _) = data;

        //let final_area = (area + offset).into_dyn();
        let final_area = zip_pair(
            area,
            offset,
            // delete me
            |a, o| *a + *o,
        )
        .into_dyn();

        self.final_buffer
            .set(prev)
            .map_err(|_| ())
            .expect("Failed to set final buffer??");

        super::resolve_defer_machine(
            rtree::Node::new(
                final_area.clone(),
                None,
                Default::default(),
                Some(Box::new(super::Concrete::new(
                    Some(&self.renderable),
                    final_area.clone(),
                ))),
            ),
            &self.machine,
            (final_area, dpi).zip().value().into_dyn(),
        )
    }
}
