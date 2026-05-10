// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::rc::Rc;

use crate::reactive::{
    DynDeferSignal, DynSignal, MutableSignal, SignalDebug, SignalZip, const_default, zip_pair,
};
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
        DynSignal<PxLimits>,
    );

    fn get_props(&self) -> &T {
        &self.props
    }

    fn presize(
        &self,
        bounds: DynSignal<PxLimits>,
        dpi: MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        let myarea = self.props.area().resolve(dpi.clone());
        let limits = (self.props.limits(), dpi.clone(), bounds, myarea.clone())
            .zip()
            .flatmap(|(limits, dpi, bounds, area)| {
                area.to_bounds(limits.resolve(*dpi).preresolve(*bounds))
            });
        let padding = self.props.padding().resolve(dpi.clone());

        let inner_limits = (limits.clone(), myarea.clone(), padding.clone())
            .zip()
            .flatmap(|(limits, area, padding)| {
                let (unsized_x, unsized_y) = area.is_unsized();
                let mut l = *limits;
                let minmax = l.v.as_mut_array();
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
                let mut w: f32 = buffer.size().0.unwrap_or(0.0);

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

        // Now we get the intrinsic size from the first buffer, but only if area is unsized.
        let is_sized = myarea.clone().map(|x| x.is_sized());
        let intrinsic_size = is_sized.cond(
            const_default().into_dyn(),
            zip_pair(buffer_dim.clone(), padding.clone(), |b, p| *b + p.total()).into_dyn(),
        );

        let evaluated_area = (intrinsic_size.clone(), myarea.clone(), inner_limits.clone())
            .zip()
            .flatmap(|(size, area, limits)| area.resolve(*size).preresolve().limit(*limits));

        let anchored_area = (evaluated_area.clone(), self.props.anchor(), dpi.clone())
            .zip()
            .flatmap(|(area, anchor, dpi)| area.anchored(anchor.resolve(*dpi)));

        (
            anchored_area.into(),
            (self.buffer.clone(), dpi, inner_limits.into()),
        )
    }

    fn size(
        &self,
        dim: DynSignal<crate::UnsizedDim>,
        bounds: DynSignal<crate::PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        let (prev, dpi, inner_limits) = data;

        let _limits = self.props.limits().resolve(dpi.clone()).resolve(zip_pair(
            bounds,
            dim.clone(),
            |b, d| b.to_bounds(*d),
        ));

        let wdriver = self.driver.clone();

        let padding = self.props.padding().resolve(dpi.clone());
        let myarea = self.props.area().resolve(dpi.clone());

        // If we are unsized, we do a second text shaping here
        let buffer = (
            prev.clone(),
            inner_limits.clone(),
            myarea.clone(),
            dim.clone(),
        )
            .zip()
            .flatmap_mut(move |(prev, limits, area, dim), buffer| {
                let mut buffer = buffer.unwrap_or_else(|| prev.clone());
                if let Some(driver) = wdriver.upgrade() {
                    let mut font_system = driver.font_system.write();
                    crate::text::copy_buffer(&mut buffer, &mut font_system, prev);

                    let (limitx, limity) = {
                        let max = limits.max();
                        (
                            max.width.is_finite().then_some(max.width),
                            max.height.is_finite().then_some(max.height),
                        )
                    };

                    let mydim = area.skip_resolve(*dim);

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

        let buffer_dim = zip_pair(buffer.clone(), inner_limits.clone(), |buffer, limits| {
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

        let is_sized = myarea.clone().map(|x| x.is_sized());
        let intrinsic_size = is_sized.cond(
            const_default().into_dyn(),
            zip_pair(buffer_dim.clone(), padding.clone(), |b, p| *b + p.total()).into_dyn(),
        );

        let evaluated_area = (
            intrinsic_size.clone(),
            myarea.clone(),
            inner_limits.clone(),
            dim.clone(),
        )
            .zip()
            .flatmap(|(size, area, limits, d)| {
                area.resolve(*size).resolve(d.zero_unsized()).limit(*limits)
            });

        let anchored_area = (evaluated_area.clone(), self.props.anchor(), dpi.clone())
            .zip()
            .flatmap(|(area, a, d)| area.anchored(a.resolve(*d)))
            .into_dyn();

        (anchored_area, (buffer.into_dyn(), dpi, inner_limits))
    }

    fn stage(
        &self,
        offset: DynSignal<crate::PxPoint>,
        area: DynSignal<PxRect>,
        data: Self::Staging,
    ) -> Rc<rtree::Node> {
        let (prev, dpi, _) = data;

        let final_area = (area + offset).into_dyn();

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
