// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::{Concrete, Desc, Renderable, base, check_unsized, map_unsized_area};
use crate::{
    PxLimits, PxRect, RelDim, UnsizedDim,
    layout::{DynLayout, zero_unsized},
    reactive::{self, DynSignal, Signal, SignalMap, SignalProvider, SignalTupleZip, zip_pair},
    rtree,
};
use std::rc::Rc;

pub trait Prop: base::Area + base::Anchor + base::Limits + base::ZIndex {}

crate::gen_dyn_prop!(Prop);

pub trait Child: base::RLimits {}

crate::gen_dyn_prop!(Child);

impl Prop for crate::DRect {}
impl Child for crate::DRect {}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Child;
    type Children = imbl::Vector<Rc<dyn DynLayout<Self::Child>>>;

    fn stage<'a>(
        props: &Self::Props,
        predim: DynSignal<crate::UnsizedDim>,
        outer_limits: DynSignal<PxLimits>,
        children: DynSignal<Self::Children>,
        renderable: Option<Rc<dyn Renderable>>,
        dpi: reactive::MutableSignal<RelDim>,
    ) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
        // If we have an unsized outer_area, any sized object with relative dimensions
        // must evaluate to 0 (or to the minimum limited size). An
        // unsized object can never have relative dimensions, as that creates a logic
        // loop - instead it can only have a single relative anchor.
        // If both axes are sized, then all limits are applied as if outer_area was
        // unsized, and children calculations are skipped.
        //
        // If we have an unsized outer_area and an unsized myarea.rel, then limits are
        // applied as if outer_area was unsized, and furthermore,
        // they are reduced by myarea.abs.bottomright(), because that will be added on
        // to the total area later, which will still be subject to size
        // limits, so we must anticipate this when calculating how much size the
        // children will have available to them. This forces limits to be
        // true infinite numbers, so we can subtract finite amounts and still have
        // infinity. We can't use infinity anywhere else, because infinity times
        // zero is NaN, so we cap certain calculations at f32::MAX
        //
        // If outer_area is sized and myarea.rel is zero or nonzero, all limits are
        // applied normally and child calculations are skipped. If outer_area is
        // sized and myarea.rel is unsized, limits are applied normally, but are once
        // again reduced by myarea.abs.bottomright() to account for how the area
        // calculations will interact with the limits later on.

        /*
        let limits = (outer_limits, props.limits(), dpi.clone())
            .zip::<(PxLimits, crate::DLimits, RelDim)>()
            .map(|(outer, limits, dpi)| *outer + limits.resolve(*dpi));

        let myarea = zip_pair::<crate::DRect, RelDim, _, _>(props.area(), dpi.clone(), |p, dpi| {
            p.resolve(dpi)
        });

        let inner_dim = (myarea.clone(), predim.clone(), limits.clone())
            .zip::<(crate::URect, crate::UnsizedDim, PxLimits)>()
            .map(|(a, o, l)| super::eval_dim(*a, zero_unsized(*o), *l));

        let limits2 = limits.clone();
        let dpi2 = dpi.clone();
        let child_presize: Signal<
            imbl::GenericVector<
                Signal<crate::Rect<crate::Pixel>, dyn SignalProvider<crate::Rect<crate::Pixel>>>,
                _,
                _,
            >,
            dyn SignalProvider<
                imbl::GenericVector<
                    Signal<
                        crate::Rect<crate::Pixel>,
                        dyn SignalProvider<crate::Rect<crate::Pixel>>,
                    >,
                    _,
                    _,
                >,
            >,
        > = Signal::into_dyn_signal(reactive::map_vec(
            move |child| todo!(),
            reactive::Identity,
            children.clone(),
        ));

        let presize = reactive::join(&reactive::fold_vec(
            |l, r| {
                zip_pair(l.clone(), r.clone(), |u: PxRect, v: PxRect| u.extend(v)).into_dyn_signal()
            },
            child_presize,
            crate::const_signal(PxRect::zero()).into(),
        ));

        let presize = Signal::into_dyn_signal(presize);

        let presize = presize.map(|x| x.extend(PxRect::zero()));

        // Check if any axis is unsized in a way that requires us to calculate baseline child sizes
        let is_unsized = myarea.clone().map(|x| {
            let (l, r) = check_unsized(*x);
            l || r
        });

        let unsized_area = (myarea.clone(), presize, predim.clone(), limits.clone())
            .zip::<(crate::URect, PxRect, UnsizedDim, PxLimits)>()
            .map(|(a, p, o, l)| {
                super::limit_area(map_unsized_area(*a, p.dim()) * zero_unsized(*o), *l)
            });

        let sized_area = (myarea.clone(), predim.clone(), limits.clone())
            .zip::<(crate::URect, UnsizedDim, PxLimits)>()
            .map(|(a, o, l)| super::limit_area(*a * zero_unsized(*o), *l));

        // We gate all our more complex operations behind whether or not this was unsized. If it was unsized, we skip the complex operations.
        let evaluated_area = reactive::cond(is_unsized, unsized_area.into(), sized_area.into());
        let dpi2 = dpi.clone();

        (
            evaluated_area.clone().into(),
            Box::new(move |offset, final_dim, final_limits| {
                // We had to evaluate the full area first because our final area calculation can
                // change the dimensions in unsized cases. Thus, we calculate the final
                // inner_area for the children from this evaluated area.
                let inner_dim = evaluated_area.map(|x| x.dim()).into_dyn_signal();
                let inner_offset = evaluated_area.map(|x| x.topleft()).into_dyn_signal();

                let nodes = reactive::map_vec(
                    |child: &Rc<dyn DynLayout<dyn Child + 'static> + 'static>| {
                        let child_props = child.get_props();
                        let child_limit =
                            zip_pair(child_props.rlimits(), inner_dim.clone(), |l, a| l * a)
                                .into_dyn_signal();

                        let (_, mut f) = child.as_ref().stage(
                            inner_dim.map(|x| x.cast_unit()).into_dyn_signal(),
                            child_limit,
                            dpi2.clone(),
                        );

                        Rc::new(f(inner_offset, inner_dim, child_limit))
                    },
                    reactive::Identity,
                    children,
                );

                // TODO: It isn't clear if the simple layout should attempt to handle children
                // changing their estimated sizes after the initial estimate. If we were
                // to handle this, we would need to recalculate the unsized
                // axis with the new child results here, and repeat until it stops changing (we
                // find the fixed point). Because the performance implications are
                // unclear, this might need to be relagated to a special layout.

                // Calculate the anchor using the final evaluated dimensions, after all unsized
                // axis and limits are calculated.
                let anchored_area = (evaluated_area, props.anchor(), dpi)
                    .zip::<(PxRect, crate::DPoint, RelDim)>()
                    .map(|(e, a, d)| *e - (a.resolve(*d) * e.dim()));

                rtree::Node::new(
                    anchored_area.into(),
                    Some(props.zindex()),
                    Some(Signal::into(nodes)),
                    Some(Box::new(Concrete::new(renderable))),
                )
            }),
        )*/

        todo!()
    }
}
