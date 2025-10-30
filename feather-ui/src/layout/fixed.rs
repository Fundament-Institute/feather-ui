// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::{Concrete, Desc, Layout, Renderable, Staged, base, check_unsized, map_unsized_area};
use crate::{
    PxDim, PxLimits, PxRect, RelDim, UnsizedDim,
    layout::{DynLayout, base::RLimits, check_unsized_dim, zero_unsized},
    reactive::{self, AsSignal, DynSignal, Identity, SignalMap, SignalTupleZip, cond, zip_pair},
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

        let limits = (outer_limits, props.limits(), dpi)
            .zip::<(PxLimits, crate::DLimits, RelDim)>()
            .map(|(outer, limits, dpi)| *outer + limits.resolve(*dpi));

        let myarea =
            zip_pair::<crate::DRect, RelDim, _, _>(props.area(), dpi, |p, dpi| p.resolve(dpi));

        let inner_dim = (myarea, predim, limits)
            .zip::<(crate::URect, crate::UnsizedDim, PxLimits)>()
            .map(|(a, o, l)| super::eval_dim(*a, zero_unsized(*o), *l));

        let child_presize = reactive::map_vec(
            move |child| {
                let child_limit = (
                    inner_dim.clone(),
                    limits,
                    child.as_ref().get_props().rlimits(),
                )
                    .zip::<(crate::UnsizedDim, PxLimits, crate::RelLimits)>()
                    .map(|(inner, l, rlimits)| super::apply_limit(*inner, *l, *rlimits));

                let (stage, _) =
                    child
                        .as_ref()
                        .stage(inner_dim.clone().into(), child_limit.into(), dpi);
                stage
            },
            reactive::Identity,
            children,
        );

        let presize: reactive::Signal<PxRect, reactive::SignalJoinProvider<_, _, _>> =
            reactive::join(reactive::fold_vec(
                |l, r| zip_pair(&l, &r, |u: PxRect, v: PxRect| u.extend(v)).into(),
                child_presize,
                PxRect::zero().to_signal().into(),
            ));

        let presize = presize.map(|x| x.extend(PxRect::zero()));

        // Check if any axis is unsized in a way that requires us to calculate baseline child sizes
        let is_unsized = myarea.map(|x| {
            let (l, r) = check_unsized(*x);
            l || r
        });

        let unsized_area = (myarea, presize, predim, limits)
            .zip::<(crate::URect, PxRect, UnsizedDim, PxLimits)>()
            .map(|(a, p, o, l)| {
                super::limit_area(map_unsized_area(*a, p.dim()) * zero_unsized(*o), *l)
            });

        let sized_area = (myarea, predim, limits)
            .zip::<(crate::URect, UnsizedDim, PxLimits)>()
            .map(|(a, o, l)| super::limit_area(*a * zero_unsized(*o), *l));

        // We gate all our more complex operations behind whether or not this was unsized. If it was unsized, we skip the complex operations.
        let evaluated_area = reactive::cond(is_unsized, unsized_area.into(), sized_area.into());

        (
            evaluated_area.clone().into(),
            Box::new(move |offset, final_dim, final_limits| {
                // We had to evaluate the full area first because our final area calculation can
                // change the dimensions in unsized cases. Thus, we calculate the final
                // inner_area for the children from this evaluated area.
                let inner_dim = evaluated_area.map(|x| x.dim()).into();
                let inner_offset = evaluated_area.map(|x| x.topleft()).into();

                let nodes = reactive::map_vec(
                    |child| {
                        let child_props = child.as_ref().get_props();
                        let child_limit =
                            zip_pair(child_props.rlimits(), inner_dim.clone(), |l, a| {
                                *l * a.dim()
                            });

                        let (_, mut f) = child.as_ref().stage(inner_dim.clone(), child_limit, dpi);

                        Rc::new(f(inner_offset, inner_dim, child_limit))
                    },
                    reactive::Identity,
                    children,
                )
                .into_dyn_signal();

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
                    nodes,
                    Some(Box::new(Concrete::new(renderable))),
                    None,
                )
            }),
        )
    }
}
