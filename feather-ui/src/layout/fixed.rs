// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::{Concrete, Desc, base, check_unsized, map_unsized_area};
use crate::{
    PxLimits, PxRect, RelDim,
    layout::{DynLayout, zero_unsized},
    reactive::{self, DynSignal, SignalMap, SignalZip, zip_pair},
    render::Prerender,
    rtree,
};
use std::rc::Rc;

pub trait Prop: base::Area + base::Anchor + base::Limits + base::ZIndex {}

crate::gen_dyn_prop!(Prop);

pub trait Child: base::RLimits {}

crate::gen_dyn_prop!(Child);

impl Prop for crate::DRect {}
impl Child for crate::DRect {}

fn stage<'a, T: Prerender + 'static>(
    props: &dyn Prop,
    predim: DynSignal<crate::PxDim>,
    children: DynSignal<imbl::Vector<Rc<dyn DynLayout<dyn Child>>>>,
    renderable: Option<T>,
    dpi: reactive::MutableSignal<RelDim>,
    layer: Option<(DynSignal<crate::color::sRGB32>, DynSignal<f32>)>,
    defer: Option<super::DeferMachine<<dyn Prop as Desc>::Provider>>,
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

    let limits = zip_pair(props.limits(), dpi.clone(), |limits, dpi| {
        limits.resolve(*dpi)
    });

    let myarea =
        zip_pair::<crate::DRect, RelDim, _, _, _, _>(props.area(), dpi.clone(), |p, dpi| {
            p.resolve(*dpi)
        });

    let inner_dim = (myarea.clone(), predim.clone(), limits.clone())
        .zip()
        .flatmap(|(a, o, l)| super::eval_dim(*a, *o, *l));

    let inner_sized = inner_dim
        .clone()
        .map(|x| zero_unsized(*x))
        .into_dyn_signal();

    let dpi2 = dpi.clone();
    let child_tuple = reactive::map_vec(
        move |child| {
            let rlimits = child.as_ref().get_props().rlimits();

            let (stage, a) = child.as_ref().stage(inner_sized.clone(), dpi2.clone());
            (
                (stage, rlimits.clone(), inner_dim.clone())
                    .zip()
                    .flatmap(|(area, l, inner)| super::limit_area(*area, *l * *inner))
                    .into_dyn_signal(),
                Rc::<super::StageThunk>::from(a),
                rlimits,
            )
        },
        |x| reactive::Identity(x.clone()),
        children,
    )
    .into_dyn_signal();

    let child_presize = reactive::map_vec(
        |x| x.0.clone(),
        |x| reactive::Identity(x.1.clone()),
        child_tuple.clone(),
    );

    let presize = reactive::join(reactive::fold_vec(
        |l, r| {
            zip_pair(l.clone(), r.clone(), |u: &PxRect, v: &PxRect| u.extend(*v)).into_dyn_signal()
        },
        child_presize,
        crate::const_signal(PxRect::zero()).into_dyn_signal(),
    ));

    let presize = presize.map(|x| x.extend(PxRect::zero()));

    // Check if any axis is unsized in a way that requires us to calculate baseline child sizes
    let is_unsized = myarea.clone().map(|x| {
        let (l, r) = check_unsized(*x);
        l || r
    });

    let unsized_area = (
        myarea.clone(),
        presize.clone(),
        predim.clone(),
        limits.clone(),
    )
        .zip()
        .flatmap(|(a, p, o, l)| super::limit_area(map_unsized_area(*a, p.dim()) * *o, *l));

    let sized_area = (myarea.clone(), predim.clone(), limits.clone())
        .zip()
        .flatmap(|(a, o, l)| super::limit_area(*a * *o, *l));

    // We gate all our more complex operations behind whether or not this was unsized. If it was unsized, we skip the complex operations.
    let evaluated_area = reactive::cond(is_unsized.clone(), unsized_area.into(), sized_area.into());
    let dpi2 = dpi.clone();

    let anchor = props.anchor();
    let zindex = props.zindex();

    (
        evaluated_area.into(),
        Box::new(move |offset, final_dim| {
            let unsized_final = (
                myarea.clone(),
                presize.clone(),
                final_dim.clone(),
                limits.clone(),
                offset.clone(),
            )
                .zip()
                .flatmap(|(a, p, dim, l, o)| {
                    super::limit_area(map_unsized_area(*a, p.dim()) * *dim, *l) + *o
                });

            let sized_final = (
                myarea.clone(),
                final_dim.clone(),
                limits.clone(),
                offset.clone(),
            )
                .zip()
                .flatmap(|(a, dim, l, o)| super::limit_area(*a * *dim, *l) + *o);

            // We gate all our more complex operations behind whether or not this was unsized. If it was unsized, we skip the complex operations.
            let final_area =
                reactive::cond(is_unsized.clone(), unsized_final.into(), sized_final.into());

            let child_area = final_area.clone();
            let nodes = reactive::map_vec(
                move |(_, f, rlimits)| {
                    let dim = zip_pair(child_area.clone(), rlimits.clone(), |area, limits| {
                        super::limit_dim_sized(area.dim(), *limits * area.dim())
                    });

                    f(
                        crate::const_signal(crate::PxPoint::zero()).into(),
                        dim.into(),
                    )
                },
                |x| reactive::Identity(x.1.clone()),
                child_tuple.clone(),
            );

            // TODO: It isn't clear if the simple layout should attempt to handle children
            // changing their estimated sizes after the initial estimate. If we were
            // to handle this, we would need to recalculate the unsized
            // axis with the new child results here, and repeat until it stops changing (we
            // find the fixed point). Because the performance implications are
            // unclear, this might need to be relagated to a special layout.

            // Calculate the anchor using the final evaluated dimensions, after all unsized
            // axis and limits are calculated.
            let anchored_area = (final_area.clone(), anchor.clone(), dpi2.clone())
                .zip()
                .flatmap(|(area, anchor, d)| *area - (anchor.resolve(*d) * area.dim()))
                .into_dyn_signal();

            super::resolve_defer_machine(
                rtree::Node::new(
                    anchored_area.clone(),
                    Some(zindex.clone()),
                    Some(nodes.into()),
                    Some(Box::new(Concrete {
                        renderable: renderable
                            .as_ref()
                            .map(|x| x.prerender(anchored_area.clone())),
                        area: anchored_area.clone(),
                        layer: if let Some(sig) = &layer {
                            Some(Rc::new(crate::render::compositor::Layer::new(
                                anchored_area.clone(),
                                anchored_area,
                                sig.0.clone(),
                                sig.1.clone(),
                                false,
                            )))
                        } else {
                            None
                        },
                    })),
                ),
                &defer,
                (final_area, dpi.clone()).zip().value().into_dyn_signal(),
            )
        }),
    )
}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Child;
    type Children = imbl::Vector<Rc<dyn DynLayout<Self::Child>>>;
    type Provider = dyn crate::reactive::SignalProvider<Item = (PxRect, crate::RelDim)>;

    fn stage<'a, T: Prerender + 'static>(
        props: &Self::Props,
        predim: DynSignal<crate::PxDim>,
        children: DynSignal<Self::Children>,
        renderable: Option<T>,
        dpi: reactive::MutableSignal<RelDim>,
        defer: Option<super::DeferMachine<Self::Provider>>,
    ) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
        stage(props, predim, children, renderable, dpi, None, defer)
    }
}

#[derive_where::derive_where(Clone)]
pub struct Layer<T, R: Clone> {
    pub props: Rc<T>,
    pub children: DynSignal<imbl::Vector<Rc<dyn DynLayout<dyn Child>>>>,
    pub renderable: Option<R>,
    pub layer: Option<(DynSignal<crate::color::sRGB32>, DynSignal<f32>)>,
    pub machine: Option<super::DeferMachine<<dyn Prop as Desc>::Provider>>,
}

impl<T: Prop + 'static, R: Prerender + Clone + 'static> super::Layout for Layer<T, R> {
    type Props = T;

    fn get_props(&self) -> &T {
        &self.props
    }
    fn stage<'a>(
        &self,
        outer: DynSignal<crate::PxDim>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, super::StageThunk<'a>) {
        stage(
            self.props.as_ref().into(),
            outer,
            self.children.clone(),
            self.renderable.clone(),
            dpi,
            self.layer.clone(),
            self.machine.clone(),
        )
    }
}
