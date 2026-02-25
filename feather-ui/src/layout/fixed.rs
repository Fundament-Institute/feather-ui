// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::{Concrete, Desc, base};
use crate::{
    PxRect, RelDim, Unsizable,
    layout::DynLayout,
    reactive::{self, DynSignal, SignalZip, zip_pair},
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
    prelimits: DynSignal<crate::PxLimits>,
    children: DynSignal<imbl::Vector<Rc<dyn DynLayout<dyn Child>>>>,
    renderable: Option<T>,
    dpi: reactive::MutableSignal<RelDim>,
    layer: Option<(DynSignal<crate::color::sRGB32>, DynSignal<f32>)>,
    defer: Option<super::DeferMachine<<dyn Prop as Desc>::Provider>>,
) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
    let limits = (props.limits(), dpi.clone(), prelimits)
        .zip()
        .flatmap(|(limits, dpi, bounds)| limits.resolve(*dpi) + *bounds);

    let myarea =
        zip_pair::<crate::DRect, RelDim, _, _, _, _>(props.area(), dpi.clone(), |p, dpi| {
            p.resolve(*dpi)
        });

    let predim = (myarea.clone(), limits.clone())
        .zip()
        .flatmap(|(a, l)| 
        // d
        super::intrinsic_dim(*a, *l));

    let dpi2 = dpi.clone();
    let child_tuple = children.map_elements(
        move |child| {
            let rlimits = child.as_ref().get_props().rlimits();
            let inner_bounds = rlimits.clone() * predim.clone();
            let (stage, a) = child.as_ref().stage(inner_bounds.into(), dpi2.clone());
            (stage, Rc::<super::StageThunk>::from(a), rlimits)
        },
        |x| reactive::Identity(x.clone()),
    );

    let child_presize = child_tuple
        .clone()
        .map_elements(|x| x.0.clone(), |x| reactive::Identity(x.1.clone()));

    let intrinsic_size = reactive::join(reactive::fold_vec(
        |l, r| zip_pair(l.clone(), r.clone(), |u: &PxRect, v: &PxRect| u.extend(*v)).into_dyn(),
        child_presize,
        crate::ConstSignal::new(PxRect::zero()).into_dyn(),
    ));

    let presize = intrinsic_size.map(|x| x.extend(PxRect::zero()));

    // Check if any axis is unsized in a way that requires us to calculate baseline child sizes
    let is_unsized = myarea.clone().map(|x| {
        let (l, r) = x.is_unsized();
        l || r
    });

    let unsized_area = (myarea.clone(), presize.clone(), limits.clone())
        .zip()
        .flatmap(|(a, p, l)| super::limit_area(a.preresolve(p.dim()), *l));

    // This is used if myarea has no unsized components, which makes it independent of the child area calculations.
    let sized_area = (myarea.clone(), limits.clone())
        .zip()
        .flatmap(|(a, l)| super::limit_area(a.preresolve_sized(), *l));

    // We are careful not to take a dependency on our children's presize if we don't have any unsized axis.
    let evaluated_area = is_unsized
        .clone()
        .cond(unsized_area.into(), sized_area.into());
    let dpi2 = dpi.clone();

    let anchor = props.anchor();
    let zindex = props.zindex();

    (
        evaluated_area.into(),
        Box::new(move |offset, final_dim, final_limits| {
            // We always calculate the final area using our previously calculated total intrinsic size (the presize). Any differences
            // between this intrinsic size and our final size should be the result of relative coordinates, which we ignore to avoid
            // cycles (although there are still a few cases we could theoretically evaluate via a fixed-point, see below).
            let unsized_final = (
                myarea.clone(),
                presize.clone(),
                final_dim.clone(),
                limits.clone(),
                final_limits.clone(),
                offset.clone(),
            )
                .zip()
                .flatmap(|(a, p, dim, l, l2, o)| {
                    super::limit_area(a.resolve(*dim, p.dim()), *l + *l2) + *o
                });

            let sized_final = (
                myarea.clone(),
                final_dim.clone(),
                limits.clone(),
                final_limits.clone(),
                offset.clone(),
            )
                .zip()
                .flatmap(|(a, dim, l, l2, o)| {
                    super::limit_area(a.resolve_sized(*dim), *l + *l2) + *o
                });

            let final_area = is_unsized
                .clone()
                .cond(unsized_final.into(), sized_final.into());

            let child_dim = final_area.clone().map(|a| a.dim());
            let parent_limits = limits.clone();
            let nodes = child_tuple.clone().map_elements(
                move |(_, f, rlimits)| {
                    let limits = (child_dim.clone(), rlimits.clone(), parent_limits.clone())
                        .zip()
                        .flatmap(|(dim, l1, l2)| (*l1 * *dim) + *l2);

                    f(
                        reactive::ConstSignal::new(crate::PxPoint::zero()).into(),
                        child_dim.clone().into(),
                        limits.into(),
                    )
                },
                |x| reactive::Identity(x.1.clone()),
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
                .into_dyn();

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
                (final_area, dpi.clone()).zip().value().into_dyn(),
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
        prelimits: DynSignal<crate::PxLimits>,
        children: DynSignal<Self::Children>,
        renderable: Option<T>,
        dpi: reactive::MutableSignal<RelDim>,
        defer: Option<super::DeferMachine<Self::Provider>>,
    ) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
        stage(props, prelimits, children, renderable, dpi, None, defer)
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
        prelimits: DynSignal<crate::PxLimits>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, super::StageThunk<'a>) {
        stage(
            self.props.as_ref().into(),
            prelimits,
            self.children.clone(),
            self.renderable.clone(),
            dpi,
            self.layer.clone(),
            self.machine.clone(),
        )
    }
}
