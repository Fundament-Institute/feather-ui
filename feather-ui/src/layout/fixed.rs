// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::{Concrete, Desc, base};
use crate::layout::DynLayout;
use crate::layout::base::Empty;
use crate::reactive::{self, DynSignal, MutableSignal, SignalDebug, SignalZip, zip_pair};
use crate::render::Prerender;
use crate::{Limited, PxRect, RelDim, Resolve, Unsizable, rtree};
use std::any::Any;
use std::rc::Rc;

pub trait Prop: base::Area + base::Anchor + base::Limits + base::ZIndex {}

crate::gen_dyn_prop!(Prop);

impl Prop for crate::DRect {}

type Staging = (
    DynSignal<crate::URect<crate::Relative>>,
    MutableSignal<RelDim>,
    DynSignal<imbl::Vector<(DynSignal<PxRect>, Rc<dyn DynLayout<dyn Empty>>, Rc<dyn Any>)>>,
);

fn presize(
    props: &dyn Prop,
    bounds: DynSignal<crate::PxLimits>,
    children: DynSignal<imbl::Vector<Rc<dyn DynLayout<dyn Empty>>>>,
    dpi: MutableSignal<RelDim>,
) -> (DynSignal<crate::PxRect>, Staging) {
    let myarea = props.area().resolve(dpi.clone());
    let limits = (props.limits(), bounds, myarea.clone(), dpi.clone())
        .zip()
        .flatmap(|(limits, bounds, area, dpi)| {
            area.to_bounds(limits.resolve(*dpi).preresolve(*bounds))
        });

    let limits2 = limits.clone().into_dyn();
    let dpi2 = dpi.clone();
    let child_tuple = children.map_elements(
        move |child| {
            let (presize, data) = child.as_ref().presize(limits2.clone(), dpi2.clone());
            (presize, child.clone(), Rc::from(data))
        },
        |x| reactive::Identity(x.clone()),
    );

    let child_rects = child_tuple
        .clone()
        .map_elements(|x| x.0.clone(), |x| reactive::Identity(x.1.clone()));

    let child_presize = reactive::join(reactive::fold_vec(
        |l, r| zip_pair(l.clone(), r.clone(), |u, v| u.extend(*v)).into_dyn(),
        child_rects,
        crate::ConstSignal::new(PxRect::zero()).into_dyn(),
    ))
    .map(|x| x.extend(PxRect::zero()).dim());

    // Check if any axis is unsized in a way that requires us to calculate baseline child sizes
    let is_sized = myarea.clone().map(|x| x.is_sized());
    let nosize = reactive::const_new(crate::PxDim::zero());

    // We are careful not to take a dependency on our children's presize if we don't have any unsized axis.
    let intrinsic_size = is_sized.cond(nosize.into_dyn(), child_presize.into_dyn());
    let evaluated_area = myarea.resolve(intrinsic_size);

    let anchored_area = (evaluated_area.clone(), props.anchor(), dpi.clone(), limits)
        .zip()
        .flatmap(|(area, anchor, dpi, l)| {
            area.preresolve().limit(*l).anchored(anchor.resolve(*dpi))
        });

    (
        anchored_area.into(),
        (evaluated_area.into(), dpi, child_tuple.into_dyn()),
    )
}

fn size(
    props: &dyn Prop,
    dim: DynSignal<crate::UnsizedDim>,
    bounds: DynSignal<crate::PxLimits>,
    data: Staging,
) -> (DynSignal<PxRect>, Staging) {
    let (prev_area, dpi, prev_tuple) = data;

    let limits =
        props
            .limits()
            .resolve(dpi.clone())
            .resolve(zip_pair(bounds, dim.clone(), |b, d| b.to_bounds(*d)));

    let child_dim = (prev_area, dim.clone(), limits.clone())
        .zip()
        .flatmap(|(a, dim, l)| a.partial_resolve(*dim).limit(*l));
    let limits2 = limits.clone();

    let child_tuple = prev_tuple.map_elements(
        move |(_, child, prev)| {
            // limits.to_bounds is redundant in a standard fixed-size layout, but necessary because
            // children control how they utilize their own bounds, so we must provide maximally
            // correct bounds even if a normal fixed-size layout doesn't need it.

            let (v, data) = child.size(
                child_dim.clone().into(),
                zip_pair(limits2.clone(), child_dim.clone(), |l, d| l.to_bounds(*d)).into(),
                prev.as_ref(),
            );
            (v, child.clone(), Rc::from(data))
        },
        |x| reactive::Identity(x.1.clone()),
    );

    let child_rects = child_tuple.clone().map_elements(
        |x: &(DynSignal<PxRect>, Rc<dyn DynLayout<dyn Empty>>, Rc<dyn Any>)| x.0.clone(),
        |x| reactive::Identity(x.1.clone()),
    );

    let child_size = reactive::join(reactive::fold_vec(
        |l, r| zip_pair(l.clone(), r.clone(), |u: &PxRect, v: &PxRect| u.extend(*v)).into_dyn(),
        child_rects,
        crate::ConstSignal::new(PxRect::zero()).into_dyn(),
    ))
    .map(|x| x.extend(PxRect::zero()).dim());

    let myarea = props.area().resolve(dpi.clone());
    let is_sized = myarea.clone().map(|x| x.is_sized());

    let nosize = reactive::const_new(crate::PxDim::zero());
    let intrinsic_size = is_sized.cond(nosize.into_dyn(), child_size.into_dyn());

    let evaluated_area = myarea.resolve(intrinsic_size);

    let anchored_area = (
        evaluated_area.clone(),
        dim,
        dpi.clone(),
        props.anchor(),
        limits,
    )
        .zip()
        .flatmap(|(area, dim, dpi, a, l)| {
            area.resolve(dim.zero_unsized())
                .limit(*l)
                .anchored(a.resolve(*dpi))
        });

    (
        anchored_area.into_dyn(),
        (evaluated_area.clone().into(), dpi, child_tuple.into()),
    )
}

fn stage<T: Prerender + 'static>(
    props: &dyn Prop,
    offset: DynSignal<crate::PxPoint>,
    area: DynSignal<PxRect>,
    renderable: Option<T>,
    defer: Option<super::DeferMachine<<dyn Prop as Desc>::Provider>>,
    data: Staging,
    layer: Option<(DynSignal<crate::color::sRGB32>, DynSignal<f32>)>,
) -> Rc<rtree::Node> {
    let (_, dpi, child_tuple) = data;
    let final_area = (area + offset).into_dyn();

    let child_offset = reactive::ConstSignal::new(crate::PxPoint::zero()).into_dyn();
    let nodes = child_tuple.clone().map_elements(
        move |(area, child, data)| child.stage(child_offset.clone(), area.clone(), data.as_ref()),
        |x| reactive::Identity(x.1.clone()),
    );

    super::resolve_defer_machine(
        rtree::Node::new(
            final_area.clone(),
            Some(props.zindex()),
            Some(nodes.into()),
            Some(Box::new(Concrete {
                renderable: renderable.as_ref().map(|x| x.prerender(final_area.clone())),
                area: final_area.clone(),
                layer: if let Some(sig) = &layer {
                    Some(Rc::new(crate::render::compositor::Layer::new(
                        final_area.clone(),
                        final_area.clone(),
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
}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Empty;
    type Children = imbl::Vector<Rc<dyn DynLayout<Self::Child>>>;
    type Provider = dyn crate::reactive::SignalProvider<Item = (PxRect, crate::RelDim)>;
    type Staging = Staging;

    fn presize(
        props: &Self::Props,
        bounds: DynSignal<crate::PxLimits>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
        children: DynSignal<Self::Children>,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        presize(props, bounds, children, dpi)
    }

    fn size(
        props: &Self::Props,
        dim: DynSignal<crate::UnsizedDim>,
        bounds: DynSignal<crate::PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        size(props, dim, bounds, data)
    }

    fn stage<T: Prerender + 'static>(
        props: &Self::Props,
        offset: DynSignal<crate::PxPoint>,
        area: DynSignal<PxRect>,
        renderable: Option<T>,
        defer: Option<super::DeferMachine<Self::Provider>>,
        data: Self::Staging,
    ) -> Rc<rtree::Node> {
        stage(props, offset, area, renderable, defer, data, None)
    }
}

#[derive_where::derive_where(Clone)]
pub struct Layer<T, R: Clone> {
    pub props: Rc<T>,
    pub children: DynSignal<imbl::Vector<Rc<dyn DynLayout<dyn Empty>>>>,
    pub renderable: Option<R>,
    pub layer: Option<(DynSignal<crate::color::sRGB32>, DynSignal<f32>)>,
    pub machine: Option<super::DeferMachine<<dyn Prop as Desc>::Provider>>,
}

#[cfg(feature = "signal-debug")]
impl<T: std::fmt::Debug, R: Clone + std::fmt::Debug> std::fmt::Debug for Layer<T, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layer")
            .field("props", &self.props)
            .field("children", &self.children)
            .field("renderable", &self.renderable)
            .field("layer", &self.layer)
            .field("machine", &self.machine)
            .finish()
    }
}

impl<T: Prop + SignalDebug + 'static, R: Prerender + Clone + SignalDebug + 'static> super::Layout
    for Layer<T, R>
{
    type Props = T;
    type Staging = Staging;

    fn get_props(&self) -> &T {
        &self.props
    }

    fn presize(
        &self,
        bounds: DynSignal<crate::PxLimits>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        presize(&*self.props, bounds, self.children.clone(), dpi)
    }

    fn size(
        &self,
        dim: DynSignal<crate::UnsizedDim>,
        bounds: DynSignal<crate::PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        size(&*self.props, dim, bounds, data)
    }

    fn stage(
        &self,
        offset: DynSignal<crate::PxPoint>,
        area: DynSignal<PxRect>,
        data: Self::Staging,
    ) -> Rc<rtree::Node> {
        stage(
            &*self.props,
            offset,
            area,
            self.renderable.clone(),
            self.machine.clone(),
            data,
            self.layer.clone(),
        )
    }
}
