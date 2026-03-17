// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use guillotiere::euclid::Size2D;

use super::{Desc, base};
use crate::layout::DynLayout;
use crate::reactive::{DynSignal, MutableSignal, SignalZip, const_default, zip_pair};
use crate::{Pixel, PxLimits, PxRect, UnsizedDim};
use std::any::Any;
use std::rc::Rc;

// The root node represents some area on the screen that contains a feather
// layout. Later this will turn into an absolute bounding volume. There can be
// multiple root nodes, each mapping to a different window.
pub trait Prop {
    fn dim(&self) -> DynSignal<Size2D<u32, Pixel>>;
}

crate::gen_dyn_prop!(Prop);

impl Prop for DynSignal<Size2D<u32, Pixel>> {
    fn dim(&self) -> DynSignal<Size2D<u32, Pixel>> {
        self.clone()
    }
}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn base::Empty;
    type Children = Rc<dyn DynLayout<Self::Child>>;
    type Provider = dyn crate::reactive::SignalProvider<Item = (crate::PxRect, crate::RelDim)>;
    type Staging = (
        MutableSignal<crate::RelDim>,
        DynSignal<Self::Children>,
        DynSignal<(DynSignal<PxRect>, Box<dyn Any>)>,
    );

    fn presize(
        props: &Self::Props,
        _: DynSignal<PxLimits>,
        dpi: crate::reactive::MutableSignal<crate::RelDim>,
        child: DynSignal<Self::Children>,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        let dpi2 = dpi.clone();
        let bounds = crate::reactive::const_default().into_dyn();
        let child_presize = child
            .clone()
            .map_ex(move |x| x.presize(bounds.clone(), dpi2.clone()));
        (
            props.dim().map(|x| PxRect::from(x.to_f32())).into(),
            (dpi, child, child_presize.into_dyn()),
        )
    }

    fn size(
        props: &Self::Props,
        _: DynSignal<UnsizedDim>,
        _: DynSignal<crate::PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        let (dpi, child, child_presize) = data;
        let dim = props.dim().map(|d| d.to_f32().cast_unit());
        let bounds = dim
            .clone()
            .map(|x| PxLimits::default().to_bounds(*x))
            .into_dyn();
        let dim = dim.into_dyn();
        let new_data = zip_pair(
            child.clone(),
            child_presize.clone(),
            move |x, (_, child_data)| x.size(dim.clone(), bounds.clone(), child_data.as_ref()),
        );

        (
            props.dim().map(|d| PxRect::from(d.to_f32())).into_dyn(),
            (dpi, child, new_data.into_dyn()),
        )
    }

    fn stage<T: crate::render::Prerender + 'static>(
        props: &Self::Props,
        _: DynSignal<crate::PxPoint>,
        _: DynSignal<PxRect>,
        _: Option<T>,
        defer: Option<super::DeferMachine<Self::Provider>>,
        data: Self::Staging,
    ) -> Rc<crate::rtree::Node> {
        let (dpi, child, child_presize) = data;

        let final_child = zip_pair(child, child_presize, |c, (area, data)| {
            c.stage(const_default().into(), area.clone(), data.as_ref())
        });

        let final_area = props.dim().map(|x| PxRect::from(x.to_f32())).into_dyn();

        super::resolve_defer_machine(
            crate::rtree::Node::new(
                final_area.clone(),
                None,
                Some(
                    final_child
                        .map_ex(|node| imbl::vector![node.clone()])
                        .into_dyn(),
                ),
                Some(Box::new(crate::layout::Concrete::<()> {
                    renderable: None,
                    layer: None,
                    area: final_area.clone(),
                })),
            ),
            &defer,
            (final_area, dpi.clone()).zip().value().into_dyn(),
        )
    }
}
