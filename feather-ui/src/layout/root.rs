// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use guillotiere::euclid::Size2D;

use super::{Desc, Renderable, base};
use crate::{
    Pixel, PxRect, UnsizedDim,
    layout::DynLayout,
    reactive::{self, DynSignal, SignalMap, const_signal},
};
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

    fn stage<'a>(
        props: &Self::Props,
        _: DynSignal<UnsizedDim>,
        _: DynSignal<crate::PxLimits>,
        child: DynSignal<Self::Children>,
        _: Option<Rc<dyn Renderable>>,
        dpi: reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
        let dim: DynSignal<UnsizedDim> = props.dim().clone().map(|d| d.to_f32().cast_unit()).into();
        let sized = props.dim().map(|d| d.to_f32()).into_dyn_signal();

        (
            props.dim().map(|x| PxRect::from(x.to_f32())).into(),
            Box::new(move |_, _, _| {
                let dim = dim.clone();
                let dpi = dpi.clone();
                let sized = sized.clone();
                let final_area = sized.clone().map(|d| PxRect::from(*d)).into_dyn_signal();

                let presize = child.clone().map(move |c| {
                    let (_, mut f) = c.stage(
                        dim.clone(),
                        const_signal(crate::PxLimits::default()).into(),
                        dpi.clone(),
                    );

                    Rc::new(f(
                        const_signal(crate::PxPoint::default()).into(),
                        sized.clone(),
                        const_signal(crate::PxLimits::default()).into(),
                    ))
                });

                crate::rtree::Node::new(
                    final_area.clone(),
                    None,
                    Some(
                        presize
                            .clone()
                            .map(|node| imbl::vector![node.clone()])
                            .into_dyn_signal(),
                    ),
                    Some(Box::new(crate::layout::Concrete::new(None))),
                )
            }),
        )
    }
}
