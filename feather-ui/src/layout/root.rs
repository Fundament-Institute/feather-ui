// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use guillotiere::euclid::Size2D;

use super::{Desc, base};
use crate::{
    Pixel, PxRect,
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

    fn stage<'a, T>(
        props: &Self::Props,
        _: DynSignal<crate::PxDim>,
        child: DynSignal<Self::Children>,
        _: Option<T>,
        dpi: reactive::MutableSignal<crate::RelDim>,
    ) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
        let dim: DynSignal<crate::PxDim> = props.dim().clone().map(|d| d.to_f32()).into();
        let sized = props.dim().map(|d| d.to_f32()).into_dyn_signal();

        (
            props.dim().map(|x| PxRect::from(x.to_f32())).into(),
            Box::new(move |_, _| {
                let dim = dim.clone();
                let dpi = dpi.clone();
                let sized = sized.clone();
                let final_area = sized.clone().map(|d| PxRect::from(*d)).into_dyn_signal();

                let presize = child.clone().map(move |c| {
                    let (_, f) = c.stage(dim.clone(), dpi.clone());

                    Rc::new(f(
                        const_signal(crate::PxPoint::default()).into(),
                        sized.clone(),
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
                    Some(Box::new(crate::layout::Concrete::<()> {
                        renderable: None,
                        layer: None,
                        area: final_area.clone(),
                    })),
                )
            }),
        )
    }
}
