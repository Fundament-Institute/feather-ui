// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use guillotiere::euclid::Size2D;

use super::{Desc, base};
use crate::{
    Pixel, PxLimits, PxRect,
    layout::DynLayout,
    reactive::{self, ConstSignal, DynSignal, SignalZip, const_default},
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
    type Provider = dyn crate::reactive::SignalProvider<Item = (crate::PxRect, crate::RelDim)>;

    fn stage<'a, T>(
        props: &Self::Props,
        _: DynSignal<PxLimits>,
        child: DynSignal<Self::Children>,
        _: Option<T>,
        dpi: reactive::MutableSignal<crate::RelDim>,
        defer: Option<super::DeferMachine<Self::Provider>>,
    ) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
        let sized = props.dim().map(|d| d.to_f32()).into_dyn();
        let no_limit = const_default().into_dyn();
        (
            props.dim().map(|x| PxRect::from(x.to_f32())).into(),
            Box::new(move |_, _, _| {
                let dpi2 = dpi.clone();
                let sized = sized.clone();
                let final_area = sized.clone().map(|d| PxRect::from(*d)).into_dyn();
                let no_limit = no_limit.clone();

                let presize = child.clone().map_ex(move |c| {
                    let (_, f) = c.stage(no_limit.clone(), dpi2.clone());

                    f(
                        ConstSignal::new(crate::PxPoint::default()).into(),
                        sized.clone(),
                        no_limit.clone(),
                    )
                });

                super::resolve_defer_machine(
                    crate::rtree::Node::new(
                        final_area.clone(),
                        None,
                        Some(
                            presize
                                .clone()
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
            }),
        )
    }
}
