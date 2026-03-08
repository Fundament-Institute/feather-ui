// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::Desc;
use super::base::{Empty, RLimits};
use crate::component::ComponentMarker;
use crate::reactive::{DynSignal, MutableSignal, zip_pair};
use crate::{CrossReferenceDomain, Limited, RelDim, Unsizable, render};
use std::sync::Arc;

// A DomainWrite layout spawns a renderable that writes it's area to the target
// cross-reference domain
pub trait Prop {
    fn domain(&self) -> Arc<CrossReferenceDomain>;
    fn id(&self) -> std::sync::Weak<dyn ComponentMarker + Send + Sync>;
}

impl ComponentMarker for CrossReferenceDomain {}

crate::gen_dyn_prop!(Prop);

impl Prop for Arc<CrossReferenceDomain> {
    fn domain(&self) -> Arc<CrossReferenceDomain> {
        self.clone()
    }
    fn id(&self) -> std::sync::Weak<dyn ComponentMarker + Send + Sync> {
        std::sync::Arc::<CrossReferenceDomain>::downgrade(self)
    }
}

impl Empty for Arc<CrossReferenceDomain> {}
impl RLimits for Arc<CrossReferenceDomain> {}
impl super::fixed::Child for Arc<CrossReferenceDomain> {}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Empty;
    type Children = ();
    type Provider = dyn crate::reactive::SignalProvider<Item = (crate::PxRect, crate::RelDim)>;

    type Staging = MutableSignal<RelDim>;

    fn presize(
        _: &Self::Props,
        dpi: MutableSignal<crate::RelDim>,
        _: DynSignal<Self::Children>,
    ) -> (DynSignal<crate::PxRect>, Self::Staging) {
        (crate::reactive::const_default().into(), dpi)
    }

    fn size(
        _: &Self::Props,
        dim: DynSignal<crate::UnsizedDim>,
        limits: DynSignal<crate::PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<crate::PxRect>, Self::Staging) {
        (
            zip_pair(dim, limits, |dim, limits| {
                crate::PxRect::from(dim.limit(*limits).zero_unsized())
            })
            .into_dyn(),
            data,
        )
    }

    fn stage<T: render::Prerender + 'static>(
        props: &Self::Props,
        offset: DynSignal<crate::PxPoint>,
        area: DynSignal<crate::PxRect>,
        renderable: Option<T>,
        defer: Option<super::DeferMachine<Self::Provider>>,
        data: Self::Staging,
    ) -> std::rc::Rc<crate::rtree::Node> {
        let final_area = (area + offset).into_dyn();

        super::resolve_defer_machine(
            crate::rtree::Node::new(
                final_area.clone(),
                None,
                None,
                Some(Box::new(crate::layout::Concrete {
                    area: final_area.clone(),
                    renderable: Some(render::domain::Write {
                        id: props.id(),
                        domain: props.domain(),
                        base: renderable.as_ref().map(|x| x.prerender(final_area.clone())),
                        area: final_area.clone(),
                    }),
                    layer: None,
                })),
            ),
            &defer,
            crate::reactive::zip((final_area, data)).value().into_dyn(),
        )
    }
}
