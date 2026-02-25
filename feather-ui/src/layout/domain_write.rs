// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::Desc;
use super::base::{Empty, RLimits};
use crate::component::ComponentMarker;
use crate::reactive::{self, DynSignal};
use crate::{CrossReferenceDomain, RelDim, render};
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

    fn stage<'a, T: render::Prerender + 'static>(
        props: &Self::Props,
        _: DynSignal<crate::PxLimits>,
        _: DynSignal<Self::Children>,
        renderable: Option<T>,
        dpi: reactive::MutableSignal<RelDim>,
        defer: Option<super::DeferMachine<Self::Provider>>,
    ) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
        use crate::reactive::SignalZip;

        let id = props.id();
        let domain = props.domain();
        (
            crate::reactive::const_default().into(),
            Box::new(move |offset, final_dim, final_limits| {
                let final_area = (offset, final_dim, final_limits)
                    .zip()
                    .flatmap(|(o, dim, limits)| {
                        super::limit_area(crate::Rect::offsetdim(*o, *dim), *limits)
                    })
                    .into_dyn();

                super::resolve_defer_machine(
                    crate::rtree::Node::new(
                        final_area.clone(),
                        None,
                        None,
                        Some(Box::new(crate::layout::Concrete {
                            area: final_area.clone(),
                            renderable: Some(render::domain::Write {
                                id: id.clone(),
                                domain: domain.clone(),
                                base: renderable.as_ref().map(|x| x.prerender(final_area.clone())),
                                area: final_area.clone(),
                            }),
                            layer: None,
                        })),
                    ),
                    &defer,
                    crate::reactive::zip((final_area, dpi.clone()))
                        .value()
                        .into_dyn(),
                )
            }),
        )
    }
}
