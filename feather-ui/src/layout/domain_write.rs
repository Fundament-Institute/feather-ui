// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::base::{Empty, RLimits};
use super::{Concrete, Desc, Layout, Renderable, Staged};
use crate::component::ComponentMarker;
use crate::layout::DynLayout;
use crate::reactive::SignalMap;
use crate::reactive::{self, DynSignal, SignalTupleZip, zip_pair};
use crate::{CrossReferenceDomain, PxLimits, RelDim, render, rtree};
use std::marker::PhantomData;
use std::rc::Rc;
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
    type Children = PhantomData<dyn DynLayout<Self::Child>>;

    fn stage<'a, T: render::Prerender + 'static>(
        props: &Self::Props,
        predim: DynSignal<crate::PxDim>,
        _: DynSignal<Self::Children>,
        renderable: Option<T>,
        _: reactive::MutableSignal<RelDim>,
    ) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
        let area = predim.map(|dim| crate::PxRect::from(*dim)).into();

        let id = props.id();
        let domain = props.domain();
        (
            area,
            Box::new(move |offset, final_dim| {
                let final_area =
                    zip_pair(offset, final_dim, |o, dim| crate::Rect::offsetdim(o, dim))
                        .into_dyn_signal();

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
                            area: final_area,
                        }),
                        layer: None,
                    })),
                )
            }),
        )
    }
}
