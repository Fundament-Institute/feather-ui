// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::base::{Empty, RLimits};
use super::{Concrete, Desc, Layout, Renderable, Staged};
use crate::{AbsRect, CrossReferenceDomain, SourceID, render, rtree};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

// A DomainWrite layout spawns a renderable that writes it's area to the target cross-reference domain
pub trait Prop {
    fn domain(&self) -> Arc<CrossReferenceDomain>;
}

crate::gen_from_to_dyn!(Prop);

impl Prop for Arc<CrossReferenceDomain> {
    fn domain(&self) -> Arc<CrossReferenceDomain> {
        self.clone()
    }
}

impl Empty for Arc<CrossReferenceDomain> {}
impl RLimits for Arc<CrossReferenceDomain> {}
impl super::fixed::Child for Arc<CrossReferenceDomain> {}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Empty;
    type Children = PhantomData<dyn Layout<Self::Child>>;

    fn stage<'a>(
        props: &Self::Props,
        mut outer_area: AbsRect,
        outer_limits: crate::AbsLimits,
        _: &Self::Children,
        id: std::sync::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        window: &mut crate::component::window::WindowState,
    ) -> Box<dyn Staged + 'a> {
        outer_area = super::nuetralize_unsized(outer_area);
        outer_area = super::limit_area(outer_area, outer_limits);

        Box::new(Concrete {
            area: outer_area,
            renderable: Some(Rc::new(render::domain::Write {
                id: id.clone(),
                domain: props.domain().clone(),
                base: renderable,
            })),
            rtree: rtree::Node::new(outer_area, None, Default::default(), id, window),
            children: Default::default(),
            layer: None,
        })
    }
}
