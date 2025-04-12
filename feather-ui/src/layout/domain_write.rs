// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::base::Empty;
use super::Concrete;
use super::Desc;
use super::LayoutWrap;
use super::Renderable;
use super::Staged;
use crate::outline::CrossReferenceDomain;
use crate::rtree;
use crate::AbsRect;
use crate::SourceID;
use crate::Vec2;
use std::marker::PhantomData;
use std::rc::Rc;

// A DomainWrite layout simply writes it's final AbsRect to the target cross-reference domain
pub trait Prop {
    fn domain(&self) -> Rc<CrossReferenceDomain>;
}

crate::gen_from_to_dyn!(Prop);

impl Prop for Rc<CrossReferenceDomain> {
    fn domain(&self) -> Rc<CrossReferenceDomain> {
        self.clone()
    }
}

impl Empty for Rc<CrossReferenceDomain> {}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Empty;
    type Children = PhantomData<dyn LayoutWrap<Self::Child>>;

    fn stage<'a>(
        props: &Self::Props,
        mut outer_area: AbsRect,
        _: &Self::Children,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        if outer_area.bottomright.x.is_infinite() {
            outer_area.bottomright.x = outer_area.topleft.x;
        }
        if outer_area.bottomright.y.is_infinite() {
            outer_area.bottomright.y = outer_area.topleft.y;
        }

        // TODO: This can't actually work because it relies on the absolute area, which we were never supposed
        // to have access to. Instead it needs to store a reference to the relative parent of this object so it can
        // convert from this relative space to another relative space correctly. This may require walking up the
        // entire layout tree.
        if let Some(idref) = id.upgrade() {
            props.domain().write_area(idref, outer_area);
        }

        Box::new(Concrete {
            area: outer_area,
            render: renderable,
            rtree: Rc::new(rtree::Node::new(outer_area, None, Default::default(), id)),
            children: Default::default(),
        })
    }
}
