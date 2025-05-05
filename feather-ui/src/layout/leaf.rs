// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::base;
use super::base::Empty;
use super::zero_unsized_area;
use super::Concrete;
use super::Desc;
use super::LayoutWrap;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use crate::SourceID;
use crate::URect;
use std::marker::PhantomData;
use std::rc::Rc;

pub trait Prop: base::Area + base::Limits + base::Anchor {}

crate::gen_from_to_dyn!(Prop);

impl Prop for URect {}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Empty;
    type Children = PhantomData<dyn LayoutWrap<Self::Child>>;

    fn stage<'a>(
        props: &Self::Props,
        mut outer_area: AbsRect,
        outer_limits: crate::AbsLimits,
        _: &Self::Children,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        _: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        outer_area = super::nuetralize_unsized(outer_area);
        let evaluated_area =
            super::limit_area(zero_unsized_area(*props.area()) * outer_area, outer_limits);

        let anchor = *props.anchor() * evaluated_area.dim();
        let evaluated_area = evaluated_area - anchor;

        Box::new(Concrete {
            area: evaluated_area,
            render: renderable,
            rtree: Rc::new(rtree::Node::new(
                evaluated_area,
                None,
                Default::default(),
                id,
            )),
            children: Default::default(),
        })
    }
}
