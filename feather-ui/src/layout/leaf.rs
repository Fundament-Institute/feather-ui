// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::base;
use super::base::Empty;
use super::Concrete;
use super::Desc;
use super::LayoutWrap;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use crate::SourceID;
use crate::URect;
use crate::Vec2;
use std::marker::PhantomData;
use std::rc::Rc;

pub trait Prop: base::Area {}

crate::gen_from_to_dyn!(Prop);

impl Prop for URect {}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Empty;
    type Children = PhantomData<dyn LayoutWrap<Self::Child>>;

    fn stage<'a>(
        props: &Self::Props,
        mut true_area: AbsRect,
        _: &Self::Children,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        if true_area.bottomright.x.is_infinite() {
            true_area.bottomright.x = true_area.topleft.x;
        }
        if true_area.bottomright.y.is_infinite() {
            true_area.bottomright.y = true_area.topleft.y;
        }

        let area = *props.area() * true_area;

        Box::new(Concrete {
            area: area - true_area.topleft,
            render: renderable,
            rtree: Rc::new(rtree::Node::new(
                area - true_area.topleft,
                None,
                Default::default(),
                id,
            )),
            children: Default::default(),
        })
    }
}
