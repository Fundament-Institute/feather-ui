// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::prop;
use super::Concrete;
use super::Desc;
use super::Layout;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use crate::SourceID;
use crate::URect;
use crate::Vec2;
use dyn_clone::DynClone;
use std::marker::PhantomData;
use std::rc::Rc;

pub trait Prop: prop::Area {}

impl Prop for URect {}

impl Desc for Rc<dyn Prop> {
    type Props = Rc<dyn Prop>;
    type Impose = ();
    type Children<A: DynClone + ?Sized> = PhantomData<dyn Layout<Self::Impose>>;

    fn stage<'a>(
        props: &Self::Props,
        mut true_area: AbsRect,
        parent_pos: Vec2,
        _: &Self::Children<dyn Layout<Self::Impose> + '_>,
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
            area: area - parent_pos,
            render: renderable
                .map(|x| x.render(area, driver))
                .unwrap_or_default(),
            rtree: Rc::new(rtree::Node::new(
                area - parent_pos,
                None,
                Default::default(),
                id,
            )),
            children: Default::default(),
        })
    }
}
