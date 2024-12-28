// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2024 Fundament Software SPC <https://fundament.software>

use super::Concrete;
use super::Desc;
use super::Layout;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use crate::UPoint;
use crate::URect;
use crate::Vec2;
use dyn_clone::DynClone;
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Clone, Default)]
pub struct Simple {
    pub margin: URect,
    pub area: URect,
    pub anchor: UPoint,
    pub limits: URect,
    pub zindex: i32,
}

impl Desc for Simple {
    type Props = Simple;
    type Impose = ();
    type Children<A: DynClone + ?Sized> = PhantomData<dyn Layout<Self::Impose>>;

    fn stage<'a>(
        props: &Self::Props,
        mut true_area: AbsRect,
        parent_pos: Vec2,
        _: &Self::Children<dyn Layout<Self::Impose> + '_>,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        if true_area.bottomright.x.is_infinite() {
            true_area.bottomright.x = true_area.topleft.x;
        }
        if true_area.bottomright.y.is_infinite() {
            true_area.bottomright.y = true_area.topleft.y;
        }

        let dim = true_area.dim();
        let inner_area = props.area * true_area;
        let anchor = props.anchor * inner_area.dim();
        let area = AbsRect {
            topleft: inner_area.topleft - anchor + (props.margin.topleft * dim),
            bottomright: (inner_area.bottomright - anchor - (props.margin.bottomright * dim))
                .into(),
        };

        Box::new(Concrete {
            area: area - parent_pos,
            render: renderable
                .map(|x| x.render(area, driver))
                .unwrap_or_default(),
            rtree: Rc::new(rtree::Node::new(
                area - parent_pos,
                Some(props.zindex),
                Default::default(),
                id,
            )),
            children: Default::default(),
        })
    }
}
