// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::Concrete;
use super::Desc;
use super::Layout;
use super::Renderable;
use super::Staged;
use crate::layout::zero_infinity;
use crate::rtree;
use crate::AbsDim;
use crate::AbsRect;
use crate::UPoint;
use crate::URect;
use crate::Vec2;
use dyn_clone::DynClone;
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
    type Children<A: DynClone + ?Sized> = im::Vector<Option<Box<dyn Layout<Self::Impose>>>>;

    fn stage<'a>(
        props: &Self::Props,
        true_area: AbsRect,
        parent_pos: Vec2,
        children: &Self::Children<dyn Layout<Self::Impose> + '_>,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        // If true_area has an infinite axis, we set it's dimensions to zero before multiplying it with our area.
        let mut area = true_area;
        if true_area.bottomright.x.is_infinite() {
            area.bottomright.x = area.topleft.x;
        }
        if true_area.bottomright.y.is_infinite() {
            area.bottomright.y = area.topleft.y;
        }

        area = props.area * area;

        // If our own area also has an infinite axis, then we need to evaluate the children
        if area.bottomright.x.is_infinite() || area.bottomright.y.is_infinite() {
            let dim = AbsDim(zero_infinity(area.dim().into()));
            let inner_area = AbsRect {
                topleft: area.topleft + (props.margin.topleft * dim),
                bottomright: (area.bottomright - (props.margin.bottomright * dim)),
            };

            let mut bottomright = inner_area.topleft - true_area.topleft;

            for child in children.iter() {
                let stage = child
                    .as_ref()
                    .unwrap()
                    .stage(inner_area, true_area.topleft, driver);

                bottomright = bottomright.max_by_component(stage.get_area().bottomright);
            }
            if area.bottomright.x.is_infinite() {
                area.bottomright.x = true_area.topleft.x + bottomright.x;
            }
            if area.bottomright.y.is_infinite() {
                area.bottomright.y = true_area.topleft.y + bottomright.y;
            }
        };

        let dim = area.dim();
        let anchor = props.anchor * dim;
        let inner_area = AbsRect {
            topleft: area.topleft - anchor + (props.margin.topleft * dim),
            bottomright: (area.bottomright - anchor - (props.margin.bottomright * dim)),
        };

        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();

        for child in children.iter() {
            let stage = child
                .as_ref()
                .unwrap()
                .stage(inner_area, inner_area.topleft, driver);
            if let Some(node) = stage.get_rtree().upgrade() {
                nodes.push_back(Some(node));
            }
            staging.push_back(Some(stage));
        }

        // Our area does not change based on child sizes, so we have no bottom-up resolution step to do here
        Box::new(Concrete {
            area: area - parent_pos,
            render: renderable
                .map(|x| x.render(area, driver))
                .unwrap_or_default(),
            rtree: Rc::new(rtree::Node::new(
                area - parent_pos,
                Some(props.zindex),
                nodes,
                id,
            )),
            children: staging,
        })
    }
}
