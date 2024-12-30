// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2024 Fundament Software SPC <https://fundament.software>

use super::zero_infinity;
use super::Concrete;
use super::Desc;
use super::Layout;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsDim;
use crate::AbsRect;
use crate::UPoint;
use crate::URect;
use crate::Vec2;
use dyn_clone::DynClone;
use std::rc::Rc;

#[derive(Clone, Default)]
pub struct Basic {
    pub padding: URect,
    pub zindex: i32,
}

impl Desc for Basic {
    type Props = Basic;
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
        // Calculate area for children
        let inner_area =
            if true_area.bottomright.x.is_infinite() || true_area.bottomright.y.is_infinite() {
                let dim = AbsDim(zero_infinity(true_area.dim().into()));

                let mut area = AbsRect {
                    topleft: true_area.topleft + (props.padding.topleft * dim),
                    bottomright: (true_area.bottomright - (props.padding.bottomright * dim)).into(),
                };
                if true_area.bottomright.x.is_infinite() {
                    area.bottomright.x = f32::INFINITY;
                }
                if true_area.bottomright.y.is_infinite() {
                    area.bottomright.y = f32::INFINITY;
                }

                let mut bottomright = area.topleft - true_area.topleft;

                for child in children.iter() {
                    let stage = child
                        .as_ref()
                        .unwrap()
                        .stage(area, true_area.topleft, driver);

                    bottomright = bottomright.max_by_component(stage.get_area().bottomright);
                }
                if true_area.bottomright.x.is_infinite() {
                    area.bottomright.x = true_area.topleft.x + bottomright.x;
                }
                if true_area.bottomright.y.is_infinite() {
                    area.bottomright.y = true_area.topleft.y + bottomright.y;
                }
                area
            } else {
                let dim = true_area.dim();
                AbsRect {
                    topleft: true_area.topleft + (props.padding.topleft * dim),
                    bottomright: (true_area.bottomright - (props.padding.bottomright * dim)).into(),
                }
            };

        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();

        for child in children.iter() {
            let stage = child
                .as_ref()
                .unwrap()
                .stage(inner_area, true_area.topleft, driver);
            if let Some(node) = stage.get_rtree().upgrade() {
                nodes.push_back(Some(node));
            }
            staging.push_back(Some(stage));
        }

        Box::new(Concrete {
            area: true_area - parent_pos,
            render: renderable
                .map(|x| x.render(true_area, driver))
                .unwrap_or_default(),
            rtree: Rc::new(rtree::Node::new(
                true_area - parent_pos,
                Some(props.zindex),
                nodes,
                id,
            )),
            children: staging,
        })
    }
}
