// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use ultraviolet::Vec2;

use super::base;
use super::check_unsized_abs;
use super::swap_axis;
use super::Concrete;
use super::Desc;
use super::LayoutWrap;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use crate::RowDirection;
use crate::SourceID;
use crate::ZERO_POINT;
use std::rc::Rc;

pub trait Prop: base::Area + base::Limits + base::Direction {}

crate::gen_from_to_dyn!(Prop);

pub trait Child: base::RLimits + base::Margin + base::Order {}

crate::gen_from_to_dyn!(Child);

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Child;
    // TODO: Make a sorted im::Vector that uses base::Order to order inserted children.
    type Children = im::Vector<Option<Box<dyn LayoutWrap<Self::Child>>>>;

    fn stage<'a>(
        props: &Self::Props,
        outer_area: AbsRect,
        outer_limits: crate::AbsLimits,
        children: &Self::Children,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        // TODO: make insertion efficient by creating a RRB tree of list layout subnodes, in a similar manner to the r-tree nodes.

        let limits = outer_limits + *props.limits();
        let myarea = props.area();
        let (unsized_x, unsized_y) = super::check_unsized(*myarea);
        let dir = props.direction();
        // For the purpose of calculating size, we only care about which axis we're distributing along
        let xaxis = match dir {
            RowDirection::LeftToRight | RowDirection::RightToLeft => true,
            RowDirection::TopToBottom | RowDirection::BottomToTop => false,
        };

        let mut areas: im::Vector<Option<AbsRect>> = im::Vector::new();

        let evaluated_area = if unsized_x || unsized_y {
            let inner_dim = super::limit_dim(super::eval_dim(*myarea, outer_area.dim()), limits);
            let inner_area = AbsRect::from(inner_dim);
            let mut cur = ZERO_POINT;
            let mut max_aux = 0.0;
            let mut bottomright = ZERO_POINT;

            for child in children.iter() {
                let child_props = child.as_ref().unwrap().get_imposed();
                let child_limit = super::apply_limit(inner_dim, limits, *child_props.rlimits());

                let stage = child
                    .as_ref()
                    .unwrap()
                    .stage(inner_area, child_limit, driver);

                let area = stage.get_area();
                bottomright = bottomright.max_by_component(cur + area.bottomright);

                let (main, aux) = super::swap_axis(xaxis, area.bottomright);
                let (mut curmain, mut curaux) = super::swap_axis(xaxis, cur);
                let (main_limit, _) = super::swap_axis(xaxis, inner_dim.0);
                if curmain + main > main_limit {
                    curaux += max_aux;
                    curmain = 0.0;
                    max_aux = 0.0;
                }
                max_aux = max_aux.max(aux);
                curmain += main;
                cur = if xaxis {
                    Vec2::new(curmain, curaux)
                } else {
                    Vec2::new(curaux, curmain)
                };

                // TODO: margins
                areas.push_back(area.into());
            }

            let mut area = *myarea;

            // Unsized objects must always have a single anchor point to make sense, so we copy over from topleft.
            if unsized_x {
                area.bottomright.rel.0.x = area.topleft.rel.0.x;
                // We also add the topleft abs corner to the unsized dimensions to make padding work
                area.bottomright.abs.x += area.topleft.abs.x + bottomright.x;
            }
            if unsized_y {
                area.bottomright.rel.0.y = area.topleft.rel.0.y;
                area.bottomright.abs.y += area.topleft.abs.y + bottomright.y;
            }

            // No need to cap this because unsized axis have now been resolved
            super::limit_area(area * crate::layout::nuetralize_unsized(outer_area), limits)
        } else {
            // No need to zero infinities because in this path, either outer_area is sized or myarea has no relative component.
            super::limit_area(
                *myarea * crate::layout::nuetralize_unsized(outer_area),
                limits,
            )
        };

        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();

        // If our parent is asking for a size estimation along the expansion axis, no need to layout the children
        // TODO: Double check this assumption is true
        let (unsized_x, unsized_y) = check_unsized_abs(outer_area.bottomright);
        if (unsized_x && xaxis) || (unsized_y && !xaxis) {
            return Box::new(Concrete {
                area: evaluated_area,
                render: None,
                rtree: Rc::new(rtree::Node::new(evaluated_area, None, nodes, id)),
                children: staging,
            });
        }

        let evaluated_dim = evaluated_area.dim();
        let mut cur = match dir {
            RowDirection::LeftToRight | RowDirection::TopToBottom => ZERO_POINT,
            RowDirection::RightToLeft => Vec2::new(evaluated_dim.0.x, 0.0),
            RowDirection::BottomToTop => Vec2::new(0.0, evaluated_dim.0.y),
        };
        let mut maxaux: f32 = 0.0;

        for (i, child) in children.iter().enumerate() {
            let child = child.as_ref().unwrap();
            let area = areas[i].unwrap();
            let dim = area.dim().0;
            let (_, aux) = swap_axis(xaxis, dim);

            // TODO: Margins
            match dir {
                RowDirection::RightToLeft => {
                    if cur.x - dim.x < 0.0 {
                        cur.y += maxaux;
                        maxaux = 0.0;
                        cur.x = evaluated_dim.0.x - dim.x;
                    } else {
                        cur.x -= dim.x
                    }
                }
                RowDirection::BottomToTop => {
                    if cur.y - dim.y < 0.0 {
                        cur.x += maxaux;
                        maxaux = 0.0;
                        cur.y = evaluated_dim.0.y - dim.y;
                    } else {
                        cur.y -= dim.y
                    }
                }
                RowDirection::LeftToRight => {
                    if cur.x + dim.x > evaluated_dim.0.x {
                        cur.y += maxaux;
                        maxaux = 0.0;
                        cur.x = 0.0;
                    }
                }
                RowDirection::TopToBottom => {
                    if cur.y + dim.y > evaluated_dim.0.y {
                        cur.x += maxaux;
                        maxaux = 0.0;
                        cur.y = 0.0;
                    }
                }
            };

            let child_area = area + cur;

            match dir {
                RowDirection::LeftToRight => cur.x += dim.x,
                RowDirection::TopToBottom => cur.y += dim.y,
                _ => (),
            };
            maxaux = maxaux.max(aux);

            let child_limit = *child.get_imposed().rlimits() * evaluated_dim;

            let stage = child.stage(child_area, child_limit, driver);
            if let Some(node) = stage.get_rtree().upgrade() {
                nodes.push_back(Some(node));
            }
            staging.push_back(Some(stage));
        }

        Box::new(Concrete {
            area: evaluated_area,
            render: renderable,
            rtree: Rc::new(rtree::Node::new(evaluated_area, None, nodes, id)),
            children: staging,
        })
    }
}
