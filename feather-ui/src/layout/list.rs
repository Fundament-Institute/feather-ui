// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use ultraviolet::Vec2;

use super::base;
use super::check_unsized_abs;
use super::nuetralize_unsized;
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

        // Even if both axis are sized, we have to precalculate the areas and margins anyway.
        let inner_dim = super::limit_dim(super::eval_dim(*myarea, outer_area.dim()), limits);
        let inner_area = AbsRect::from(inner_dim);
        let outer_safe = nuetralize_unsized(outer_area);
        // The inner_dim must preserve whether an axis is unsized, but the actual limits must be respected regardless.
        let (main_limit, _) = super::swap_axis(xaxis, inner_dim.0.min_by_component(limits.max()));

        // This should eventually be a persistent fold
        let mut areas: im::Vector<Option<(AbsRect, f32)>> = im::Vector::new();
        let mut aux_margins: im::Vector<f32> = im::Vector::new();
        let mut cur = ZERO_POINT;
        let mut max_main = 0.0;
        let mut max_aux: f32 = 0.0;
        let mut prev_margin = f32::NAN;
        let mut aux_margin: f32 = 0.0;
        let mut aux_margin_bottom = f32::NAN;
        let mut prev_aux_margin = f32::NAN;

        for child in children.iter() {
            let child_props = child.as_ref().unwrap().get_imposed();
            let child_limit = super::apply_limit(inner_dim, limits, *child_props.rlimits());
            let child_margin = *child_props.margin() * outer_safe;

            let stage = child
                .as_ref()
                .unwrap()
                .stage(inner_area, child_limit, driver);
            let area = stage.get_area();

            let (margin_main, child_margin_aux) = super::swap_axis(xaxis, child_margin.topleft);
            let (main, aux) = super::swap_axis(xaxis, area.dim().0);
            let mut margin = super::merge_margin(prev_margin, margin_main);
            // Have to add the margin here before we zero it
            areas.push_back(Some((area, margin)));

            if !prev_margin.is_nan() && cur.x + main + margin > main_limit {
                max_main = cur.x.max(max_main);
                cur.x = 0.0;
                let aux_merge = super::merge_margin(prev_aux_margin, aux_margin);
                aux_margins.push_back(aux_merge);
                cur.y += max_aux + aux_merge;
                max_aux = 0.0;
                margin = 0.0;
                aux_margin = 0.0;
                prev_aux_margin = aux_margin_bottom;
                aux_margin_bottom = f32::NAN;
            }

            cur.x += main + margin;
            aux_margin = aux_margin.max(child_margin_aux);
            max_aux = max_aux.max(aux);

            let (margin, child_margin_aux) = super::swap_axis(xaxis, child_margin.bottomright);
            prev_margin = margin;
            aux_margin_bottom = aux_margin_bottom.max(child_margin_aux);
        }

        // Final bounds calculations
        max_main = cur.x.max(max_main);
        let aux_merge = super::merge_margin(prev_aux_margin, aux_margin);
        aux_margins.push_back(aux_merge);
        cur.y += max_aux + aux_margin;
        let mut area = *myarea;

        // Unsized objects must always have a single anchor point to make sense, so we copy over from topleft.
        if unsized_x {
            area.bottomright.rel.0.x = area.topleft.rel.0.x;
            // We also add the topleft abs corner to the unsized dimensions to make padding work
            area.bottomright.abs.x += area.topleft.abs.x + max_main;
        }
        if unsized_y {
            area.bottomright.rel.0.y = area.topleft.rel.0.y;
            area.bottomright.abs.y += area.topleft.abs.y + cur.y;
        }

        // No need to cap this because unsized axis have now been resolved
        let evaluated_area = super::limit_area(area * outer_safe, limits);

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
        aux_margins.pop_front();
        let mut aux_margin = aux_margins.pop_front().unwrap_or_default();

        for (i, child) in children.iter().enumerate() {
            let child = child.as_ref().unwrap();
            let (area, margin) = areas[i].unwrap();
            let dim = area.dim().0;
            let (_, aux) = swap_axis(xaxis, dim);

            match dir {
                RowDirection::RightToLeft => {
                    if cur.x - dim.x - margin < 0.0 {
                        cur.y += maxaux + aux_margin;
                        aux_margin = aux_margins.pop_front().unwrap_or_default();
                        maxaux = 0.0;
                        cur.x = evaluated_dim.0.x - dim.x;
                    } else {
                        cur.x -= dim.x + margin
                    }
                }
                RowDirection::BottomToTop => {
                    if cur.y - dim.y - margin < 0.0 {
                        cur.x += maxaux + aux_margin;
                        aux_margin = aux_margins.pop_front().unwrap_or_default();
                        maxaux = 0.0;
                        cur.y = evaluated_dim.0.y - dim.y;
                    } else {
                        cur.y -= dim.y + margin
                    }
                }
                RowDirection::LeftToRight => {
                    if cur.x + dim.x + margin > evaluated_dim.0.x {
                        cur.y += maxaux + aux_margin;
                        aux_margin = aux_margins.pop_front().unwrap_or_default();
                        maxaux = 0.0;
                        cur.x = 0.0;
                    } else {
                        cur.x += margin;
                    }
                }
                RowDirection::TopToBottom => {
                    if cur.y + dim.y + margin > evaluated_dim.0.y {
                        cur.x += maxaux + aux_margin;
                        aux_margin = aux_margins.pop_front().unwrap_or_default();
                        maxaux = 0.0;
                        cur.y = 0.0;
                    } else {
                        cur.y += margin;
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
