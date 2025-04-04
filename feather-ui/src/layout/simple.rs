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
use crate::Vec2;
use std::rc::Rc;

pub trait Prop: base::Margin + base::Area + base::Anchor + base::Limits + base::ZIndex {}

crate::gen_from_to_dyn!(Prop);

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Empty;
    type Children = im::Vector<Option<Box<dyn LayoutWrap<dyn Empty>>>>;

    fn stage<'a>(
        props: &Self::Props,
        true_area: AbsRect,
        children: &Self::Children,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        // If true_area has an infinite axis, we set it's dimensions to zero before doing anything else.
        let mut curarea = true_area;
        if true_area.bottomright.x.is_infinite() {
            curarea.bottomright.x = curarea.topleft.x;
        }
        if true_area.bottomright.y.is_infinite() {
            curarea.bottomright.y = curarea.topleft.y;
        }

        // We apply margin first - in many cases, this order doesn't matter, but it does in the general case.
        // If we have a 30x30 area, with 20 right margin and 0 left margin, then the relative point 0.5 depends
        // on if we apply margin first before evaluating:
        //
        // Margin First: (30 - 20) * 0.5 = 5
        // Margin Second: (30 * 0.5) - 20 = -5
        //
        // Having a point that's supposed to be in the middle of our area be at -5 is nonsensical. However,
        // We also have to be careful not to overshoot. Margin cannot shrink the area beyond 0x0, and we should
        // ideally keep the resulting degenerate point at a sensible location. Because of this ordering,
        // relative margin is always evaluated against the entire available child area, unlike everything else.

        // TODO: optimize using SSE instructions
        let margin = *props.margin() * curarea.dim(); // area has already had infinities removed
        curarea.topleft =
            Vec2::min_by_component(curarea.topleft + margin.topleft, curarea.bottomright);
        curarea.bottomright =
            Vec2::max_by_component(curarea.bottomright - margin.bottomright, curarea.topleft);

        let area = props.area();

        // If our absolute area contains infinities, we must calculate that axis from the children
        curarea = if area.bottomright.abs.x.is_infinite() || area.bottomright.abs.y.is_infinite() {
            // We only calculate the area for the non-infinite axis here to try to avoid slow float calculations
            // This might be unnecessary when using optimized SSE operations.

            // The anchor does not change the dimensions, so we elide calculating it in this step.

            let mut inner_area = AbsRect {
                topleft: area.topleft.abs,
                bottomright: area.bottomright.abs,
            };

            // TODO: Relative limits are already weird, should they be relative to the original untouched area?
            let limits = props.limits().0 * curarea.dim();
            let dim = curarea.dim().0;

            // Apply the relative dimensions and limits to the non-infinite axis
            if !area.bottomright.abs.x.is_infinite() {
                inner_area.topleft.x += area.topleft.rel.x * dim.x;
                inner_area.bottomright.x += area.bottomright.rel.x * dim.x;
                inner_area.bottomright.x = inner_area
                    .bottomright
                    .x
                    .max(inner_area.topleft.x + limits.topleft.x);
                inner_area.bottomright.x = inner_area
                    .bottomright
                    .x
                    .min(inner_area.topleft.x + limits.bottomright.x);
            }
            if !area.bottomright.abs.y.is_infinite() {
                inner_area.topleft.y += area.topleft.rel.y * dim.y;
                inner_area.bottomright.y += area.bottomright.rel.y * dim.y;
                inner_area.bottomright.y = inner_area
                    .bottomright
                    .y
                    .max(inner_area.topleft.y + limits.topleft.y);
                inner_area.bottomright.y = inner_area
                    .bottomright
                    .y
                    .min(inner_area.topleft.y + limits.bottomright.y);
            }

            // Currently we do not have seperate padding, if we did, we would calculate it here
            let mut bottomright = true_area.topleft;

            for child in children.iter() {
                let stage = child.as_ref().unwrap().stage(inner_area, driver);
                bottomright = bottomright.max_by_component(stage.get_area().bottomright);
            }

            if area.bottomright.abs.x.is_infinite() {
                inner_area.bottomright.x = inner_area.topleft.x + bottomright.x;
            }
            if area.bottomright.abs.y.is_infinite() {
                inner_area.bottomright.y = inner_area.topleft.y + bottomright.y;
            }

            // Re-apply size limits
            inner_area.bottomright = inner_area
                .bottomright
                .max_by_component(inner_area.topleft + limits.topleft);
            inner_area.bottomright = inner_area
                .bottomright
                .min_by_component(inner_area.topleft + limits.bottomright);

            inner_area
        } else {
            *area * props.limits().apply(curarea.dim())
        };

        // Calculate the anchor using the final dimensions, after all infinite axis and limits are calculated
        let anchor = *props.anchor() * curarea.dim();
        curarea.topleft -= anchor;
        curarea.bottomright -= anchor;

        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();

        for child in children.iter() {
            let stage = child.as_ref().unwrap().stage(curarea, driver);
            if let Some(node) = stage.get_rtree().upgrade() {
                nodes.push_back(Some(node));
            }
            staging.push_back(Some(stage));
        }

        // TODO: It isn't clear if the simple layout should attempt to handle children changing their estimated
        // sizes after the initial estimate. If we were to handle this, we would need to recalculate the infinite
        // axis with the new child results here, and repeat until it stops changing (we find the fixed point).
        // Because the performance implications are unclear, this might need to be relagated to a special layout.

        Box::new(Concrete {
            area: curarea,
            render: renderable,
            rtree: Rc::new(rtree::Node::new(curarea, Some(props.zindex()), nodes, id)),
            children: staging,
        })
    }
}
