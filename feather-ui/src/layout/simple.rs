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
        outer_area: AbsRect,
        children: &Self::Children,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        // If outer_area has an infinite axis, we set it's dimensions to zero before doing anything else.
        let mut myarea = outer_area;
        if outer_area.bottomright.x.is_infinite() {
            myarea.bottomright.x = myarea.topleft.x;
        }
        if outer_area.bottomright.y.is_infinite() {
            myarea.bottomright.y = myarea.topleft.y;
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
        let margin = *props.margin() * myarea.dim(); // area has already had infinities removed
        myarea.topleft =
            Vec2::min_by_component(myarea.topleft + margin.topleft, myarea.bottomright);
        myarea.bottomright =
            Vec2::max_by_component(myarea.bottomright - margin.bottomright, myarea.topleft);

        let area = props.area();
        let mydim = myarea.dim();

        // If our absolute area contains infinities, we must calculate that axis from the children
        let inner_area =
            if area.bottomright.abs.x.is_infinite() || area.bottomright.abs.y.is_infinite() {
                // We only calculate the area for the non-infinite axis here to try to avoid slow float calculations
                // This might be unnecessary when using optimized SSE operations.

                // The area we pass to children must be independent of our own area. Because a simple layout doesn't
                // arrange it's children, the inner_area will always be a 0,0 rectangle with our current dimensions.
                let mut inner_area = AbsRect {
                    topleft: Vec2::zero(),
                    bottomright: area.bottomright.abs,
                };

                // TODO: Relative limits are already weird, should they be relative to the original untouched area?
                let limits = props.limits().0 * mydim;

                // Apply the relative dimensions and limits to the non-infinite axis
                if !area.bottomright.abs.x.is_infinite() {
                    inner_area.bottomright.x = area.bottomright.abs.x - area.topleft.abs.x;
                    inner_area.bottomright.x +=
                        (area.bottomright.rel.x - area.topleft.rel.x) * mydim.0.x;
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
                    inner_area.bottomright.y = area.bottomright.abs.y - area.topleft.abs.y;
                    inner_area.bottomright.y +=
                        (area.bottomright.rel.y - area.topleft.rel.y) * mydim.0.y;
                    inner_area.bottomright.y = inner_area
                        .bottomright
                        .y
                        .max(inner_area.topleft.y + limits.topleft.y);
                    inner_area.bottomright.y = inner_area
                        .bottomright
                        .y
                        .min(inner_area.topleft.y + limits.bottomright.y);
                }

                let mut bottomright = inner_area.topleft;

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
                (*area * props.limits().apply(mydim)) - area.topleft.abs
            };

        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();

        for child in children.iter() {
            let stage = child.as_ref().unwrap().stage(inner_area, driver);
            if let Some(node) = stage.get_rtree().upgrade() {
                nodes.push_back(Some(node));
            }
            staging.push_back(Some(stage));
        }

        // TODO: It isn't clear if the simple layout should attempt to handle children changing their estimated
        // sizes after the initial estimate. If we were to handle this, we would need to recalculate the infinite
        // axis with the new child results here, and repeat until it stops changing (we find the fixed point).
        // Because the performance implications are unclear, this might need to be relagated to a special layout.

        // We derive the final area by calculating the correct topleft corner of our self-area, then shifting the
        // inner_area rectangle to this final position.
        myarea = inner_area + ((area.topleft * mydim) + myarea.topleft);

        // Calculate the anchor using the final dimensions, after all infinite axis and limits are calculated
        let anchor = *props.anchor() * mydim;
        myarea.topleft -= anchor;
        myarea.bottomright -= anchor;

        Box::new(Concrete {
            area: myarea,
            render: renderable,
            rtree: Rc::new(rtree::Node::new(myarea, Some(props.zindex()), nodes, id)),
            children: staging,
        })
    }
}
