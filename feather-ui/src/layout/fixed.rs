// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::base;
use super::Concrete;
use super::Desc;
use super::LayoutWrap;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use crate::ZERO_POINT;
use std::rc::Rc;

pub trait Prop: base::Area + base::Anchor + base::Limits + base::ZIndex {}

crate::gen_from_to_dyn!(Prop);

pub trait Child: base::RLimits {}

crate::gen_from_to_dyn!(Child);

impl Child for crate::URect {}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Child;
    type Children = im::Vector<Option<Box<dyn LayoutWrap<Self::Child>>>>;

    fn stage<'a>(
        props: &Self::Props,
        mut outer_area: AbsRect,
        outer_limits: AbsRect,
        children: &Self::Children,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        outer_area = super::nuetralize_infinity(outer_area);
        let myarea = props.area();
        let outer_dim = outer_area.dim();
        let limits = super::merge_limits(outer_limits, *props.limits());

        // If our absolute area contains infinities, we must calculate that axis from the children
        let evaluated_dim =
            if myarea.bottomright.abs.x.is_infinite() || myarea.bottomright.abs.y.is_infinite() {
                let mut inner_dim = super::limit_dim(super::eval_dim(*myarea, outer_dim), limits);
                let inner_area = AbsRect::from(inner_dim);
                // The area we pass to children must be independent of our own area, so it starts at 0,0
                let mut bottomright = ZERO_POINT;

                for child in children.iter() {
                    let child_props = child.as_ref().unwrap().get_imposed();
                    let child_limit = super::eval_limits(*child_props.rlimits(), inner_dim);

                    let stage = child
                        .as_ref()
                        .unwrap()
                        .stage(inner_area, child_limit, driver);
                    bottomright = bottomright.max_by_component(stage.get_area().bottomright);
                }

                if inner_dim.0.x.is_infinite() {
                    inner_dim.0.x = bottomright.x;
                }
                if inner_dim.0.y.is_infinite() {
                    inner_dim.0.y = bottomright.y;
                }

                inner_dim
            } else {
                (*myarea * outer_dim).dim()
            };

        let evaluated_dim = crate::AbsDim(
            evaluated_dim
                .0
                .max_by_component(limits.topleft)
                .min_by_component(limits.bottomright),
        );
        let inner_area = AbsRect::from(evaluated_dim);
        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();

        for child in children.iter() {
            let child_props = child.as_ref().unwrap().get_imposed();
            let child_limit = super::eval_limits(*child_props.rlimits(), evaluated_dim);

            let stage = child
                .as_ref()
                .unwrap()
                .stage(inner_area, child_limit, driver);
            if let Some(node) = stage.get_rtree().upgrade() {
                nodes.push_back(Some(node));
            }
            staging.push_back(Some(stage));
        }

        // TODO: It isn't clear if the simple layout should attempt to handle children changing their estimated
        // sizes after the initial estimate. If we were to handle this, we would need to recalculate the infinite
        // axis with the new child results here, and repeat until it stops changing (we find the fixed point).
        // Because the performance implications are unclear, this might need to be relagated to a special layout.

        // Calculate the anchor using the final evaluated dimensions, after all infinite axis and limits are calculated
        let anchor = *props.anchor() * evaluated_dim;

        // We derive the final area by calculating the correct topleft corner of our self-area, then applying our
        // final evaluated_dim dimensions
        let mut evaluated_area = AbsRect {
            topleft: (myarea.topleft * outer_dim) + outer_area.topleft - anchor,
            bottomright: evaluated_dim.0,
        };
        evaluated_area.bottomright += evaluated_area.topleft;

        Box::new(Concrete {
            area: evaluated_area,
            render: renderable,
            rtree: Rc::new(rtree::Node::new(
                evaluated_area,
                Some(props.zindex()),
                nodes,
                id,
            )),
            children: staging,
        })
    }
}
