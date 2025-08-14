// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::{
    Concrete, Desc, Layout, Renderable, Staged, base, check_unsized, check_unsized_abs,
    map_unsized_area,
};
use crate::{EDim, ERect, rtree};
use std::rc::Rc;

pub trait Prop: base::Area + base::Anchor + base::Limits + base::ZIndex {}

crate::gen_from_to_dyn!(Prop);

pub trait Child: base::RLimits {}

crate::gen_from_to_dyn!(Child);

impl Child for crate::DRect {}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Child;
    type Children = im::Vector<Option<Box<dyn Layout<Self::Child>>>>;

    fn stage<'a>(
        props: &Self::Props,
        outer_area: ERect,
        outer_limits: crate::ELimits,
        children: &Self::Children,
        id: std::sync::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        window: &mut crate::component::window::WindowState,
    ) -> Box<dyn Staged + 'a> {
        // If we have an unsized outer_area, any sized object with relative dimensions must evaluate to 0 (or to the minimum limited size). An
        // unsized object can never have relative dimensions, as that creates a logic loop - instead it can only have a single relative anchor.
        // If both axes are sized, then all limits are applied as if outer_area was unsized, and children calculations are skipped.
        //
        // If we have an unsized outer_area and an unsized myarea.rel, then limits are applied as if outer_area was unsized, and furthermore,
        // they are reduced by myarea.abs.bottomright(), because that will be added on to the total area later, which will still be subject to size
        // limits, so we must anticipate this when calculating how much size the children will have available to them. This forces limits to be
        // true infinite numbers, so we can subtract finite amounts and still have infinity. We can't use infinity anywhere else, because infinity
        // times zero is NaN, so we cap certain calculations at f32::MAX
        //
        // If outer_area is sized and myarea.rel is zero or nonzero, all limits are applied normally and child calculations are skipped.
        // If outer_area is sized and myarea.rel is unsized, limits are applied normally, but are once again reduced by myarea.abs.bottomright() to
        // account for how the area calculations will interact with the limits later on.

        let limits = outer_limits + props.limits().resolve(window.dpi);
        let myarea = props.area().resolve(window.dpi);
        let (unsized_x, unsized_y) = check_unsized(myarea);

        // Check if any axis is unsized in a way that requires us to calculate baseline child sizes
        let evaluated_area = if unsized_x || unsized_y {
            // When an axis is unsized, we don't apply any limits to it, so we don't have to worry about
            // cases where the full evaluated area would invalidate the limit.
            let inner_dim = super::limit_dim(super::eval_dim(myarea, outer_area.dim()), limits);
            let inner_area = ERect::from(inner_dim);
            // The area we pass to children must be independent of our own area, so it starts at 0,0
            let mut bottomright = EDim::zero();

            for child in children.iter() {
                let child_props = child.as_ref().unwrap().get_props();
                let child_limit = super::apply_limit(inner_dim, limits, *child_props.rlimits());

                let stage = child
                    .as_ref()
                    .unwrap()
                    .stage(inner_area, child_limit, window);
                bottomright = bottomright.max(stage.get_area().bottomright().to_vector().to_size());
            }

            let area = map_unsized_area(myarea, bottomright);

            // No need to cap this because unsized axis have now been resolved
            super::limit_area(area * crate::layout::nuetralize_unsized(outer_area), limits)
        } else {
            // If outer_area is unsized here, we nuetralize it when evaluating the relative coordinates.
            super::limit_area(
                myarea * crate::layout::nuetralize_unsized(outer_area),
                limits,
            )
        };

        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();

        // If our parent just wants a size estimate, no need to layout children or render anything
        let (unsized_x, unsized_y) = check_unsized_abs(outer_area.bottomright());
        if unsized_x || unsized_y {
            return Box::new(Concrete::new(
                None,
                evaluated_area,
                rtree::Node::new(
                    evaluated_area.to_untyped(),
                    Some(props.zindex()),
                    nodes,
                    id,
                    window,
                ),
                staging,
            ));
        }

        // We had to evaluate the full area first because our final area calculation can change the dimensions in
        // unsized cases. Thus, we calculate the final inner_area for the children from this evaluated area.
        let evaluated_dim = evaluated_area.dim();

        let inner_area = ERect::from(evaluated_dim);

        for child in children.iter() {
            let child_props = child.as_ref().unwrap().get_props();
            let child_limit = *child_props.rlimits() * evaluated_dim;

            let stage = child
                .as_ref()
                .unwrap()
                .stage(inner_area, child_limit, window);
            if let Some(node) = stage.get_rtree().upgrade() {
                nodes.push_back(Some(node));
            }
            staging.push_back(Some(stage));
        }

        // TODO: It isn't clear if the simple layout should attempt to handle children changing their estimated
        // sizes after the initial estimate. If we were to handle this, we would need to recalculate the unsized
        // axis with the new child results here, and repeat until it stops changing (we find the fixed point).
        // Because the performance implications are unclear, this might need to be relagated to a special layout.

        // Calculate the anchor using the final evaluated dimensions, after all unsized axis and limits are
        // calculated. However, we can only apply the anchor if the parent isn't unsized on that axis.
        let mut anchor = props.anchor().resolve(window.dpi) * evaluated_dim;
        let (unsized_outer_x, unsized_outer_y) =
            crate::layout::check_unsized_abs(outer_area.bottomright());
        if unsized_outer_x {
            anchor.x = 0.0;
        }
        if unsized_outer_y {
            anchor.y = 0.0;
        }
        let evaluated_area = evaluated_area - anchor;

        Box::new(Concrete::new(
            renderable,
            evaluated_area,
            rtree::Node::new(
                evaluated_area.to_untyped(),
                Some(props.zindex()),
                nodes,
                id,
                window,
            ),
            staging,
        ))
    }
}
