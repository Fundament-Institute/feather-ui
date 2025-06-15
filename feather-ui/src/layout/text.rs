// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::rc::Rc;

use derive_where::derive_where;

use crate::{AbsRect, SourceID, render, rtree};

use super::{Layout, check_unsized, leaf, limit_area};

#[derive_where(Clone)]
pub struct Node<T: leaf::Padded> {
    pub id: std::rc::Weak<SourceID>,
    pub props: Rc<T>,
    pub text_render: Rc<render::text::Instance>,
    pub renderable: Rc<dyn render::Renderable>,
}

impl<T: leaf::Padded> Layout<T> for Node<T> {
    fn get_props(&self) -> &T {
        &self.props
    }
    fn stage<'a>(
        &self,
        outer_area: AbsRect,
        outer_limits: crate::AbsLimits,
        window: &mut crate::component::window::WindowState,
    ) -> Box<dyn super::Staged + 'a> {
        let mut limits = self.props.limits().resolve(window.dpi) + outer_limits;
        let myarea = self.props.area().resolve(window.dpi);
        let (unsized_x, unsized_y) = check_unsized(myarea);
        let padding = self.props.padding().resolve(window.dpi);
        let allpadding = myarea.bottomright().abs() + padding.topleft() + padding.bottomright();
        let minmax = limits.0.as_array_mut();
        if unsized_x {
            minmax[2] -= allpadding.x;
            minmax[0] -= allpadding.x;
        }
        if unsized_y {
            minmax[3] -= allpadding.y;
            minmax[1] -= allpadding.y;
        }

        let mut evaluated_area = limit_area(
            super::cap_unsized(myarea * crate::layout::nuetralize_unsized(outer_area)),
            limits,
        );

        let (limitx, limity) = {
            let max = limits.max();
            (
                max.x.is_finite().then_some(max.x),
                max.y.is_finite().then_some(max.y),
            )
        };

        let mut text_binding = self.text_render.text_buffer.borrow_mut();
        let text_buffer = text_binding.as_mut().unwrap();
        let driver = window.graphics.clone();
        let dim = evaluated_area.dim();
        {
            let mut font_system = driver.font_system.borrow_mut();

            text_buffer.set_size(
                &mut font_system,
                if unsized_x { limitx } else { Some(dim.0.x) },
                if unsized_y { limity } else { Some(dim.0.y) },
            );

            text_buffer.shape_until_scroll(&mut font_system, false);
        }

        // If we have indeterminate area, calculate the size
        if unsized_x || unsized_y {
            let mut h = 0.0;
            let mut w: f32 = 0.0;
            for run in text_buffer.layout_runs() {
                w = w.max(run.line_w);
                h += run.line_height;
            }

            // Apply adjusted limits to inner size calculation
            w = w.max(limits.min().x).min(limits.max().x);
            h = h.max(limits.min().y).min(limits.max().y);
            let ltrb = evaluated_area.0.as_array_mut();
            if unsized_x {
                ltrb[2] = ltrb[0] + w + allpadding.x;
            }
            if unsized_y {
                ltrb[3] = ltrb[1] + h + allpadding.y;
            }
        };

        // We always return the full area so we can correctly capture input, but we need
        // to pass our padding to our renderer so it can pad the text while setting the
        // clip rect to the full area
        self.text_render.padding.set(padding);

        evaluated_area = crate::layout::apply_anchor(
            evaluated_area,
            outer_area,
            self.props.anchor().resolve(window.dpi) * evaluated_area.dim(),
        );

        Box::new(crate::layout::Concrete::new(
            Some(self.renderable.clone()),
            evaluated_area,
            rtree::Node::new(
                evaluated_area,
                None,
                Default::default(),
                self.id.clone(),
                window,
            ),
            Default::default(),
        ))
    }
}
