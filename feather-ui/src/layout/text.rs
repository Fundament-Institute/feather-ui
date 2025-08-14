// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::cell::RefCell;
use std::rc::Rc;

use derive_where::derive_where;

use crate::{ERect, SourceID, render, rtree};

use super::{Layout, check_unsized, leaf, limit_area};

#[derive_where(Clone)]
pub struct Node<T> {
    pub id: std::sync::Weak<SourceID>,
    pub props: Rc<T>,
    pub buffer: Rc<RefCell<cosmic_text::Buffer>>,
    pub renderable: Rc<dyn render::Renderable>,
}

impl<T: leaf::Padded> Layout<T> for Node<T> {
    fn get_props(&self) -> &T {
        &self.props
    }
    fn stage<'a>(
        &self,
        outer_area: ERect,
        outer_limits: crate::ELimits,
        window: &mut crate::component::window::WindowState,
    ) -> Box<dyn super::Staged + 'a> {
        let mut limits = self.props.limits().resolve(window.dpi) + outer_limits;
        let myarea = self.props.area().resolve(window.dpi);
        let (unsized_x, unsized_y) = check_unsized(myarea);
        let padding = self.props.padding().to_perimeter(window.dpi);
        let allpadding = myarea.bottomright().abs().to_vector().to_size().cast_unit()
            + padding.topleft()
            + padding.bottomright();
        let minmax = limits.v.as_array_mut();
        if unsized_x {
            minmax[2] -= allpadding.width;
            minmax[0] -= allpadding.width;
        }
        if unsized_y {
            minmax[3] -= allpadding.height;
            minmax[1] -= allpadding.height;
        }

        let mut evaluated_area = limit_area(
            super::cap_unsized(myarea * crate::layout::nuetralize_unsized(outer_area)),
            limits,
        );

        let (limitx, limity) = {
            let max = limits.max();
            (
                max.width.is_finite().then_some(max.width),
                max.height.is_finite().then_some(max.height),
            )
        };

        let mut text_buffer = self.buffer.borrow_mut();
        let driver = window.driver.clone();
        let dim = evaluated_area.dim() - allpadding;
        {
            let mut font_system = driver.font_system.write();

            text_buffer.set_size(
                &mut font_system,
                if unsized_x {
                    limitx
                } else {
                    Some(dim.width.max(0.0))
                },
                if unsized_y {
                    limity
                } else {
                    Some(dim.height.max(0.0))
                },
            );
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
            w = w.max(limits.min().width).min(limits.max().width);
            h = h.max(limits.min().height).min(limits.max().height);
            let ltrb = evaluated_area.v.as_array_mut();
            if unsized_x {
                ltrb[2] = ltrb[0] + w + allpadding.width;
            }
            if unsized_y {
                ltrb[3] = ltrb[1] + h + allpadding.height;
            }
        };

        evaluated_area = crate::layout::apply_anchor(
            evaluated_area,
            outer_area,
            self.props.anchor().resolve(window.dpi) * evaluated_area.dim(),
        );

        Box::new(crate::layout::Concrete::new(
            Some(self.renderable.clone()),
            evaluated_area,
            rtree::Node::new(
                evaluated_area.to_untyped(),
                None,
                Default::default(),
                self.id.clone(),
                window,
            ),
            Default::default(),
        ))
    }
}
