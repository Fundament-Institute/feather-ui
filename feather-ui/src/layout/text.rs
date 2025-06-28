// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::{cell::RefCell, rc::Rc};

use derive_where::derive_where;

use crate::{AbsRect, SourceID, render, rtree};

use super::{Layout, check_unsized, leaf, limit_area};

#[derive_where(Clone)]
pub struct Node<T: leaf::Padded> {
    pub id: std::rc::Weak<SourceID>,
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

        let mut text_buffer = self.buffer.borrow_mut();
        let driver = window.driver.clone();
        let dim = evaluated_area.dim().0 - allpadding;
        {
            let mut font_system = driver.font_system.write();

            text_buffer.set_size(
                &mut font_system,
                if unsized_x { limitx } else { Some(dim.x) },
                if unsized_y { limity } else { Some(dim.y) },
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
