// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use ultraviolet::Vec2;

use super::{
    Concrete, Desc, Layout, Renderable, Staged, base, check_unsized, map_unsized_area,
    nuetralize_unsized, swap_axis,
};
use crate::{
    AbsRect, DPoint, DValue, MINUS_BOTTOMRIGHT, RowDirection, SourceID, UNSIZED_AXIS, rtree,
};
use std::rc::Rc;

// TODO: use sparse vectors here? Does that even make sense if rows require a default size of some kind?
pub trait Prop: base::Area + base::Limits + base::Anchor + base::Padding {
    fn rows(&self) -> &[DValue];
    fn columns(&self) -> &[DValue];
    fn spacing(&self) -> DPoint; // Spacing is specified as (row, column)
    fn direction(&self) -> RowDirection; // Note that a "normal" grid is TopToBottom here by default, not LeftToRight
}

crate::gen_from_to_dyn!(Prop);

pub trait Child: base::RLimits {
    /// (Row, Column) index of the item
    fn index(&self) -> (usize, usize);
    /// (Row, Column) span of the item, lets items span across multiple rows or columns.
    /// Minimum is (1,1), and the layout won't save you if you tell it to overlap items.
    fn span(&self) -> (usize, usize);
}

crate::gen_from_to_dyn!(Child);

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Child;
    type Children = im::Vector<Option<Box<dyn Layout<Self::Child>>>>;

    fn stage<'a>(
        props: &Self::Props,
        outer_area: AbsRect,
        outer_limits: crate::AbsLimits,
        children: &Self::Children,
        id: std::sync::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        window: &mut crate::component::window::WindowState,
    ) -> Box<dyn Staged + 'a> {
        let mut limits = outer_limits + props.limits().resolve(window.dpi);
        let padding = props.padding().resolve(window.dpi);
        let myarea = props.area().resolve(window.dpi);
        let (unsized_x, unsized_y) = check_unsized(myarea);
        let allpadding = padding.topleft() + padding.bottomright();
        let minmax = limits.0.as_array_mut();
        if unsized_x {
            minmax[2] -= allpadding.x;
            minmax[0] -= allpadding.x;
        }
        if unsized_y {
            minmax[3] -= allpadding.y;
            minmax[1] -= allpadding.y;
        }

        let outer_safe = nuetralize_unsized(outer_area);
        let inner_dim = crate::AbsDim(
            super::limit_dim(super::eval_dim(myarea, outer_area.dim()), limits).0
                - padding.topleft()
                - padding.bottomright(),
        );

        let yaxis = match props.direction() {
            RowDirection::LeftToRight | RowDirection::RightToLeft => false,
            RowDirection::TopToBottom | RowDirection::BottomToTop => true,
        };

        let (outer_column, outer_row) = swap_axis(yaxis, outer_safe.dim().0);
        let (dpi_column, dpi_row) = swap_axis(yaxis, window.dpi);

        let spacing = props.spacing().resolve(Vec2::new(dpi_row, dpi_column))
            * crate::AbsDim(Vec2::new(outer_row, outer_column));
        let nrows = props.rows().len();
        let ncolumns = props.columns().len();

        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();

        let evaluated_area =
            crate::util::alloca_array::<f32, AbsRect>((nrows + ncolumns) * 2, |x| {
                let (resolved, sizes) = x.split_at_mut(nrows + ncolumns);
                {
                    let (rows, columns) = resolved.split_at_mut(nrows);

                    // Fill our max calculation rows with NANs (this ensures max()/min() behave properly)
                    sizes.fill(f32::NAN);

                    let (maxrows, maxcolumns) = sizes.split_at_mut(nrows);

                    // First we precalculate all row/column sizes that we can (if an outer axis is unsized, relative sizes are set to 0)
                    for (i, row) in props.rows().iter().enumerate() {
                        rows[i] = row.resolve(dpi_row).resolve(outer_row);
                    }
                    for (i, column) in props.columns().iter().enumerate() {
                        columns[i] = column.resolve(dpi_column).resolve(outer_column);
                    }

                    // Then we go through all child elements so we can precalculate the maximum area of all rows and columns
                    for child in children.iter() {
                        let child_props = child.as_ref().unwrap().get_props();
                        let child_limit =
                            super::apply_limit(inner_dim, limits, *child_props.rlimits());
                        let (row, column) = child_props.index();

                        if rows[row] == UNSIZED_AXIS || columns[column] == UNSIZED_AXIS {
                            let (w, h) = swap_axis(yaxis, Vec2::new(columns[column], rows[row]));
                            let child_area = AbsRect::new(0.0, 0.0, w, h);

                            let stage =
                                child
                                    .as_ref()
                                    .unwrap()
                                    .stage(child_area, child_limit, window);
                            let area = stage.get_area();
                            let (c, r) = swap_axis(yaxis, area.dim().0);
                            maxrows[row] = maxrows[row].max(r);
                            maxcolumns[column] = maxcolumns[column].max(c);
                        }
                    }
                }

                // Copy back our resolved row or column to any unsized ones
                for (i, size) in sizes.iter().enumerate() {
                    if resolved[i] == UNSIZED_AXIS {
                        resolved[i] = if size.is_nan() { 0.0 } else { *size };
                    }
                }
                let (rows, columns) = resolved.split_at_mut(nrows);
                let (x_used, y_used) = swap_axis(
                    yaxis,
                    Vec2::new(
                        columns.iter().fold(0.0, |x, y| x + y)
                            + (spacing.y * ncolumns.saturating_sub(1) as f32),
                        rows.iter().fold(0.0, |x, y| x + y)
                            + (spacing.x * nrows.saturating_sub(1) as f32),
                    ),
                );
                let area = map_unsized_area(myarea, Vec2::new(x_used, y_used));

                // Calculate the offset to each row or column, without overwriting the size we stored in resolved
                let (row_offsets, column_offsets) = sizes.split_at_mut(nrows);
                let mut offset = 0.0;

                for (i, row) in rows.iter().enumerate() {
                    row_offsets[i] = offset;
                    offset += row + spacing.x;
                }

                offset = 0.0;
                for (i, column) in columns.iter().enumerate() {
                    column_offsets[i] = offset;
                    offset += column + spacing.y;
                }

                for child in children.iter() {
                    let child_props = child.as_ref().unwrap().get_props();
                    let child_limit = super::apply_limit(inner_dim, limits, *child_props.rlimits());
                    let (row, column) = child_props.index();

                    let (x, y) =
                        swap_axis(yaxis, Vec2::new(column_offsets[column], row_offsets[row]));
                    let (w, h) = swap_axis(yaxis, Vec2::new(columns[column], rows[row]));
                    let child_area = AbsRect::new(x, y, x + w, y + h);

                    let stage = child
                        .as_ref()
                        .unwrap()
                        .stage(child_area, child_limit, window);
                    if let Some(node) = stage.get_rtree().upgrade() {
                        nodes.push_back(Some(node));
                    }
                    staging.push_back(Some(stage));
                }

                // No need to cap this because unsized axis have now been resolved
                let evaluated_area = AbsRect(
                    super::limit_area(area * outer_safe, limits).0
                        + (padding.0 * MINUS_BOTTOMRIGHT),
                );

                let anchor = props.anchor().resolve(window.dpi) * evaluated_area.dim();
                evaluated_area - anchor
            });

        Box::new(Concrete {
            area: evaluated_area,
            renderable,
            rtree: rtree::Node::new(evaluated_area, None, nodes, id, window),
            children: staging,
            layer: None,
        })
    }
}
