// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::{
    Concrete, Desc, Layout, Renderable, Staged, base, cap_unsized, check_unsized_abs,
    map_unsized_area, merge_margin, nuetralize_unsized,
};
use crate::persist::{FnPersist2, VectorFold};
use crate::{AbsLimits, AbsRect, DAbsRect, DValue, RowDirection, UNSIZED_AXIS, Vec2, rtree};
use derive_more::TryFrom;
use smallvec::SmallVec;
use std::rc::Rc;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, TryFrom)]
#[repr(u8)]
pub enum FlexJustify {
    #[default]
    Start,
    Center,
    End,
    SpaceBetween,
    SpaceAround,
    SpaceFull,
}

pub trait Prop:
    base::ZIndex + base::Obstacles + base::Limits + base::Direction + base::Area
{
    fn wrap(&self) -> bool;
    fn justify(&self) -> FlexJustify;
    fn align(&self) -> FlexJustify;
}

crate::gen_from_to_dyn!(Prop);

pub trait Child: base::Margin + base::RLimits + base::Order {
    fn grow(&self) -> f32;
    fn shrink(&self) -> f32;
    fn basis(&self) -> DValue;
}

crate::gen_from_to_dyn!(Child);

#[allow(clippy::too_many_arguments)]
fn next_obstacle(
    obstacles: &[DAbsRect],
    max_aux: f32,
    main: f32,
    aux: f32,
    xaxis: bool,
    total_main: f32,
    min: &mut usize,
    dpi: Vec2,
) -> (f32, f32) {
    // Given our current X/Y position, what is the next obstacle we would run into?
    let mut i = *min;
    while i < obstacles.len() {
        let obstacle = obstacles[i].resolve(dpi);
        let (mut start, aux_start) = super::swap_axis(xaxis, obstacle.topleft());

        if total_main > 0.0 {
            start = total_main - start;
        }

        // If we've reached an obstacle whose top/left is past the (current known) bottom of this line, we won't match anything.
        if aux_start > aux + max_aux {
            break;
        }

        let (mut end, aux_end) = super::swap_axis(xaxis, obstacle.bottomright());

        if total_main > 0.0 {
            end = total_main - end;
        }

        // If aux is past past the bottom/right, we can skip this obstacle for all future lines in this wrap attempt.
        if aux_end < aux {
            i += 1;
            *min = i;
            continue;
        }

        // If our main axis has gone past this obstacles starting edge, we already passed it, so skip forward
        if main > start {
            i += 1;
            continue;
        }

        return (start, end);
    }

    (f32::INFINITY, 0.0)
}

fn justify_inner_outer(align: FlexJustify, total: f32, used: f32, count: i32) -> (f32, f32) {
    match align {
        FlexJustify::SpaceBetween => (0.0, (total - used) / (count - 1) as f32),
        FlexJustify::SpaceAround => {
            let r = (total - used) / count as f32;
            (r / 2.0, r)
        }
        FlexJustify::SpaceFull => {
            let r = (total - used) / (count + 1) as f32;
            (r, r)
        }
        FlexJustify::Center => ((total - used) / 2.0, 0.0),
        FlexJustify::End => (total - used, 0.0),
        _ => (0.0, 0.0),
    }
}

// Stores information about a calculated linebreak
struct Linebreak {
    index: usize,
    lineheight: f32,
    skip: f32, // If this is just skipping over an obstacle, tells us where to skip to.
    aux_margin: f32, // Maximum margin value between lines
}

impl Linebreak {
    pub fn new(index: usize, lineheight: f32, skip: f32, aux_margin: f32) -> Self {
        Self {
            index,
            lineheight,
            skip,
            aux_margin,
        }
    }
}

#[derive(Clone)]
struct ChildCache {
    basis: f32,
    grow: f32,
    shrink: f32,
    aux: f32,
    margin: AbsRect,
    limits: AbsLimits,
}

#[allow(clippy::too_many_arguments)]
fn wrap_line(
    childareas: &im::Vector<Option<ChildCache>>,
    props: &dyn Prop,
    xaxis: bool,
    total_main: f32,
    total_aux: f32,
    mut used_aux: f32,
    mut linecount: i32,
    justify: FlexJustify,
    backwards: bool,
    dpi: Vec2,
) -> (SmallVec<[Linebreak; 10]>, i32, f32) {
    let mut breaks: SmallVec<[Linebreak; 10]> = SmallVec::new();

    let (mut aux, inner) = justify_inner_outer(justify, total_aux, used_aux, linecount);

    // Reset linecount and used_aux
    linecount = 1;
    used_aux = 0.0;
    let mut max_aux = 0.0;
    let mut max_aux_margin: f32 = 0.0;
    let mut max_aux_upper_margin: f32 = 0.0;
    let mut main = 0.0;
    let mut prev_margin = f32::NAN;
    let mut min_obstacle = 0;
    let reversed = if backwards { total_main } else { -1.0 };
    let mut obstacle = next_obstacle(
        props.obstacles(),
        max_aux,
        main,
        aux,
        xaxis,
        reversed,
        &mut min_obstacle,
        dpi,
    );

    let mut i = 0;
    while i < childareas.len() {
        let Some(ref b) = childareas[i] else {
            i += 1;
            continue;
        };

        // This is a bit unintuitive, but this is adding the margin for the previous element, not this one.
        if !prev_margin.is_nan() {
            main += prev_margin.max(b.margin.topleft().x);
        }

        // If we hit an obstacle, mark it as an obstacle breakpoint, then jump forward. We ignore margins here, because
        // obstacles are not items, they are edges.
        if main + b.basis > obstacle.0 {
            breaks.push(Linebreak::new(i, obstacle.1, obstacle.0, f32::NAN));
            main = obstacle.1; // Set the axis to the end of the obstacle
            prev_margin = f32::NAN; // Reset the margin, because this counts as an edge instead of an item

            obstacle = next_obstacle(
                props.obstacles(),
                max_aux,
                main,
                aux,
                xaxis,
                reversed,
                &mut min_obstacle,
                dpi,
            );

            // We DO NOT update any other values here, nor do we increment i, because we might hit another obstacle
            // or the end of the line, so we immediately loop around to try again.
            continue;
        }

        // Once we hit the end of the line we mark the true breakpoint.
        if main + b.basis > total_main {
            // If our line was empty, then nothing could fit on it. Because we don't have line-height information, we simply
            // have to use the height of the element we are pushing to the next line.
            let emptyline = if max_aux == 0.0 {
                max_aux = b.aux;

                // Normally, if an obstacle is present on a line, we want to skip it entirely. However, if we can't fit an item on
                // a line that has no obstacle, we have to give up and put it there anyway to prevent an infinite loop.
                if let Some(b) = breaks.last() {
                    b.lineheight >= 0.0
                } else {
                    true
                }
            } else {
                false
            };

            main = 0.0;
            if let Some(b) = breaks.last_mut() {
                b.aux_margin = b.aux_margin.max(max_aux_upper_margin);
            }
            breaks.push(Linebreak::new(i, max_aux, f32::INFINITY, max_aux_margin));
            used_aux += max_aux;
            aux += max_aux + inner;
            max_aux = 0.0;
            max_aux_margin = 0.0;
            max_aux_upper_margin = 0.0;
            linecount += 1;
            obstacle = next_obstacle(
                props.obstacles(),
                max_aux,
                main,
                aux,
                xaxis,
                reversed,
                &mut min_obstacle,
                dpi,
            );

            if !emptyline {
                continue;
            }
        }

        main += b.basis;
        prev_margin = b.margin.bottomright().x;
        max_aux_margin = max_aux_margin.max(b.margin.bottomright().y);
        max_aux_upper_margin = max_aux_upper_margin.max(b.margin.topleft().y);
        max_aux = max_aux.max(b.aux);
        i += 1;
    }

    if let Some(b) = breaks.last_mut() {
        b.aux_margin = b.aux_margin.max(max_aux_upper_margin);
    }

    breaks.push(Linebreak::new(
        childareas.len(),
        f32::INFINITY,
        f32::INFINITY,
        f32::NAN,
    ));

    (breaks, linecount, used_aux)
}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Child;
    type Children = im::Vector<Option<Box<dyn Layout<Self::Child>>>>;

    fn stage<'a>(
        props: &Self::Props,
        outer_area: AbsRect,
        outer_limits: AbsLimits,
        children: &Self::Children,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        window: &mut crate::component::window::WindowState,
    ) -> Box<dyn Staged + 'a> {
        let myarea = props.area().resolve(window.dpi);
        //let (unsized_x, unsized_y) = super::check_unsized(*myarea);

        let limits = outer_limits + props.limits().resolve(window.dpi);
        let inner_dim = super::limit_dim(super::eval_dim(myarea, outer_area.dim()), limits);
        let outer_safe = nuetralize_unsized(outer_area);

        let xaxis = match props.direction() {
            RowDirection::LeftToRight | RowDirection::RightToLeft => true,
            RowDirection::TopToBottom | RowDirection::BottomToTop => false,
        };

        let mut childareas: im::Vector<Option<ChildCache>> = im::Vector::new();
        let (dpi_main, _) = super::swap_axis(xaxis, window.dpi);
        let (outer_main, _) = super::swap_axis(xaxis, outer_safe.dim().0);

        // We re-use a lot of concepts from flexbox in this calculation. First we acquire the natural size of all child elements.
        for child in children.iter() {
            let imposed = child.as_ref().unwrap().get_props();

            let child_limit = super::apply_limit(inner_dim, limits, *imposed.rlimits());
            let basis = imposed.basis().resolve(dpi_main).resolve(outer_main);

            assert!(
                basis.is_finite(),
                "Basis can be unsized, but never infinite!"
            );

            let inner_area = AbsRect::corners(
                Vec2::zero(),
                if xaxis {
                    Vec2::new(basis, UNSIZED_AXIS)
                } else {
                    Vec2::new(UNSIZED_AXIS, basis)
                },
            );

            let stage = child
                .as_ref()
                .unwrap()
                .stage(inner_area, child_limit, window);

            let (main, aux) = super::swap_axis(xaxis, stage.get_area().dim().0);

            let mut cache = ChildCache {
                basis,
                grow: imposed.grow(),
                shrink: imposed.shrink(),
                aux,
                margin: imposed.margin().resolve(window.dpi) * outer_safe,
                limits: child_limit,
            };
            if cache.basis == UNSIZED_AXIS {
                cache.basis = main;
            }

            // Swap the margin axis if necessary
            if !xaxis {
                std::mem::swap(&mut cache.margin.topleft().x, &mut cache.margin.topleft().y);
                std::mem::swap(
                    &mut cache.margin.bottomright().x,
                    &mut cache.margin.bottomright().y,
                );
            }

            childareas.push_back(Some(cache));
        }

        // This fold calculates the maximum size of the main axis, followed by the off-axis, followed
        // by carrying the previous margin amount from the main axis so it can be collapsed properly.
        // Note that margins only ever apply between elements, not edges, so we completely ignore the
        // off-axis margin, as this calculation assumes there is only 1 line of items, and the off-axis
        // margin doesn't apply until there are linebreaks.
        let fold = VectorFold::new(
            |prev: &(f32, f32, f32), n: &Option<ChildCache>| -> (f32, f32, f32) {
                let cache = n.as_ref().unwrap();
                (
                    cache.basis + prev.0 + merge_margin(prev.2, cache.margin.topleft().x),
                    cache.aux.max(prev.1),
                    cache.margin.bottomright().x,
                )
            },
        );

        let (_, (used_main, used_aux, _)) =
            fold.call(fold.init(), &(0.0, 0.0, f32::NAN), &childareas);

        let evaluated_area = {
            let (used_x, used_y) = super::swap_axis(
                xaxis,
                Vec2 {
                    x: used_main,
                    y: used_aux,
                },
            );
            let area = map_unsized_area(myarea, Vec2::new(used_x, used_y));

            // No need to cap this because unsized axis have now been resolved
            super::limit_area(area * outer_safe, limits)
        };

        let (unsized_x, unsized_y) = check_unsized_abs(outer_area.bottomright());

        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();

        if (unsized_x && xaxis) || (unsized_y && !xaxis) {
            // If we are evaluating our staged area along the main axis, no further calculations can be done
            return Box::new(Concrete {
                area: evaluated_area,
                renderable: None,
                rtree: rtree::Node::new(evaluated_area, Some(props.zindex()), nodes, id, window),
                children: staging,
            });
        }

        let (total_main, total_aux) = super::swap_axis(xaxis, inner_dim.0);
        // If we need to do wrapping, we do this first, before calculating anything else.
        let (breaks, linecount, used_aux) = if props.wrap() {
            // Anything other than `start` for main-axis justification causes problems if there are any obstacles we need to
            // flow around. To make our first wrapping guess, we simply assume there is only one line when choosing our starting location.

            let r = wrap_line(
                &childareas,
                props,
                xaxis,
                total_main,
                total_aux,
                used_aux,
                1,
                props.align(),
                props.direction() == RowDirection::BottomToTop
                    || props.direction() == RowDirection::RightToLeft,
                window.dpi,
            );

            if !props.obstacles().is_empty() && props.align() != FlexJustify::Start {
                // If there were obstacles and multiple rows, our initial guess was probably wrong, so rewrap until we converge
                let mut used_aux = used_aux;
                let mut prev = (SmallVec::new(), 1, used_aux);
                let mut linecount = 0;

                // Given the linecount and how we are arranging the rows, figure out the correct initial height
                while linecount != prev.1 || (used_aux - prev.2) > 0.001 {
                    linecount = prev.1;
                    used_aux = prev.2;
                    prev = wrap_line(
                        &childareas,
                        props,
                        xaxis,
                        total_main,
                        total_aux,
                        prev.2,
                        prev.1,
                        props.align(),
                        props.direction() == RowDirection::BottomToTop
                            || props.direction() == RowDirection::RightToLeft,
                        window.dpi,
                    );
                }

                prev
            } else {
                (r.0, r.1, used_aux)
            }
        } else {
            let mut breaks = SmallVec::<[Linebreak; 10]>::new();
            breaks.push(Linebreak::new(
                childareas.len(),
                f32::INFINITY,
                f32::INFINITY,
                f32::NAN,
            ));
            (breaks, 1, used_aux)
        };

        // Now we calculate the outer spacing (at the start and end) vs the inner spacing.
        let (mut aux, inner_aux) =
            justify_inner_outer(props.align(), total_aux, used_aux, linecount);

        let mut main = 0.0;
        let mut curindex = 0;
        let mut prev_margin = f32::NAN;

        // Now we go through each line and apply flex sizing along the main axis.
        for indice in 0..breaks.len() {
            let b = &breaks[indice];
            let mut totalgrow = 0.0;
            let mut totalshrink = 0.0;
            let mut used = 0.0;

            // Gather the total basis, grow and shrink values
            for i in curindex..b.index {
                let Some(a) = &childareas[i] else {
                    continue;
                };
                totalgrow += a.grow;
                totalshrink += a.shrink;
                used += a.basis;
            }

            // Get the total length of this span, and if necessary, find the line height by scanning ahead.
            let (total_span, max_aux) = if b.skip.is_finite() {
                let mut max_aux = breaks.last().unwrap().lineheight;
                for j in indice..breaks.len() {
                    if breaks[j].skip.is_infinite() {
                        max_aux = breaks[j].lineheight;
                        break;
                    }
                }

                (b.skip - main, max_aux)
            } else {
                (total_main - main, b.lineheight)
            };

            let diff = total_span - used;
            let ratio = if diff > 0.0 {
                if totalgrow != 0.0 {
                    diff / totalgrow
                } else {
                    0.0
                }
            } else if totalshrink != 0.0 {
                diff / totalshrink
            } else {
                0.0
            };

            for i in curindex..b.index {
                if let Some(child) = &mut childareas[i] {
                    child.basis += ratio * (if diff > 0.0 { child.grow } else { child.shrink });
                }
            }

            let (outer, inner) = justify_inner_outer(
                props.justify(),
                total_span,
                used,
                (b.index - curindex) as i32,
            );
            main += outer;

            // Construct the final area rectangle for each child
            for i in curindex..b.index {
                let Some(c) = &childareas[i] else {
                    continue;
                };

                // Apply our margin first
                if !prev_margin.is_nan() {
                    main += prev_margin.max(c.margin.topleft().x);
                }
                prev_margin = c.margin.bottomright().x;

                // If we're growing backwards, we flip along the main axis (but not the aux axis)
                let mut area = if props.direction() == RowDirection::RightToLeft
                    || props.direction() == RowDirection::BottomToTop
                {
                    AbsRect::new(
                        total_main - main,
                        aux,
                        total_main - (main + c.basis),
                        aux + max_aux,
                    )
                } else {
                    AbsRect::new(main, aux, main + c.basis, aux + max_aux)
                };

                area = cap_unsized(area);
                area.set_topleft(Vec2::min_by_component(area.topleft(), area.bottomright()));
                // If our axis is swapped, swap the rectangle axis
                if !xaxis {
                    std::mem::swap(&mut area.topleft().x, &mut area.topleft().y);
                    std::mem::swap(&mut area.bottomright().x, &mut area.bottomright().y);
                }

                let stage = children[i]
                    .as_ref()
                    .unwrap()
                    .stage(area, c.limits + limits, window);
                if let Some(node) = stage.get_rtree().upgrade() {
                    nodes.push_back(Some(node));
                }
                staging.push_back(Some(stage));

                main += c.basis + inner;
            }

            if b.skip.is_finite() {
                main = b.lineheight;
            } else {
                main = 0.0;
                aux += b.lineheight + inner_aux;

                if !b.aux_margin.is_nan() {
                    aux += b.aux_margin;
                }
            }
            prev_margin = f32::NAN;
            curindex = b.index;
        }

        Box::new(Concrete {
            area: evaluated_area,
            renderable,
            rtree: rtree::Node::new(evaluated_area, Some(props.zindex()), nodes, id, window),
            children: staging,
        })
    }
}
