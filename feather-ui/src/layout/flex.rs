// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::base;
use super::zero_unsized;
use super::Concrete;
use super::Desc;
use super::LayoutWrap;
use super::Renderable;
use super::Staged;
use crate::persist::FnPersist2;
use crate::persist::VectorFold;
use crate::rtree;
use crate::AbsRect;
use crate::RowDirection;
use crate::Vec2;
use core::f32;
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

pub trait Prop: base::ZIndex + base::Obstacles + base::Limits + base::Direction {
    fn wrap(&self) -> bool;
    fn justify(&self) -> FlexJustify;
    fn align(&self) -> FlexJustify;
}

crate::gen_from_to_dyn!(Prop);

pub trait Child: base::Margin + base::RLimits + base::Order {
    fn grow(&self) -> f32;
    fn shrink(&self) -> f32;
    fn basis(&self) -> f32;
}

crate::gen_from_to_dyn!(Child);

fn swap_axis(xaxis: bool, v: Vec2) -> (f32, f32) {
    if xaxis {
        (v.x, v.y)
    } else {
        (v.y, v.x)
    }
}

fn next_obstacle(
    obstacles: &[AbsRect],
    max_aux: f32,
    main: f32,
    aux: f32,
    xaxis: bool,
    total_main: f32,
    min: &mut usize,
) -> (f32, f32) {
    // Given our current X/Y position, what is the next obstacle we would run into?
    let mut i = *min;
    while i < obstacles.len() {
        let (mut start, aux_start) = swap_axis(xaxis, obstacles[i].topleft);

        if total_main > 0.0 {
            start = total_main - start;
        }

        // If we've reached an obstacle whose top/left is past the (current known) bottom of this line, we won't match anything.
        if aux_start > aux + max_aux {
            break;
        }

        let (mut end, aux_end) = swap_axis(xaxis, obstacles[i].bottomright);

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

type LineTuple = (usize, f32, f32);

#[derive(Clone)]
struct ChildCache {
    basis: f32,
    grow: f32,
    shrink: f32,
    aux: f32,
    margin: AbsRect,
    limits: AbsRect,
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
) -> (SmallVec<[LineTuple; 10]>, i32, f32) {
    let mut breaks: SmallVec<[LineTuple; 10]> = SmallVec::new();

    let (mut aux, inner) = justify_inner_outer(justify, total_aux, used_aux, linecount);

    // Reset linecount and used_aux
    linecount = 1;
    used_aux = 0.0;
    let mut max_aux = 0.0;
    let mut main = 0.0;
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
    );

    let mut i = 0;
    while i < childareas.len() {
        let Some(ref b) = childareas[i] else {
            i += 1;
            continue;
        };

        // If we hit an obstacle, mark it as an obstacle breakpoint, then jump forward.
        if main + b.basis > obstacle.0 {
            breaks.push((i, obstacle.1, obstacle.0));
            main = obstacle.1; // Set the axis to the end of the obstacle

            obstacle = next_obstacle(
                props.obstacles(),
                max_aux,
                main,
                aux,
                xaxis,
                reversed,
                &mut min_obstacle,
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
                    b.1 >= 0.0
                } else {
                    true
                }
            } else {
                false
            };

            main = 0.0;
            breaks.push((i, max_aux, f32::INFINITY));
            used_aux += max_aux;
            aux += max_aux + inner;
            max_aux = 0.0;
            linecount += 1;
            obstacle = next_obstacle(
                props.obstacles(),
                max_aux,
                main,
                aux,
                xaxis,
                reversed,
                &mut min_obstacle,
            );

            if !emptyline {
                continue;
            }
        }

        main += b.basis;
        if max_aux < b.aux {
            max_aux = b.aux;
        }
        i += 1;
    }

    breaks.push((childareas.len(), f32::INFINITY, f32::INFINITY));

    (breaks, linecount, used_aux)
}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Child;
    type Children = im::Vector<Option<Box<dyn LayoutWrap<Self::Child>>>>;

    fn stage<'a>(
        props: &Self::Props,
        outer_area: AbsRect,
        outer_limits: AbsRect,
        children: &Self::Children,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        let mut childareas: im::Vector<Option<ChildCache>> = im::Vector::new();

        // If we are currently also being evaluated with infinite area, we have to set a few things to zero.
        let outer_dim = zero_unsized(outer_area.dim());

        let xaxis = match props.direction() {
            RowDirection::LeftToRight | RowDirection::RightToLeft => true,
            RowDirection::TopToBottom | RowDirection::BottomToTop => false,
        };

        // We re-use a lot of concepts from flexbox in this calculation. First we acquire the natural size of all child elements.
        for child in children.iter() {
            let imposed = child.as_ref().unwrap().get_imposed();

            let inner_area = AbsRect {
                topleft: Vec2::zero(),
                bottomright: if xaxis {
                    Vec2::new(imposed.basis(), f32::INFINITY)
                } else {
                    Vec2::new(f32::INFINITY, imposed.basis())
                },
            };

            let child_limit = super::eval_limits(*imposed.rlimits(), outer_dim);
            let stage = child
                .as_ref()
                .unwrap()
                .stage(inner_area, child_limit, driver);

            let (main, aux) = swap_axis(xaxis, stage.get_area().dim().0);

            let mut cache = ChildCache {
                basis: imposed.basis(),
                grow: imposed.grow(),
                shrink: imposed.shrink(),
                aux,
                margin: *imposed.margin() * outer_dim,
                limits: child_limit,
            };
            if cache.basis.is_infinite() {
                cache.basis = main;
            }

            // Swap the margin axis if necessary
            if !xaxis {
                std::mem::swap(&mut cache.margin.topleft.x, &mut cache.margin.topleft.y);
                std::mem::swap(
                    &mut cache.margin.bottomright.x,
                    &mut cache.margin.bottomright.y,
                );
            }

            childareas.push_back(Some(cache));
        }

        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();

        // This fold calculates the maximum size of the main axis, followed by the off-axis, followed
        // by carrying the previous margin amount from the main axis so it can be collapsed properly
        let fold = VectorFold::new(
            |prev: &(f32, f32, f32), n: &Option<ChildCache>| -> (f32, f32, f32) {
                let cache = n.as_ref().unwrap();
                (
                    cache.basis + prev.0 + prev.2.max(cache.margin.topleft.x),
                    cache.aux.max(prev.1) + cache.margin.topleft.y + cache.margin.bottomright.y, // off-axis always adds both margins, no collapsing is done yet
                    cache.margin.bottomright.x,
                )
            },
        );

        let (_, (used_main, used_aux, trailing)) =
            fold.call(fold.init(), &(0.0, 0.0, 0.0), &childareas);
        let used_main = used_main + trailing; // Add trailing margin to used space

        if (outer_area.bottomright.x.is_infinite() && xaxis)
            || (outer_area.bottomright.y.is_infinite() && !xaxis)
        {
            let mut area = outer_area;
            if xaxis {
                area.bottomright.x = outer_area.topleft.x + used_main;
            } else {
                area.bottomright.y = outer_area.topleft.y + used_main;
            }
            // If we are evaluating our staged area along the main axis, no further calculations can be done
            return Box::new(Concrete {
                area: area,
                render: None,
                rtree: Rc::new(rtree::Node::new(area, Some(props.zindex()), nodes, id)),
                children: staging,
            });
        }

        let (total_main, total_aux) = swap_axis(xaxis, outer_dim.0);
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
                    );
                }

                prev
            } else {
                (r.0, r.1, used_aux)
            }
        } else {
            let mut breaks = SmallVec::<[LineTuple; 10]>::new();
            breaks.push((childareas.len(), f32::INFINITY, f32::INFINITY));
            (breaks, 1, used_aux)
        };

        // Now we calculate the outer spacing (at the start and end) vs the inner spacing.
        let (mut aux, inner_aux) =
            justify_inner_outer(props.align(), total_aux, used_aux, linecount);

        let mut main = 0.0;
        let mut curindex = 0;
        // Now we go through each line and apply flex sizing along the main axis.
        for indice in 0..breaks.len() {
            let b = &breaks[indice];
            let mut totalgrow = 0.0;
            let mut totalshrink = 0.0;
            let mut used = 0.0;

            // Gather the total basis, grow and shrink values
            for i in curindex..b.0 {
                let Some(a) = &childareas[i] else {
                    continue;
                };
                totalgrow += a.grow;
                totalshrink += a.shrink;
                used += a.basis;
            }

            // Get the total length of this span, and if necessary, find the line height by scanning ahead.
            let (total_span, max_aux) = if b.2.is_finite() {
                let mut max_aux = breaks.last().unwrap().1;
                for j in indice..breaks.len() {
                    if breaks[j].2.is_infinite() {
                        max_aux = breaks[j].1;
                        break;
                    }
                }

                (b.2 - main, max_aux)
            } else {
                (total_main - main, b.1)
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

            // TODO: respect min and max limits on dimensions
            for i in curindex..b.0 {
                let Some(a) = &mut childareas[i] else {
                    continue;
                };
                a.basis += ratio * (if diff > 0.0 { a.grow } else { a.shrink });
            }

            let (outer, inner) =
                justify_inner_outer(props.justify(), total_span, used, (b.0 - curindex) as i32);
            main += outer;

            // Construct the final area rectangle for each child
            for i in curindex..b.0 {
                let Some(c) = &childareas[i] else {
                    continue;
                };

                // If we're growing backwards, we flip along the main axis (but not the aux axis)
                let mut area = if props.direction() == RowDirection::RightToLeft
                    || props.direction() == RowDirection::BottomToTop
                {
                    AbsRect {
                        topleft: Vec2::new(total_main - main, aux),
                        bottomright: Vec2::new(total_main - (main + c.basis), aux + max_aux),
                    }
                } else {
                    AbsRect {
                        topleft: Vec2::new(main, aux),
                        bottomright: Vec2::new(main + c.basis, aux + max_aux),
                    }
                };

                // Because both our margin and are area rect has swapped axis, we can just apply the margin directly, then swap the axis
                area.topleft += c.margin.topleft;
                area.bottomright -= c.margin.bottomright;
                area.topleft = Vec2::min_by_component(area.topleft, area.bottomright);

                // If our axis is swapped, swap the rectangle axis
                if !xaxis {
                    std::mem::swap(&mut area.topleft.x, &mut area.topleft.y);
                    std::mem::swap(&mut area.bottomright.x, &mut area.bottomright.y);
                }

                let stage = children[i].as_ref().unwrap().stage(area, c.limits, driver);
                if let Some(node) = stage.get_rtree().upgrade() {
                    nodes.push_back(Some(node));
                }
                staging.push_back(Some(stage));

                main += c.basis + inner;
            }

            if b.2.is_finite() {
                main = b.1;
            } else {
                main = 0.0;
                aux += b.1 + inner_aux;
            }
            curindex = b.0;
        }

        Box::new(Concrete {
            area: outer_area,
            render: renderable,
            rtree: Rc::new(rtree::Node::new(
                outer_area,
                Some(props.zindex()),
                nodes,
                id,
            )),
            children: staging,
        })
    }
}
