// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2024 Fundament Software SPC <https://fundament.software>

use super::zero_infinity;
use super::Concrete;
use super::Desc;
use super::Layout;
use super::Renderable;
use super::Staged;
use crate::persist::FnPersist2;
use crate::persist::VectorFold;
use crate::rtree;
use crate::AbsRect;
use crate::URect;
use crate::Vec2;
use core::f32;
use dyn_clone::DynClone;
use smallvec::SmallVec;
use std::rc::Rc;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum FlexDirection {
    LeftToRight,
    RightToLeft,
    TopToBottom,
    BottomToTop,
}

impl Default for FlexDirection {
    fn default() -> Self {
        FlexDirection::LeftToRight
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum FlexJustify {
    Start,
    Center,
    End,
    SpaceBetween,
    SpaceAround,
    SpaceFull,
}

impl Default for FlexJustify {
    fn default() -> Self {
        FlexJustify::Start
    }
}

#[derive(Clone, Default)]
pub struct Flex {
    pub direction: FlexDirection,
    pub wrap: bool,
    pub justify: FlexJustify, // Main-axis justification of items inside a single line
    pub align: FlexJustify,   // Off-axis justification of lines inside the flexbox.
    pub zindex: i32,
    pub obstacles: Vec<AbsRect>,
}

#[derive(Clone, Default)]
pub struct Inherited {
    pub order: i64,
    pub grow: f32,
    pub shrink: f32,
    pub basis: f32,
    pub margin: URect,
    pub limits: URect,
}

fn swap_axis(xaxis: bool, v: Vec2) -> (f32, f32) {
    if xaxis {
        (v.x, v.y)
    } else {
        (v.y, v.x)
    }
}

fn next_obstacle(
    obstacles: &Vec<AbsRect>,
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
        if aux_start >= aux + max_aux {
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

    return (f32::INFINITY, 0.0);
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

fn wrap_line(
    childareas: &im::Vector<Option<(Inherited, f32)>>,
    props: &Flex,
    xaxis: bool,
    total_main: f32,
    total_aux: f32,
    used_aux: f32,
    linecount: i32,
    justify: FlexJustify,
    backwards: bool,
) -> (SmallVec<[(usize, f32); 10]>, i32, f32) {
    let mut breaks: SmallVec<[(usize, f32); 10]> = SmallVec::new();

    let (mut aux, inner) = justify_inner_outer(justify, total_aux, used_aux, linecount);

    // Reset linecount and used_aux to 0
    let mut linecount = 0;
    let mut used_aux = 0.0;
    let mut max_aux = 0.0;
    let mut main = 0.0;
    let mut i = 0;
    let mut min_obstacle = 0;
    let reversed = if backwards { total_main } else { -1.0 };
    let mut obstacle = next_obstacle(
        &props.obstacles,
        max_aux,
        main,
        aux,
        xaxis,
        reversed,
        &mut min_obstacle,
    );

    for v in childareas.iter() {
        i += 1;
        let Some(b) = v else {
            continue;
        };
        // We only increment the linecount after we actually assign something to a new line
        if main == 0.0 {
            linecount += 1;
        }
        main += b.0.basis;
        if max_aux < b.1 {
            max_aux = b.1;
        }

        // If we hit an obstacle, mark it as an obstacle breakpoint, then jump forward.
        if main > obstacle.0 {
            breaks.push((i, -obstacle.1));
            main = obstacle.1; // Set the axis to the end of the obstacle

            obstacle = next_obstacle(
                &props.obstacles,
                max_aux,
                main,
                aux,
                xaxis,
                reversed,
                &mut min_obstacle,
            );
        }

        // Once we hit the end of the line (which could happen after navigating an obstacle) we mark the true breakpoint.
        if main > total_main {
            breaks.push((i, max_aux));
            main = 0.0;
            used_aux += max_aux;
            aux += max_aux + inner;
            max_aux = 0.0;

            obstacle = next_obstacle(
                &props.obstacles,
                max_aux,
                main,
                aux,
                xaxis,
                reversed,
                &mut min_obstacle,
            );
        }
    }

    breaks.push((childareas.len(), f32::INFINITY));

    (breaks, linecount, used_aux)
}

impl Desc for Flex {
    type Props = Flex;
    type Impose = Inherited;
    type Children<A: DynClone + ?Sized> = im::Vector<Option<Box<dyn Layout<Self::Impose>>>>;

    fn stage<'a>(
        props: &Self::Props,
        true_area: AbsRect,
        parent_pos: Vec2,
        children: &Self::Children<dyn Layout<Self::Impose> + '_>,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        let mut childareas: im::Vector<Option<(Inherited, f32)>> = im::Vector::new();

        // If we are currently also being evaluated with infinite area, we have to set a few things to zero.
        let dim = crate::AbsDim(zero_infinity(true_area.dim().into()));

        let xaxis = match props.direction {
            FlexDirection::LeftToRight | FlexDirection::RightToLeft => true,
            FlexDirection::TopToBottom | FlexDirection::BottomToTop => false,
        };

        // We re-use a lot of concepts from flexbox in this calculation. First we acquire the natural size of all child elements.
        for child in children.iter() {
            let mut imposed = child.as_ref().unwrap().get_imposed().clone();

            let mut area = AbsRect {
                topleft: true_area.topleft + (imposed.margin.topleft * dim),
                bottomright: (true_area.bottomright - (imposed.margin.bottomright * dim)).into(),
            };

            // If we don't have a basis, we need to set it to infinity
            if xaxis {
                area.bottomright.x = imposed.basis;
                area.bottomright.y = f32::INFINITY;
            } else {
                area.bottomright.x = f32::INFINITY;
                area.bottomright.y = imposed.basis;
            }

            let stage = child
                .as_ref()
                .unwrap()
                .stage(area, true_area.topleft, driver);

            let (main, aux) = swap_axis(xaxis, stage.get_area().dim().0);

            if imposed.basis.is_infinite() {
                imposed.basis = main;
            }

            childareas.push_back(Some((imposed, aux)));
        }

        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();

        let fold = VectorFold::new(
            |basis: &(f32, f32), n: &Option<(Inherited, f32)>| -> (f32, f32) {
                (
                    n.as_ref().unwrap().0.basis + basis.0,
                    n.as_ref().unwrap().1.max(basis.1),
                )
            },
        );

        let (_, (used_main, used_aux)) = fold.call(fold.init(), &(0.0, 0.0), &childareas);

        if (true_area.bottomright.x.is_infinite() && xaxis)
            || (true_area.bottomright.y.is_infinite() && !xaxis)
        {
            let mut area = true_area;
            if xaxis {
                area.bottomright.x = used_main;
            } else {
                area.bottomright.y = used_main;
            }
            // If we are evaluating our staged area along the main axis, no further calculations can be done
            return Box::new(Concrete {
                area: area - parent_pos,
                render: im::Vector::new(),
                rtree: Rc::new(rtree::Node::new(
                    area - parent_pos,
                    Some(props.zindex),
                    nodes,
                    id,
                )),
                children: staging,
            });
        }

        let (total_main, total_aux) = swap_axis(xaxis, dim.0);
        // If we need to do wrapping, we do this first, before calculating anything else.
        let (breaks, linecount, used_aux) = if props.wrap {
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
                props.align,
                props.direction == FlexDirection::BottomToTop
                    || props.direction == FlexDirection::RightToLeft,
            );

            if props.obstacles.len() > 0 && props.align != FlexJustify::Start {
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
                        props.align,
                        props.direction == FlexDirection::BottomToTop
                            || props.direction == FlexDirection::RightToLeft,
                    );
                }

                prev
            } else {
                (r.0, r.1, used_aux)
            }
        } else {
            let mut breaks = SmallVec::<[(usize, f32); 10]>::new();
            breaks.push((childareas.len(), f32::INFINITY));
            (breaks, 1, used_aux)
        };

        // Now we calculate the outer spacing (at the start and end) vs the inner spacing.
        let (mut aux, inner_aux) = justify_inner_outer(props.align, total_aux, used_aux, linecount);

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
                totalgrow += a.0.grow;
                totalshrink += a.0.shrink;
                used += a.0.basis;
            }

            // Get the total length of this span, and if necessary, find the line height by scanning ahead.
            let (total_span, max_aux) = if b.1 < 0.0 {
                let mut max_aux = breaks.last().unwrap().1;
                for j in indice..breaks.len() {
                    if breaks[j].1 >= 0.0 {
                        max_aux = breaks[j].1;
                        break;
                    }
                }

                (-b.1 - main, max_aux)
            } else {
                (total_main - main, b.1)
            };

            let diff = total_span - used;
            let ratio = diff / (if diff > 0.0 { totalgrow } else { totalshrink });

            // TODO: respect min and max limits on dimensions
            for i in curindex..b.0 {
                let Some(a) = &mut childareas[i] else {
                    continue;
                };
                a.0.basis += ratio * (if diff > 0.0 { a.0.grow } else { a.0.shrink });
            }

            let (outer, inner) =
                justify_inner_outer(props.justify, total_span, used, (b.0 - curindex) as i32);
            main += outer;

            // Construct the final area rectangle for each child
            for i in curindex..b.0 {
                let Some(a) = &mut childareas[i] else {
                    continue;
                };

                // If we're growing backwards, we flip along the main axis (but not the aux axis)
                let mut area = if props.direction == FlexDirection::RightToLeft
                    || props.direction == FlexDirection::BottomToTop
                {
                    AbsRect {
                        topleft: Vec2::new(total_main - main, aux),
                        bottomright: Vec2::new(total_main - (main + a.0.basis), aux + max_aux),
                    }
                } else {
                    AbsRect {
                        topleft: Vec2::new(main, aux),
                        bottomright: Vec2::new(main + a.0.basis, aux + max_aux),
                    }
                };

                // If our axis is swapped, swap the rectangle axis
                if !xaxis {
                    std::mem::swap(&mut area.topleft.x, &mut area.topleft.y);
                    std::mem::swap(&mut area.bottomright.x, &mut area.bottomright.y);
                }

                let stage = children[i].as_ref().unwrap().stage(
                    area + true_area.topleft,
                    true_area.topleft,
                    driver,
                );
                if let Some(node) = stage.get_rtree().upgrade() {
                    nodes.push_back(Some(node));
                }
                staging.push_back(Some(stage));

                main += a.0.basis + inner;
            }

            if b.1 < 0.0 {
                main = -b.1;
            } else {
                main = 0.0;
                aux += b.1 + inner_aux;
            }
            curindex = b.0;
        }

        Box::new(Concrete {
            area: true_area - parent_pos,
            render: renderable
                .map(|x| x.render(true_area, driver))
                .unwrap_or_default(),
            rtree: Rc::new(rtree::Node::new(
                true_area - parent_pos,
                Some(props.zindex),
                nodes,
                id,
            )),
            children: staging,
        })
    }
}
