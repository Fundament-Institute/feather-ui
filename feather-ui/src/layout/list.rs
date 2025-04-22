// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::base;
use super::base::Empty;
use super::merge_limits;
use super::Concrete;
use super::Desc;
use super::LayoutWrap;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use crate::SourceID;
use crate::ZERO_POINT;
use base::RowDirection;
use std::rc::Rc;

pub trait Prop: base::Area + base::Limits + base::Direction {}

crate::gen_from_to_dyn!(Prop);

pub trait Child: base::RLimits + base::Margin {}

crate::gen_from_to_dyn!(Child);

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Child;
    // TODO: Make a sorted im::Vector that uses base::Order to order inserted children.
    type Children = im::Vector<Option<Box<dyn LayoutWrap<dyn Empty>>>>;

    fn stage<'a>(
        props: &Self::Props,
        outer_area: AbsRect,
        children: &Self::Children,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        // TODO: make insertion efficient by creating a RRB tree of list layout subnodes, in a similar manner to the r-tree nodes.

        let mut myarea = outer_area;
        myarea = super::nuetralize_infinity(myarea);
        //let limits = merge_limits(outer_limits, *props.limits());
        let limits = *props.limits();
        myarea = *props.area() * myarea;

        let mut areas: im::Vector<Option<AbsRect>> = im::Vector::new();
        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();
        if myarea.bottomright.x.is_infinite() || myarea.bottomright.y.is_infinite() {
            let inner_dim = super::limit_dim(myarea.dim(), limits);
            let inner_area = AbsRect::from(inner_dim);
            let mut bottomright = ZERO_POINT;

            for child in children.iter() {
                let stage = child.as_ref().unwrap().stage(inner_area, driver);
                bottomright = bottomright.max_by_component(stage.get_area().bottomright);
                areas.push_back(stage.get_area().into());
            }

            if myarea.bottomright.x.is_infinite() {
                myarea.bottomright.x = myarea.topleft.x + bottomright.x;
            }
            if myarea.bottomright.y.is_infinite() {
                myarea.bottomright.y = myarea.topleft.y + bottomright.y;
            }
        }

        myarea = super::limit_area(myarea, limits);

        let dir = props.direction();
        let xaxis = match dir {
            RowDirection::LeftToRight | RowDirection::RightToLeft => true,
            RowDirection::TopToBottom | RowDirection::BottomToTop => false,
        };

        // If our parent is asking for a size estimation along the expansion axis, no need to layout the children
        // TODO: Double check this assumption is true
        if (outer_area.bottomright.x.is_infinite() && xaxis)
            || (outer_area.bottomright.y.is_infinite() && !xaxis)
        {
            return Box::new(Concrete {
                area: myarea,
                render: renderable,
                rtree: Rc::new(rtree::Node::new(myarea, None, nodes, id)),
                children: staging,
            });
        }

        let mut cur = match dir {
            RowDirection::LeftToRight | RowDirection::TopToBottom => myarea.topleft,
            RowDirection::RightToLeft | RowDirection::BottomToTop => myarea.bottomright,
        };

        for (i, child) in children.iter().enumerate() {
            let inner_dim = areas[i].unwrap().dim().into();
            let inner_area = match dir {
                RowDirection::LeftToRight | RowDirection::TopToBottom => AbsRect {
                    topleft: cur,
                    bottomright: cur + inner_dim,
                },
                RowDirection::RightToLeft | RowDirection::BottomToTop => AbsRect {
                    topleft: cur - inner_dim,
                    bottomright: cur,
                },
            };

            let stage = child.as_ref().unwrap().stage(inner_area, driver);
            if let Some(node) = stage.get_rtree().upgrade() {
                nodes.push_back(Some(node));
            }
            staging.push_back(Some(stage));

            match dir {
                RowDirection::LeftToRight => cur.x += inner_dim.x,
                RowDirection::RightToLeft => cur.x -= inner_dim.x,
                RowDirection::TopToBottom => cur.y += inner_dim.y,
                RowDirection::BottomToTop => cur.y -= inner_dim.y,
            };
        }

        Box::new(Concrete {
            area: myarea,
            render: renderable,
            rtree: Rc::new(rtree::Node::new(myarea, None, nodes, id)),
            children: staging,
        })
    }
}
