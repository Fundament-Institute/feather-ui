// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::prop;
use super::Desc;
use super::Layout;
use super::Renderable;
use super::Staged;
use crate::AbsRect;
use dyn_clone::DynClone;
use std::rc::Rc;
use ultraviolet::Vec2;

pub trait Inherited: prop::Area {}

// The root node represents some area on the screen that contains a feather layout. Later this will turn
// into an absolute bounding volume. There can be multiple root nodes, each mapping to a different window.
pub trait Prop {
    fn dim(&self) -> &crate::AbsDim;
}

impl Desc for Rc<dyn Prop> {
    type Props = Rc<dyn Prop>;
    type Impose = Rc<dyn Inherited>;
    type Children<A: DynClone + ?Sized> = Box<dyn Layout<Self::Impose>>;

    fn stage<'a>(
        props: &Self::Props,
        _: AbsRect,
        _: Vec2,
        child: &Self::Children<dyn Layout<Self::Impose> + '_>,
        _: std::rc::Weak<crate::SourceID>,
        _: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        // We bypass creating our own node here because we can never have a nonzero topleft corner, so our node would be redundant.
        child.stage(
            *child.get_imposed().area() * *props.dim(),
            Vec2::zero(),
            driver,
        )
    }
}
