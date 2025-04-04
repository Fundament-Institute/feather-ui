// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::base;
use super::Desc;
use super::LayoutWrap;
use super::Renderable;
use super::Staged;
use crate::AbsDim;
use crate::AbsRect;
use std::rc::Rc;
use ultraviolet::Vec2;

// The root node represents some area on the screen that contains a feather layout. Later this will turn
// into an absolute bounding volume. There can be multiple root nodes, each mapping to a different window.
pub trait Prop {
    fn dim(&self) -> &crate::AbsDim;
}

crate::gen_from_to_dyn!(Prop);

impl Prop for AbsDim {
    fn dim(&self) -> &crate::AbsDim {
        self
    }
}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn base::Empty;
    type Children = Box<dyn LayoutWrap<Self::Child>>;

    fn stage<'a>(
        props: &Self::Props,
        _: AbsRect,
        child: &Self::Children,
        _: std::rc::Weak<crate::SourceID>,
        _: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        // We bypass creating our own node here because we can never have a nonzero topleft corner, so our node would be redundant.
        child.stage(
            AbsRect {
                topleft: Vec2::zero(),
                bottomright: (*props.dim()).into(),
            },
            driver,
        )
    }
}
