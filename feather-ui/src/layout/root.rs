// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::{Desc, Layout, Renderable, Staged, base};
use crate::{PxDim, PxRect};
use std::rc::Rc;

// The root node represents some area on the screen that contains a feather layout. Later this will turn
// into an absolute bounding volume. There can be multiple root nodes, each mapping to a different window.
pub trait Prop {
    fn dim(&self) -> &PxDim;
}

crate::gen_from_to_dyn!(Prop);

impl Prop for PxDim {
    fn dim(&self) -> &PxDim {
        self
    }
}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn base::Empty;
    type Children = Box<dyn Layout<Self::Child>>;

    fn stage<'a>(
        props: &Self::Props,
        _: PxRect,
        _: crate::PxLimits,
        child: &Self::Children,
        _: std::sync::Weak<crate::SourceID>,
        _: Option<Rc<dyn Renderable>>,
        window: &mut crate::component::window::WindowState,
    ) -> Box<dyn Staged + 'a> {
        // We bypass creating our own node here because we can never have a nonzero topleft corner, so our node would be redundant.
        child.stage(
            (*props.dim()).cast_unit().into(),
            Default::default(),
            window,
        )
    }
}
