// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::base::Empty;
use super::{Concrete, Desc, Layout, Renderable, Staged, base, map_unsized_area};
use crate::{AbsRect, DRect, SourceID, ZERO_POINT, rtree};
use std::marker::PhantomData;
use std::rc::Rc;

pub trait Prop: base::Area + base::Limits + base::Anchor {}

crate::gen_from_to_dyn!(Prop);

impl Prop for DRect {}

// Actual leaves do not require padding, but a lot of raw elements do (text, shape, images, etc.)
// This inherits Prop to allow elements to "extract" the padding for the rendering system for
// when it doesn't affect layouts.
pub trait Padded: Prop + base::Padding {}

crate::gen_from_to_dyn!(Padded);

impl Padded for DRect {}

impl Desc for dyn Prop {
    type Props = dyn Prop;
    type Child = dyn Empty;
    type Children = PhantomData<dyn Layout<Self::Child>>;

    fn stage<'a>(
        props: &Self::Props,
        outer_area: AbsRect,
        outer_limits: crate::AbsLimits,
        _: &Self::Children,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        window: &mut crate::component::window::WindowState,
    ) -> Box<dyn Staged + 'a> {
        let limits = outer_limits + props.limits().resolve(window.dpi);
        let evaluated_area = super::limit_area(
            map_unsized_area(props.area().resolve(window.dpi), ZERO_POINT)
                * super::nuetralize_unsized(outer_area),
            limits,
        );

        let anchor = props.anchor().resolve(window.dpi) * evaluated_area.dim();
        let evaluated_area = evaluated_area - anchor;

        Box::new(Concrete {
            area: evaluated_area,
            render: renderable,
            rtree: rtree::Node::new(evaluated_area, None, Default::default(), id, window),
            children: Default::default(),
        })
    }
}
