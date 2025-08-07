// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use ultraviolet::Vec2;

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
        id: std::sync::Weak<SourceID>,
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
            renderable,
            rtree: rtree::Node::new(evaluated_area, None, Default::default(), id, window),
            children: Default::default(),
            layer: None,
        })
    }
}

/// A sized leaf is one with inherent size, like an image. This is used to preserve aspect ratio when
/// encounting an unsized axis. This must be provided in pixels.

#[derive_where::derive_where(Clone)]
pub struct Sized<T> {
    pub id: std::sync::Weak<SourceID>,
    pub props: Rc<T>,
    pub size: Vec2,
    pub renderable: Option<Rc<dyn Renderable>>,
}

impl<T: Padded> Layout<T> for Sized<T> {
    fn get_props(&self) -> &T {
        &self.props
    }
    fn stage<'a>(
        &self,
        outer_area: AbsRect,
        outer_limits: crate::AbsLimits,
        window: &mut crate::component::window::WindowState,
    ) -> Box<dyn super::Staged + 'a> {
        let limits = outer_limits + self.props.limits().resolve(window.dpi);
        let padding = self.props.padding().resolve(window.dpi);
        let area = self.props.area().resolve(window.dpi);
        let aspect_ratio = self.size.x / self.size.y; // Will be NAN if both are 0, which disables any attempt to preserve aspect ratio

        // The way we handle unsized here is different from how we normally handle it. If both axes are unsized, we
        // simply set the area to the internal size. If only one axis is unsized, we stretch it to maintain an aspect
        // ratio relative to the size of the other axis.
        let (unsized_x, unsized_y) = super::check_unsized(area);
        let outer_area = super::nuetralize_unsized(outer_area);
        let mapped_area = match (unsized_x, unsized_y, aspect_ratio.is_nan()) {
            (true, false, false) => {
                let mut presize = map_unsized_area(area, ZERO_POINT) * outer_area;
                let adjust = presize.dim().0.y * aspect_ratio;
                let v = presize.0.as_array_mut();
                v[2] += adjust;
                presize
            }
            (false, true, false) => {
                let mut presize = map_unsized_area(area, ZERO_POINT) * outer_area;
                // Be careful, the aspect ratio here is being divided instead of multiplied
                let adjust = presize.dim().0.x / aspect_ratio;
                let v = presize.0.as_array_mut();
                v[3] += adjust;
                presize
            }
            _ => {
                map_unsized_area(area, self.size + padding.topleft() + padding.bottomright())
                    * outer_area
            }
        };

        let evaluated_area = super::limit_area(mapped_area, limits);

        let anchor = self.props.anchor().resolve(window.dpi) * evaluated_area.dim();
        let evaluated_area = evaluated_area - anchor;

        Box::new(Concrete {
            area: evaluated_area,
            renderable: self.renderable.clone(),
            rtree: rtree::Node::new(
                evaluated_area,
                None,
                Default::default(),
                self.id.clone(),
                window,
            ),
            children: Default::default(),
            layer: None,
        })
    }
}
