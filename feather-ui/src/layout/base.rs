// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::{AbsRect, UPoint, URect, ZERO_RECT, ZERO_UPOINT, ZERO_URECT};
use std::rc::Rc;

#[macro_export]
macro_rules! gen_from_to_dyn {
    ($idx:ident) => {
        impl<'a, T: $idx + 'static> From<&'a T> for &'a (dyn $idx + 'static) {
            fn from(value: &'a T) -> Self {
                return value;
            }
        }
    };
}

pub trait Empty {}

impl Empty for () {}
impl RLimits for () {}
impl Margin for () {}
impl Order for () {}
impl crate::layout::fixed::Child for () {}
impl crate::layout::list::Child for () {}

impl<T: Empty> Empty for Rc<T> {}

impl Empty for URect {}

gen_from_to_dyn!(Empty);

impl crate::layout::Desc for dyn Empty {
    type Props = dyn Empty;
    type Child = dyn Empty;
    type Children = ();

    fn stage<'a>(
        _: &Self::Props,
        mut outer_area: AbsRect,
        outer_limits: crate::AbsLimits,
        _: &Self::Children,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn crate::outline::Renderable>>,
        _: &crate::DriverState,
    ) -> Box<dyn super::Staged + 'a> {
        outer_area = super::nuetralize_unsized(outer_area);
        outer_area = super::limit_area(outer_area, outer_limits);

        Box::new(crate::layout::Concrete::new(
            renderable,
            outer_area,
            Rc::new(crate::rtree::Node::new(
                outer_area,
                None,
                Default::default(),
                id,
            )),
            Default::default(),
        ))
    }
}

//static SENTINEL: std::sync::LazyLock<std::rc::Rc<()>> =
//  std::sync::LazyLock::new(|| std::rc::Rc::new(()));

pub trait Obstacles {
    fn obstacles(&self) -> &[AbsRect];
}

pub trait ZIndex {
    fn zindex(&self) -> i32 {
        0
    }
}

// Padding is used so an element's actual area can be larger than the area it draws children inside (like text).
pub trait Padding {
    fn padding(&self) -> &AbsRect {
        &ZERO_RECT
    }
}

impl Padding for URect {}

// Relative to parent's area, but only ever used to determine spacing between child elements.
pub trait Margin {
    fn margin(&self) -> &URect {
        &ZERO_URECT
    }
}

// Relative to child's assigned area (outer area)
pub trait Area {
    fn area(&self) -> &URect;
}

impl Area for URect {
    fn area(&self) -> &URect {
        self
    }
}

gen_from_to_dyn!(Area);

// Relative to child's evaluated area (inner area)
pub trait Anchor {
    fn anchor(&self) -> &UPoint {
        &ZERO_UPOINT
    }
}

impl Anchor for URect {}

pub trait Limits {
    fn limits(&self) -> &crate::AbsLimits {
        &crate::DEFAULT_LIMITS
    }
}

// Relative to parent's area
pub trait RLimits {
    fn rlimits(&self) -> &crate::RelLimits {
        &crate::DEFAULT_RLIMITS
    }
}

pub trait Order {
    fn order(&self) -> i64 {
        0
    }
}

pub trait Direction {
    fn direction(&self) -> crate::RowDirection {
        crate::RowDirection::LeftToRight
    }
}

impl Limits for URect {}
impl RLimits for URect {}
