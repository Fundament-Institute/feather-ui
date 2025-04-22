// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use derive_more::TryFrom;

use crate::{AbsRect, UPoint, URect, ZERO_RECT, ZERO_URECT};
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
        _: &Self::Children,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn crate::outline::Renderable>>,
        _: &crate::DriverState,
    ) -> Box<dyn super::Staged + 'a> {
        outer_area = super::nuetralize_infinity(outer_area);

        Box::new(crate::layout::Concrete {
            area: outer_area,
            render: renderable,
            rtree: Rc::new(crate::rtree::Node::new(
                outer_area,
                None,
                Default::default(),
                id,
            )),
            children: Default::default(),
        })
    }
}

//static SENTINEL: std::sync::LazyLock<std::rc::Rc<()>> =
//  std::sync::LazyLock::new(|| std::rc::Rc::new(()));

pub trait Obstacles {
    fn obstacles(&self) -> &[AbsRect];
}

pub trait ZIndex {
    fn zindex(&self) -> i32;
}

// Currently unused (you can create padding by adjusting the child's area rectangle instead)
//pub trait Padding {
//    fn padding(&self) -> &URect;
//}

// Relative to parent's area
pub trait Margin {
    fn margin(&self) -> &URect;
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
    fn anchor(&self) -> &UPoint;
}

pub trait Limits {
    fn limits(&self) -> &AbsRect;
}

// Relative to parent's area
pub trait RLimits {
    fn rlimits(&self) -> &crate::RelRect;
}

pub trait Order {
    fn order(&self) -> i64;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, TryFrom)]
#[repr(u8)]
pub enum RowDirection {
    #[default]
    LeftToRight,
    RightToLeft,
    TopToBottom,
    BottomToTop,
}

pub trait Direction {
    fn direction(&self) -> RowDirection;
}

impl Limits for URect {
    fn limits(&self) -> &AbsRect {
        &crate::DEFAULT_LIMITS
    }
}

impl RLimits for URect {
    fn rlimits(&self) -> &crate::RelRect {
        &crate::DEFAULT_RLIMITS
    }
}

impl Margin for URect {
    fn margin(&self) -> &URect {
        &ZERO_URECT
    }
}
