// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::reactive::{ConstSignal, DynSignal, MutableSignal, SignalZip, ToSignal, zip_pair};
use crate::{DPerimeter, DPoint, DRect, Limited, PxRect, UPerimeter};
use std::rc::Rc;

#[macro_export]
macro_rules! gen_dyn_prop {
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
impl Limits for () {}
impl Margin for () {}
impl Order for () {}
//impl crate::layout::list::Child for () {}

impl<T: Empty> Empty for Rc<T> {}

impl Empty for DRect {}

gen_dyn_prop!(Empty);

impl crate::layout::Desc for dyn Empty {
    type Props = dyn Empty;
    type Child = dyn Empty;
    type Children = ();
    type Provider = dyn crate::reactive::SignalProvider<Item = (PxRect, crate::RelDim)>;
    type Staging = MutableSignal<crate::RelDim>;

    fn presize(
        _: &Self::Props,
        bounds: DynSignal<crate::PxLimits>,
        dpi: MutableSignal<crate::RelDim>,
        _: DynSignal<Self::Children>,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        (bounds.map(|x| PxRect::zero().limit(*x)).into_dyn(), dpi)
    }

    fn size(
        _: &Self::Props,
        dim: DynSignal<crate::UnsizedDim>,
        bounds: DynSignal<crate::PxLimits>,
        data: Self::Staging,
    ) -> (DynSignal<PxRect>, Self::Staging) {
        (
            zip_pair(dim, bounds, |d, b| PxRect::zero().limit(b.to_bounds(*d))).into_dyn(),
            data,
        )
    }

    fn stage<T: crate::render::Prerender + 'static>(
        _: &Self::Props,
        offset: DynSignal<crate::PxPoint>,
        area: DynSignal<PxRect>,
        renderable: Option<T>,
        defer: Option<super::DeferMachine<Self::Provider>>,
        data: Self::Staging,
    ) -> Rc<crate::rtree::Node> {
        let final_area = (area + offset).into_dyn();

        super::resolve_defer_machine(
            crate::rtree::Node::new(
                final_area.clone(),
                None,
                None,
                Some(Box::new(crate::layout::Concrete::new(
                    renderable.as_ref(),
                    final_area.clone(),
                ))),
            ),
            &defer,
            (final_area, data).zip().value().into_dyn(),
        )
    }
}

pub trait Obstacles {
    fn obstacles(&self) -> DynSignal<&[UPerimeter]>;
}

pub trait ZIndex {
    fn zindex(&self) -> DynSignal<i32> {
        0.to_signal().into()
    }
}

// Padding is used so an element's actual area can be larger than the area it
// draws children inside (like text).
pub trait Padding {
    fn padding(&self) -> DynSignal<UPerimeter> {
        ConstSignal::new(crate::ZERO_PERIMETER).into()
    }
}

// Relative to parent's area, but only ever used to determine spacing between
// child elements.
pub trait Margin {
    fn margin(&self) -> DynSignal<DPerimeter> {
        ConstSignal::new(crate::ZERO_DPERIMETER).into()
    }
}

// Relative to child's assigned area (outer area)
pub trait Area {
    fn area(&self) -> DynSignal<DRect>;
}

gen_dyn_prop!(Area);

// Relative to child's evaluated area (inner area)
pub trait Anchor {
    fn anchor(&self) -> DynSignal<DPoint> {
        ConstSignal::new(crate::ZERO_DPOINT).into()
    }
}

pub trait Limits {
    fn limits(&self) -> DynSignal<crate::DLimits> {
        ConstSignal::new(crate::DEFAULT_DLIMITS).into()
    }
}

pub trait Order {
    fn order(&self) -> DynSignal<i64> {
        0.to_signal().into()
    }
}

pub trait Direction {
    fn direction(&self) -> DynSignal<crate::RowDirection> {
        ConstSignal::new(crate::RowDirection::LeftToRight).into()
    }
}

impl Area for DRect {
    fn area(&self) -> DynSignal<DRect> {
        ConstSignal::new(*self).into() // TODO: This doesn't make sense
    }
}
impl Padding for DRect {}
impl ZIndex for DRect {}
impl Anchor for DRect {}
impl Limits for DRect {}

pub trait TextEdit {
    fn textedit(&self) -> &crate::text::EditView;
}
