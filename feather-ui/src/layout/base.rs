// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::{
    DAbsRect, DPoint, DRect, ZERO_DRECT,
    reactive::{DynSignal, MutableSignal, SignalMap, ToSignal, const_signal, zip_pair},
};
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
impl RLimits for () {}
impl Margin for () {}
impl Order for () {}
impl crate::layout::fixed::Child for () {}
//impl crate::layout::list::Child for () {}

impl<T: Empty> Empty for Rc<T> {}

impl Empty for DRect {}

gen_dyn_prop!(Empty);

impl crate::layout::Desc for dyn Empty {
    type Props = dyn Empty;
    type Child = dyn Empty;
    type Children = ();
    type Provider = dyn crate::reactive::SignalProvider<Item = (crate::PxRect, crate::RelDim)>;

    fn stage<'a, T: crate::render::Prerender + 'static>(
        _: &Self::Props,
        predim: DynSignal<crate::PxDim>,
        _: DynSignal<Self::Children>,
        renderable: Option<T>,
        dpi: MutableSignal<crate::RelDim>,
        defer: Option<super::DeferMachine<Self::Provider>>,
    ) -> (DynSignal<crate::PxRect>, super::StageThunk<'a>) {
        let area = predim.map(|dim| crate::PxRect::from(*dim)).into();

        (
            area,
            Box::new(move |offset, final_dim| {
                let final_area =
                    zip_pair(offset, final_dim, |o, dim| crate::Rect::offsetdim(o, dim))
                        .into_dyn_signal();

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
                    crate::reactive::zip(final_area, dpi.clone()).into_dyn_signal(),
                )
            }),
        )
    }
}

pub trait Obstacles {
    fn obstacles(&self) -> DynSignal<&[DAbsRect]>;
}

pub trait ZIndex {
    fn zindex(&self) -> DynSignal<i32> {
        0.to_signal().into()
    }
}

impl ZIndex for DRect {}

// Padding is used so an element's actual area can be larger than the area it
// draws children inside (like text).
pub trait Padding {
    fn padding(&self) -> DynSignal<DAbsRect> {
        const_signal(crate::ZERO_DABSRECT).into()
    }
}

impl Padding for DRect {}

// Relative to parent's area, but only ever used to determine spacing between
// child elements.
pub trait Margin {
    fn margin(&self) -> DynSignal<DRect> {
        const_signal(ZERO_DRECT).into()
    }
}

// Relative to child's assigned area (outer area)
pub trait Area {
    fn area(&self) -> DynSignal<DRect>;
}

impl Area for DRect {
    fn area(&self) -> DynSignal<DRect> {
        const_signal(self.clone()).into() // TODO: This doesn't make sense
    }
}

gen_dyn_prop!(Area);

// Relative to child's evaluated area (inner area)
pub trait Anchor {
    fn anchor(&self) -> DynSignal<DPoint> {
        const_signal(crate::ZERO_DPOINT).into()
    }
}

impl Anchor for DRect {}

pub trait Limits {
    fn limits(&self) -> DynSignal<crate::DLimits> {
        const_signal(crate::DEFAULT_DLIMITS).into()
    }
}

// Relative to parent's area
pub trait RLimits {
    fn rlimits(&self) -> DynSignal<crate::RelLimits> {
        const_signal(crate::DEFAULT_RLIMITS).into()
    }
}

pub trait Order {
    fn order(&self) -> DynSignal<i64> {
        0.to_signal().into()
    }
}

pub trait Direction {
    fn direction(&self) -> DynSignal<crate::RowDirection> {
        const_signal(crate::RowDirection::LeftToRight).into()
    }
}

impl Limits for DRect {}
impl RLimits for DRect {}

pub trait TextEdit {
    fn textedit(&self) -> &crate::text::EditView;
}
