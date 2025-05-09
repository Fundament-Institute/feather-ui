// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>
use std::collections::HashMap;

use crate::{DEFAULT_LIMITS, DEFAULT_RLIMITS, ZERO_UPOINT, ZERO_URECT};

#[derive(Default)]
pub struct PropBag {
    props: HashMap<PropBagElement, Box<dyn std::any::Any>>,
}

#[allow(dead_code)]
impl PropBag {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn contains(&self, element: PropBagElement) -> bool {
        self.props.contains_key(&element)
    }
    fn get_value<T: Default + Copy + 'static>(&self, e: PropBagElement) -> T {
        if let Some(t) = self.props.get(&e) {
            *t.downcast_ref::<T>().unwrap()
        } else {
            Default::default()
        }
    }
    fn set_value<T: Copy + 'static>(&mut self, e: PropBagElement, v: T) -> Option<T> {
        self.props
            .insert(e, Box::new(v))
            .map(|x| *x.downcast().unwrap())
    }
}

macro_rules! gen_prop_bag_base {
    ($prop:path, $name:ident, $setter:ident, $t:ty, $default:expr) => {
        impl $prop for PropBag {
            fn $name(&self) -> &$t {
                if let Some(v) = self.props.get(&PropBagElement::$name) {
                    v.downcast_ref().expect(concat!(
                        stringify!($name),
                        " in PropBag was the wrong type!"
                    ))
                } else {
                    $default
                }
            }
        }
        impl PropBag {
            #[allow(dead_code)]
            pub fn $setter(&mut self, v: $t) -> Option<$t> {
                self.props
                    .insert(PropBagElement::$name, Box::new(v))
                    .map(|x| {
                        *x.downcast().expect(concat!(
                            stringify!($name),
                            " in PropBag was the wrong type!"
                        ))
                    })
            }
        }
    };
}

macro_rules! gen_prop_bag_value_clone {
    ($prop:path, $name:ident, $setter:ident, $t:ty, $default:expr) => {
        impl $prop for PropBag {
            fn $name(&self) -> $t {
                if let Some(v) = self.props.get(&PropBagElement::$name) {
                    (*v.downcast_ref::<$t>().expect(concat!(
                        stringify!($name),
                        " in PropBag was the wrong type!"
                    )))
                    .clone()
                } else {
                    $default
                }
            }
        }
        impl PropBag {
            #[allow(dead_code)]
            pub fn $setter(&mut self, v: $t) -> Option<$t> {
                self.props
                    .insert(PropBagElement::$name, Box::new(v))
                    .map(|x| {
                        *x.downcast().expect(concat!(
                            stringify!($name),
                            " in PropBag was the wrong type!"
                        ))
                    })
            }
        }
    };
}

macro_rules! gen_prop_bag_all {
  ($prop:path, $name:ident, $setter:ident, $ty:ty, $default:expr) => (gen_prop_bag_base!($prop, $name, $setter, $ty, $default););
  ($prop:path, $name:ident, $setter:ident, $ty:ty, $default:expr, $($props:path, $names:ident, $setters:ident, $types:ty, $defaults:expr),+) => (
    gen_prop_bag_base!($prop, $name, $setter, $ty, $default);
    gen_prop_bag_all!($($props, $names, $setters, $types, $defaults),+);
  )
}

macro_rules! gen_prop_bag {
  ($($props:path, $names:ident, $setters:ident, $types:ty, $defaults:expr),+) => (
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    #[allow(non_camel_case_types)]
    #[repr(u16)]
    pub enum PropBagElement {
      domain,
      zindex,
      obstacles,
      direction,
      wrap,
      justify,
      align,
      order,
      grow,
      shrink,
      basis,
      $($names),+
    }
    gen_prop_bag_all!($($props, $names, $setters, $types, $defaults),+);
  )
}

gen_prop_bag_value_clone!(crate::layout::base::Order, order, set_order, i64, 0);
gen_prop_bag_value_clone!(crate::layout::base::ZIndex, zindex, set_zindex, i32, 0);
gen_prop_bag_value_clone!(
    crate::layout::base::Direction,
    direction,
    set_direction,
    crate::RowDirection,
    crate::RowDirection::LeftToRight
);
gen_prop_bag_value_clone!(
    crate::layout::domain_write::Prop,
    domain,
    set_domain,
    std::rc::Rc<crate::outline::CrossReferenceDomain>,
    panic!("PropBag didn't have domain!")
);

impl crate::layout::base::Obstacles for PropBag {
    fn obstacles(&self) -> &[crate::AbsRect] {
        // We have to be careful here because the actual stored type is a Vec<>, not a slice.
        self.props
            .get(&PropBagElement::obstacles)
            .expect("PropBag didn't have obstacles")
            .downcast_ref::<Vec<crate::AbsRect>>()
            .expect("obstacles in PropBag was the wrong type!")
    }
}

impl PropBag {
    #[allow(dead_code)]
    pub fn set_obstacles(&mut self, v: &[crate::AbsRect]) -> Option<Vec<crate::AbsRect>> {
        self.props
            .insert(PropBagElement::zindex, Box::new(v.to_vec()))
            .map(move |x| {
                *x.downcast()
                    .expect("obstacles in PropBag was the wrong type!")
            })
    }
}

#[rustfmt::skip]
gen_prop_bag!(
  crate::layout::base::Area, area, set_area, crate::URect, panic!("No area set and no default available!"),
  crate::layout::base::Padding, padding, set_padding, crate::AbsRect, &crate::ZERO_RECT,
  crate::layout::base::Margin, margin, set_margin, crate::URect, &ZERO_URECT,
  crate::layout::base::Limits, limits, set_limits, crate::AbsLimits, &DEFAULT_LIMITS,
  crate::layout::base::RLimits, rlimits, set_rlimits, crate::RelLimits, &DEFAULT_RLIMITS,
  crate::layout::base::Anchor, anchor, set_anchor, crate::UPoint, &ZERO_UPOINT,
  crate::layout::root::Prop, dim, set_dim, crate::AbsDim, panic!("No dim set and no default available!")
);

impl crate::layout::base::Empty for PropBag {}
impl crate::layout::leaf::Prop for PropBag {}
impl crate::layout::fixed::Prop for PropBag {}
impl crate::layout::fixed::Child for PropBag {}
impl crate::layout::list::Child for PropBag {}
impl crate::layout::list::Prop for PropBag {}
impl crate::layout::leaf::Padded for PropBag {}

impl crate::layout::flex::Prop for PropBag {
    fn wrap(&self) -> bool {
        self.get_value(PropBagElement::wrap)
    }

    fn justify(&self) -> crate::layout::flex::FlexJustify {
        self.get_value(PropBagElement::justify)
    }

    fn align(&self) -> crate::layout::flex::FlexJustify {
        self.get_value(PropBagElement::align)
    }
}

impl crate::layout::flex::Child for PropBag {
    fn grow(&self) -> f32 {
        self.get_value(PropBagElement::grow)
    }

    fn shrink(&self) -> f32 {
        self.get_value(PropBagElement::shrink)
    }

    fn basis(&self) -> f32 {
        self.get_value(PropBagElement::basis)
    }
}
