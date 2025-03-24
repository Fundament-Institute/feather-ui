// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::collections::HashMap;

pub struct PropBag {
    props: HashMap<PropBagElement, Box<dyn std::any::Any>>,
}

impl PropBag {
    pub fn contains(&self, element: PropBagElement) -> bool {
        self.props.contains_key(&element)
    }
}

macro_rules! gen_prop_bag_base {
    ($prop:path, $name:ident, $setter:ident, $t:ty) => {
        impl $prop for PropBag {
            fn $name(&self) -> &$t {
                self.props
                    .get(&PropBagElement::$name)
                    .expect(concat!("PropBag didn't have ", stringify!($name)))
                    .downcast_ref()
                    .expect(concat!(
                        stringify!($name),
                        " in PropBag was the wrong type!"
                    ))
            }
        }
        impl PropBag {
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
    ($prop:path, $name:ident, $setter:ident, $t:ty) => {
        impl $prop for PropBag {
            fn $name(&self) -> $t {
                (*self
                    .props
                    .get(&PropBagElement::$name)
                    .expect(concat!("PropBag didn't have ", stringify!($name)))
                    .downcast_ref::<$t>()
                    .expect(concat!(
                        stringify!($name),
                        " in PropBag was the wrong type!"
                    )))
                .clone()
            }
        }
        impl PropBag {
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
  ($prop:path, $name:ident, $setter:ident, $ty:ty) => (gen_prop_bag_base!($prop, $name, $setter, $ty););
  ($prop:path, $name:ident, $setter:ident, $ty:ty, $($props:path, $names:ident, $setters:ident, $types:ty),+) => (
    gen_prop_bag_base!($prop, $name, $setter, $ty);
    gen_prop_bag_all!($($props, $names, $setters, $types),+);
  )
}

macro_rules! gen_prop_bag {
  ($($props:path, $names:ident, $setters:ident, $types:ty),+) => (
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    #[allow(non_camel_case_types)]
    #[repr(u16)]
    pub enum PropBagElement {
      domain,
      zindex,
      obstacles,
      $($names),+
    }
    gen_prop_bag_all!($($props, $names, $setters, $types),+);
  )
}
/*
impl crate::layout::base::ZIndex for PropBag {
    fn zindex(&self) -> i32 {
        *self
            .props
            .get(&PropBagElement::zindex)
            .expect("PropBag didn't have zindex")
            .downcast_ref()
            .expect("zindex in PropBag was the wrong type!")
    }
}

impl PropBag {
    pub fn set_zindex(&mut self, v: i32) -> Option<i32> {
        self.props
            .insert(PropBagElement::zindex, Box::new(v))
            .map(|x| *x.downcast().expect("zindex in PropBag was the wrong type!"))
    }
}*/

gen_prop_bag_value_clone!(crate::layout::base::ZIndex, zindex, set_zindex, i32);
gen_prop_bag_value_clone!(
  crate::layout::domain_write::Prop,
  domain,
  set_domain,
  std::rc::Rc<crate::outline::CrossReferenceDomain>
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
  crate::layout::base::Area, area, set_area, crate::URect,
  crate::layout::base::Padding, padding, set_padding, crate::URect,
  crate::layout::base::Margin, margin, set_margin, crate::URect,
  crate::layout::base::Limits, limits, set_limits, crate::URect,
  crate::layout::base::Anchor, anchor, set_anchor, crate::UPoint,
  crate::layout::root::Prop, dim, set_dim, crate::AbsDim
);

impl crate::layout::base::Empty for PropBag {}
impl crate::layout::leaf::Prop for PropBag {}
impl crate::layout::root::Child for PropBag {}
impl crate::layout::simple::Prop for PropBag {}