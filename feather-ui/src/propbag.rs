// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>
use std::collections::HashMap;

pub struct PropBag {
    props: HashMap<PropBagElement, Box<dyn std::any::Any>>,
}

#[allow(dead_code)]
impl PropBag {
    pub fn contains(&self, element: PropBagElement) -> bool {
        self.props.contains_key(&element)
    }
    fn get_value<T: Copy + 'static>(&self, e: PropBagElement) -> T {
        *self.props.get(&e).unwrap().downcast_ref::<T>().unwrap()
    }
    fn set_value<T: Copy + 'static>(&mut self, e: PropBagElement, v: T) -> Option<T> {
        self.props
            .insert(e, Box::new(v))
            .map(|x| *x.downcast().unwrap())
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
    gen_prop_bag_all!($($props, $names, $setters, $types),+);
  )
}

gen_prop_bag_value_clone!(crate::layout::base::Order, order, set_order, i64);
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
  crate::layout::base::Area, area, set_area, crate::URect,
  crate::layout::base::Padding, padding, set_padding, crate::URect,
  crate::layout::base::Margin, margin, set_margin, crate::URect,
  crate::layout::base::Limits, limits, set_limits, crate::URect,
  crate::layout::base::Anchor, anchor, set_anchor, crate::UPoint,
  crate::layout::root::Prop, dim, set_dim, crate::AbsDim
);

impl crate::layout::base::Empty for PropBag {}
impl crate::layout::leaf::Prop for PropBag {}
impl crate::layout::simple::Prop for PropBag {}

impl crate::layout::flex::Prop for PropBag {
    fn direction(&self) -> crate::layout::flex::FlexDirection {
        self.get_value(PropBagElement::direction)
    }

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
