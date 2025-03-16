use crate::{AbsRect, UPoint, URect};
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

//static SENTINEL: std::sync::LazyLock<std::rc::Rc<()>> =
//  std::sync::LazyLock::new(|| std::rc::Rc::new(()));

pub trait Obstacles {
    fn obstacles(&self) -> &[AbsRect];
}

pub trait ZIndex {
    fn zindex(&self) -> i32;
}

pub trait Padding {
    fn padding(&self) -> &URect;
}

pub trait Margin {
    fn margin(&self) -> &URect;
}

pub trait Area {
    fn area(&self) -> &URect;
}

impl Area for URect {
    fn area(&self) -> &URect {
        self
    }
}

gen_from_to_dyn!(Area);

pub trait Anchor {
    fn anchor(&self) -> &UPoint;
}

pub trait Limits {
    fn limits(&self) -> &URect;
}
