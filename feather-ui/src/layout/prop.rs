use crate::{AbsRect, UPoint, URect};

pub trait Empty {}

impl Empty for () {}

const SENTINEL: std::rc::Rc<()> = std::rc::Rc::new(());

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

pub trait Anchor {
    fn anchor(&self) -> &UPoint;
}

pub trait Limits {
    fn limits(&self) -> &URect;
}
