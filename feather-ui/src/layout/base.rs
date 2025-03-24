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

impl crate::layout::Desc for dyn Empty {
    type Props = dyn Empty;
    type Child = dyn Empty;
    type Children = ();

    fn stage<'a>(
        _: &Self::Props,
        mut true_area: AbsRect,
        parent_pos: ultraviolet::Vec2,
        _: &Self::Children,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn crate::outline::Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn super::Staged + 'a> {
        if true_area.bottomright.x.is_infinite() {
            true_area.bottomright.x = true_area.topleft.x;
        }
        if true_area.bottomright.y.is_infinite() {
            true_area.bottomright.y = true_area.topleft.y;
        }

        Box::new(crate::layout::Concrete {
            area: true_area - parent_pos,
            render: renderable
                .map(|x| x.render(true_area, driver))
                .unwrap_or_default(),
            rtree: Rc::new(crate::rtree::Node::new(
                true_area - parent_pos,
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

pub trait Order {
    fn order(&self) -> i64;
}
