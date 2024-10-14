use super::Concrete;
use super::Desc;
use super::Layout;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use crate::UPoint;
use crate::URect;
use crate::Vec2;
use dyn_clone::DynClone;
use std::rc::Rc;

#[derive(Clone, Default)]
pub struct Inherited {
    pub center: UPoint,
}

#[derive(Clone, Default)]
pub struct Center {
    pub padding: URect,
    pub zindex: i32,
}

impl Desc for Center {
    type Props = Center;
    type Impose = Inherited;
    type Children<A: DynClone + ?Sized> = im::Vector<Option<Box<dyn Layout<Self::Impose>>>>;

    fn stage<'a>(
        props: &Self::Props,
        true_area: AbsRect,
        parent_pos: Vec2,
        children: &Self::Children<dyn Layout<Self::Impose> + '_>,
        id: std::rc::Weak<crate::SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        let inner_area = {
            let dim = crate::AbsDim(true_area.bottomright - true_area.topleft);
            AbsRect {
                topleft: true_area.topleft + (props.padding.topleft * dim),
                bottomright: (true_area.bottomright - (props.padding.bottomright * dim)).into(),
            }
        };

        let mut inspect: im::Vector<(&Box<dyn Layout<Self::Impose>>, Box<dyn Staged>)> =
            im::Vector::new();

        // First we let the children calculate their own size using our entire padded area
        for child in children.iter() {
            inspect.push_back((
                child.as_ref().unwrap(),
                child
                    .as_ref()
                    .unwrap()
                    .stage(inner_area, true_area.topleft, driver),
            ));
        }

        let mut nodes: im::Vector<Option<Rc<rtree::Node>>> = im::Vector::new();
        let mut staging: im::Vector<Option<Box<dyn Staged>>> = im::Vector::new();

        // Now we check to see if the child ended up centered or if we need to update again
        for child in inspect.into_iter() {
            let props = child.0.get_imposed();
            let target = props.center * inner_area;
            let area = child.1.get_area();
            let center = area.topleft + ((area.bottomright - area.topleft) / 2.0f32);
            let diff = center - target;
            if diff.x < std::f32::EPSILON && diff.y < std::f32::EPSILON {
                if let Some(node) = child.1.get_rtree().upgrade() {
                    nodes.push_back(Some(node));
                }
                staging.push_back(Some(child.1));
            } else {
                let targetarea = area + diff;
                let stage = child.0.stage(targetarea, true_area.topleft, driver);
                if let Some(node) = stage.get_rtree().upgrade() {
                    nodes.push_back(Some(node));
                }
                staging.push_back(Some(stage));
            }
        }

        Box::new(Concrete {
            area: true_area - parent_pos,
            render: renderable
                .map(|x| x.render(true_area, driver))
                .unwrap_or_default(),
            rtree: Rc::new(rtree::Node::new(
                true_area - parent_pos,
                Some(props.zindex),
                nodes,
                id,
            )),
            children: staging,
        })
    }
}
