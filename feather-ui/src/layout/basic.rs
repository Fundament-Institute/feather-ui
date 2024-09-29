use super::Concrete;
use super::Desc;
use super::Layout;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use crate::URect;
use dyn_clone::DynClone;
use std::rc::Rc;

#[derive(Clone)]
pub struct Inherited {
    margin: URect,
    area: URect,
    zindex: i32,
}

#[derive(Clone)]
pub struct Basic {
    padding: URect,
}

impl<AppData: 'static> Desc<AppData> for Basic {
    type Props = Basic;
    type Impose = Inherited;
    type Children<A: DynClone + ?Sized> = im::Vector<Box<dyn Layout<Self::Impose, AppData>>>;

    fn stage(
        props: &Self::Props,
        area: AbsRect,
        children: &Self::Children<dyn Layout<Self::Impose, AppData>>,
    ) -> Box<dyn Staged<AppData>> {
        let padding = props.padding * area;
        let area = AbsRect {
            topleft: area.topleft + padding.topleft,
            bottomright: (area.bottomright - padding.bottomright).into(),
        };

        let mut staging: im::Vector<Box<dyn Staged<AppData>>> = im::Vector::new();
        let mut nodes: im::Vector<Rc<rtree::Node<AppData>>> = im::Vector::new();

        for child in children.iter() {
            let props = child.get_imposed();
            let margin = props.margin * area;
            let inner = props.area * area;
            let result = AbsRect {
                topleft: inner.topleft + margin.topleft,
                bottomright: (inner.bottomright - margin.bottomright).into(),
            };

            let stage = child.stage(result);
            if let Some(node) = stage.get_rtree().upgrade() {
                nodes.push_back(node);
            }
            staging.push_back(stage);
        }

        // Our area does not change based on child sizes, so we have no bottom-up resolution step to do here

        Box::new(Concrete {
            area,
            render: im::Vector::new(),
            rtree: Rc::new(rtree::Node::new(area, None, nodes, None)),
            children: staging,
        })
    }
}
