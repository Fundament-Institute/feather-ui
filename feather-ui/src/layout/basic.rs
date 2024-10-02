use super::Concrete;
use super::Desc;
use super::EventList;
use super::Layout;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use crate::URect;
use dyn_clone::DynClone;
use std::rc::Rc;

#[derive(Clone, Default)]
pub struct Inherited {
    pub margin: URect,
    pub area: URect,
}

#[derive(Clone, Default)]
pub struct Basic {
    pub padding: URect,
    pub zindex: i32,
}

impl<AppData: 'static> Desc<AppData> for Basic {
    type Props = Basic;
    type Impose = Inherited;
    type Children<A: DynClone + ?Sized> =
        im::Vector<Option<Box<dyn Layout<Self::Impose, AppData>>>>;

    fn stage<'a>(
        props: &Self::Props,
        area: AbsRect,
        children: &Self::Children<dyn Layout<Self::Impose, AppData> + '_>,
        events: Option<Rc<EventList<AppData>>>,
        renderable: Option<Rc<dyn Renderable<AppData>>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged<AppData> + 'a>
    where
        AppData: 'a,
    {
        let padding = props.padding * area;
        let area = AbsRect {
            topleft: area.topleft + padding.topleft,
            bottomright: (area.bottomright - padding.bottomright).into(),
        };

        let mut staging: im::Vector<Option<Box<dyn Staged<AppData>>>> = im::Vector::new();
        let mut nodes: im::Vector<Option<Rc<rtree::Node<AppData>>>> = im::Vector::new();

        for child in children.iter() {
            let props = child.as_ref().unwrap().get_imposed();
            let margin = props.margin * area;
            let inner = props.area * area;
            let result = AbsRect {
                topleft: inner.topleft + margin.topleft,
                bottomright: (inner.bottomright - margin.bottomright).into(),
            };

            let stage = child.as_ref().unwrap().stage(result, driver);
            if let Some(node) = stage.get_rtree().upgrade() {
                nodes.push_back(Some(node));
            }
            staging.push_back(Some(stage));
        }

        // Our area does not change based on child sizes, so we have no bottom-up resolution step to do here

        Box::new(Concrete {
            area,
            render: renderable
                .map(|x| x.render(area, driver))
                .unwrap_or_default(),
            rtree: Rc::new(rtree::Node::new(area, Some(props.zindex), nodes, events)),
            children: staging,
        })
    }
}
