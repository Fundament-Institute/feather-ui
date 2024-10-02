use super::Concrete;
use super::Desc;
use super::EventList;
use super::Layout;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use dyn_clone::DynClone;
use std::marker::PhantomData;
use std::rc::Rc;

// An Empty layout is used in components that only contain the properties of their parents
#[derive(Clone, Default)]
pub struct Empty {}

impl<AppData> Desc<AppData> for Empty {
    type Props = ();
    type Impose = ();
    type Children<A: DynClone + ?Sized> = PhantomData<dyn Layout<Self::Impose, AppData>>;

    fn stage<'a>(
        _: &Self::Props,
        area: AbsRect,
        _: &Self::Children<dyn Layout<Self::Impose, AppData> + '_>,
        events: Option<Rc<EventList<AppData>>>,
        renderable: Option<Rc<dyn Renderable<AppData>>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged<AppData> + 'a>
    where
        AppData: 'a,
    {
        // While we have no children or layouts, components using None often render things or handle events
        Box::new(Concrete {
            area,
            render: renderable
                .map(|x| x.render(area, driver))
                .unwrap_or_default(),
            rtree: Rc::new(rtree::Node::new(area, None, Default::default(), events)),
            children: Default::default(),
        })
    }
}
