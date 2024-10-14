use super::Concrete;
use super::Desc;
use super::Layout;
use super::Renderable;
use super::Staged;
use crate::rtree;
use crate::AbsRect;
use crate::SourceID;
use dyn_clone::DynClone;
use std::marker::PhantomData;
use std::rc::Rc;

// An Empty layout is used in outlines that only contain the properties of their parents
#[derive(Clone, Default)]
pub struct Empty {}

impl Desc for Empty {
    type Props = ();
    type Impose = ();
    type Children<A: DynClone + ?Sized> = PhantomData<dyn Layout<Self::Impose>>;

    fn stage<'a>(
        _: &Self::Props,
        area: AbsRect,
        _: &Self::Children<dyn Layout<Self::Impose> + '_>,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &crate::DriverState,
    ) -> Box<dyn Staged + 'a> {
        // While we have no children or layouts, outlines using None often render things or handle events
        Box::new(Concrete {
            area,
            render: renderable
                .map(|x| x.render(area, driver))
                .unwrap_or_default(),
            rtree: Rc::new(rtree::Node::new(area, None, Default::default(), id)),
            children: Default::default(),
        })
    }
}
