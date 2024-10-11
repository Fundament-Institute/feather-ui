use super::mouse_area::MouseArea;
use crate::layout::basic::{self, Basic};
use crate::layout::Layout;
use crate::outline::OutlineFrom;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::EventHandler;
use crate::SourceID;
use crate::{layout, Slot};
use derive_where::derive_where;
use std::rc::Rc;

// A button component that contains a mousearea alongside it's children
#[derive_where(Clone)]
pub struct Button<AppData: 'static, Parent: Clone> {
    pub id: Rc<SourceID>,
    props: Parent,
    basic: Basic,
    marea: MouseArea<AppData, basic::Inherited>,
    children: im::Vector<Option<Box<OutlineFrom<AppData, Basic>>>>,
}

impl<AppData: 'static, Parent: Clone> Button<AppData, Parent> {
    pub fn new(
        id: Rc<SourceID>,
        props: Parent,
        basic: Basic,
        onclick: Slot,
        children: im::Vector<Option<Box<OutlineFrom<AppData, Basic>>>>,
    ) -> Self {
        Self {
            id,
            props,
            basic,
            marea: MouseArea::new(
                basic::Inherited {
                    margin: Default::default(),
                    area: crate::FILL_URECT,
                },
                onclick,
                [],
            ),
            children,
        }
    }
}

impl<AppData: 'static, Parent: Clone + 'static> super::Outline<AppData, Parent>
    for Button<AppData, Parent>
{
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn layout(
        &self,
        state: &mut crate::StateManager,
        driver: &crate::DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Parent, AppData>> {
        let map = VectorMap::new(
            |child: &Option<Box<OutlineFrom<AppData, Basic>>>|
             -> Option<Box<dyn Layout<basic::Inherited, AppData>>> { Some(child.as_ref().unwrap().layout(state, driver, config)) },
        );

        let (_, mut children) = map.call(Default::default(), &self.children);
        children.push_back(Some(self.marea.layout(state, driver, config)));

        Box::new(layout::Node::<AppData, Basic, Parent> {
            props: self.basic.clone(),
            imposed: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
