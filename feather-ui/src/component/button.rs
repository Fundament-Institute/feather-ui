use super::mouse_area::MouseArea;
use crate::component::ComponentFrom;
use crate::layout;
use crate::layout::basic::{self, Basic};
use crate::layout::EventList;
use crate::layout::Layout;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::EventHandler;
use derive_where::derive_where;

// A button component that contains a mousearea alongside it's children
#[derive_where(Clone)]
pub struct Button<AppData: 'static, Parent: Clone> {
    props: Parent,
    basic: Basic,
    marea: MouseArea<AppData, basic::Inherited>,
    children: im::Vector<Option<Box<ComponentFrom<AppData, Basic>>>>,
}

impl<AppData: 'static, Parent: Clone> Button<AppData, Parent> {
    pub fn new(
        props: Parent,
        basic: Basic,
        onclick: EventHandler<AppData>,
        children: im::Vector<Option<Box<ComponentFrom<AppData, Basic>>>>,
    ) -> Self {
        Self {
            props,
            basic,
            marea: MouseArea::new(
                basic::Inherited {
                    margin: Default::default(),
                    area: crate::FILL_URECT,
                },
                onclick,
            ),
            children,
        }
    }
}

impl<AppData: 'static, Parent: Clone + 'static> super::Component<AppData, Parent>
    for Button<AppData, Parent>
{
    fn layout(
        &self,
        data: &AppData,
        driver: &crate::DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Parent, AppData>> {
        let map = VectorMap::new(
            |child: &Option<Box<ComponentFrom<AppData, Basic>>>|
             -> Option<Box<dyn Layout<basic::Inherited, AppData>>> { Some(child.as_ref().unwrap().layout(data, driver, config)) },
        );

        let (_, mut children) = map.call(Default::default(), &self.children);
        children.push_back(Some(self.marea.layout(data, driver, config)));

        Box::new(layout::Node::<AppData, Basic, Parent> {
            props: self.basic.clone(),
            imposed: self.props.clone(),
            children,
            events: std::rc::Weak::<EventList<AppData>>::new(),
            renderable: None,
        })
    }
}
