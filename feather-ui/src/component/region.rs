use crate::component::ComponentFrom;
use crate::layout;
use crate::layout::basic::Basic;
use crate::layout::Desc;
use crate::layout::EventList;
use crate::layout::Layout;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::Component;
use derive_where::derive_where;

#[derive_where(Clone)]
pub struct Region<AppData, Parent: Clone> {
    props: Parent,
    basic: Basic,
    children: im::Vector<Box<dyn Component<AppData, <Basic as Desc<AppData>>::Impose>>>,
}

impl<AppData, Parent: Clone> Region<AppData, Parent> {
    pub fn new(
        props: Parent,
        basic: Basic,
        children: im::Vector<Box<ComponentFrom<AppData, Basic>>>,
    ) -> Self {
        Self {
            props,
            basic,
            children,
        }
    }
}
impl<AppData, Parent: Clone> super::Component<AppData, Parent> for Region<AppData, Parent> {
    fn layout(&self, data: &AppData) -> Box<dyn Layout<Parent, AppData>> {
        let map = VectorMap::new(
            |child: &Box<ComponentFrom<AppData, Basic>>|
             -> Box<dyn Layout<<Basic as Desc<AppData>>::Impose, AppData>> { child.layout(data) },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<AppData, Basic, Parent> {
            props: self.basic.clone(),
            imposed: self.props.clone(),
            children,
            events: std::rc::Weak::<EventList<AppData>>::new(),
            renderable: None,
        })
    }
}
