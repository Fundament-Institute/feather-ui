use crate::layout;
use crate::layout::basic::Basic;
use crate::layout::Desc;
use crate::layout::Layout;
use crate::outline::OutlineFrom;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::Outline;
use crate::SourceID;
use derive_where::derive_where;
use std::rc::Rc;

#[derive_where(Clone)]
pub struct Region<AppData: 'static, Parent: Clone> {
    pub id: Rc<SourceID>,
    pub props: Parent,
    pub basic: Basic,
    pub children: im::Vector<Option<Box<dyn Outline<AppData, <Basic as Desc<AppData>>::Impose>>>>,
}

impl<AppData: 'static, Parent: Clone + 'static> super::Outline<AppData, Parent>
    for Region<AppData, Parent>
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
             -> Option<Box<dyn Layout<<Basic as Desc<AppData>>::Impose, AppData>>> { Some(child.as_ref().unwrap().layout(data, driver, config)) },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<AppData, Basic, Parent> {
            props: self.basic.clone(),
            imposed: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
