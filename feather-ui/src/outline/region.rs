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
pub struct Region<Parent: Clone> {
    pub id: Rc<SourceID>,
    pub props: Parent,
    pub basic: Basic,
    pub children: im::Vector<Option<Box<dyn Outline<<Basic as Desc>::Impose>>>>,
}

impl<Parent: Clone + 'static> super::Outline<Parent> for Region<Parent> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, manager: &mut crate::StateManager) -> eyre::Result<()> {
        for child in self.children.iter() {
            manager.init_outline(child.as_ref().unwrap().as_ref())?;
        }
        Ok(())
    }

    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &crate::DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Parent>> {
        let map = VectorMap::new(
            |child: &Option<Box<OutlineFrom<Basic>>>|
             -> Option<Box<dyn Layout<<Basic as Desc>::Impose>>> { Some(child.as_ref().unwrap().layout(state, driver, config)) },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<Basic, Parent> {
            props: self.basic.clone(),
            imposed: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
