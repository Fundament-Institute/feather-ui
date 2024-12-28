use super::mouse_area::MouseArea;
use crate::layout::basic::{self, Basic};
use crate::layout::Layout;
use crate::outline::OutlineFrom;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::SourceID;
use crate::{layout, Slot};
use derive_where::derive_where;
use std::rc::Rc;

// A button component that contains a mousearea alongside it's children
#[derive_where(Clone)]
pub struct Button<Parent: Clone> {
    pub id: Rc<SourceID>,
    props: Parent,
    basic: Basic,
    marea: MouseArea<basic::Inherited>,
    children: im::Vector<Option<Box<OutlineFrom<Basic>>>>,
}

impl<Parent: Clone> Button<Parent> {
    pub fn new(
        id: Rc<SourceID>,
        props: Parent,
        basic: Basic,
        onclick: Slot,
        children: im::Vector<Option<Box<OutlineFrom<Basic>>>>,
    ) -> Self {
        Self {
            id: id.clone(),
            props,
            basic,
            marea: MouseArea::new(
                SourceID {
                    parent: Rc::downgrade(&id),
                    id: crate::DataID::Named("__marea_internal__"),
                }
                .into(),
                basic::Inherited {
                    margin: Default::default(),
                    area: crate::FILL_URECT,
                    limits: crate::DEFAULT_LIMITS,
                    anchor: Default::default(),
                },
                [Some(onclick), None, None],
            ),
            children,
        }
    }
}

impl<Parent: Clone + 'static> super::Outline<Parent> for Button<Parent> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, manager: &mut crate::StateManager) -> eyre::Result<()> {
        for child in self.children.iter() {
            manager.init_outline(child.as_ref().unwrap().as_ref())?;
        }
        manager.init_outline(&self.marea)?;
        Ok(())
    }

    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &crate::DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Parent>> {
        let map = VectorMap::new(
            |child: &Option<Box<OutlineFrom<Basic>>>| -> Option<Box<dyn Layout<basic::Inherited>>> {
                Some(child.as_ref().unwrap().layout(state, driver, config))
            },
        );

        let (_, mut children) = map.call(Default::default(), &self.children);
        children.push_back(Some(self.marea.layout(state, driver, config)));

        Box::new(layout::Node::<Basic, Parent> {
            props: self.basic.clone(),
            imposed: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
