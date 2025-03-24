// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::gen_id;
use crate::layout;
use crate::layout::flex;
use crate::layout::Desc;
use crate::layout::Layout;
use crate::outline::text::Text;
use crate::outline::OutlineFrom;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::Outline;
use crate::SourceID;
use derive_where::derive_where;
use std::rc::Rc;

#[derive_where(Clone)]
pub struct Paragraph<Parent: Clone> {
    pub id: Rc<SourceID>,
    pub props: Parent,
    pub flex: flex::Flex,
    pub children: im::Vector<Option<Box<dyn Outline<<flex::Flex as Desc>::Child>>>>,
}

impl<Parent: Clone + 'static> Paragraph<Parent> {
    pub fn new(id: Rc<SourceID>, props: Parent, flex: flex::Flex) -> Self {
        Self {
            id,
            props,
            flex,
            children: im::Vector::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_text(
        &mut self,
        text: &str,
        font_size: f32,
        line_height: f32,
        font: glyphon::FamilyOwned,
        color: glyphon::Color,
        weight: glyphon::Weight,
        style: glyphon::Style,
        fullwidth: bool,
    ) {
        self.children.clear();
        for word in text.split_ascii_whitespace() {
            let text = Text::<flex::Child> {
                id: gen_id!().into(),
                props: flex::Child {
                    margin: Default::default(),
                    limits: crate::DEFAULT_LIMITS,
                    order: 0,
                    grow: if fullwidth { 1.0 } else { 0.0 },
                    shrink: 0.0,
                    basis: f32::INFINITY,
                },
                text: word.to_owned() + " ",
                font_size,
                line_height,
                font: font.clone(),
                color,
                weight,
                style,
            };
            self.children.push_back(Some(Box::new(text)));
        }
    }
}

impl<Parent: Clone + 'static> super::Outline<Parent> for Paragraph<Parent> {
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
            |child: &Option<Box<OutlineFrom<flex::Flex>>>|
             -> Option<Box<dyn Layout<<flex::Flex as Desc>::Child>>> { Some(child.as_ref().unwrap().layout(state, driver, config)) },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<flex::Flex, Parent> {
            props: self.flex.clone(),
            imposed: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
