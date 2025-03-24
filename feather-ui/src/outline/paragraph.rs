// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::gen_id;
use crate::layout;
use crate::layout::base;
use crate::layout::flex;
use crate::layout::leaf;
use crate::layout::Desc;
use crate::layout::Layout;
use crate::layout::LayoutWrap;
use crate::outline::text::Text;
use crate::persist::FnPersist;
use crate::persist::VectorMap;
use crate::SourceID;
use core::f32;
use derive_where::derive_where;
use std::rc::Rc;

use super::OutlineWrap;

#[derive_where(Clone)]
pub struct Paragraph<T: flex::Prop + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub children: im::Vector<Option<Box<dyn OutlineWrap<<dyn flex::Prop as Desc>::Child>>>>,
}

struct MinimalFlexChild {
    grow: f32,
}

impl flex::Child for MinimalFlexChild {
    fn grow(&self) -> f32 {
        self.grow
    }

    fn shrink(&self) -> f32 {
        0.0
    }

    fn basis(&self) -> f32 {
        f32::INFINITY
    }
}

impl base::Order for MinimalFlexChild {
    fn order(&self) -> i64 {
        0
    }
}

impl base::Margin for MinimalFlexChild {
    fn margin(&self) -> &crate::URect {
        &crate::ZERO_URECT
    }
}

impl base::Limits for MinimalFlexChild {
    fn limits(&self) -> &crate::URect {
        &crate::DEFAULT_LIMITS
    }
}

impl leaf::Prop for MinimalFlexChild {}

impl base::Area for MinimalFlexChild {
    fn area(&self) -> &crate::URect {
        &crate::AUTO_URECT
    }
}

impl<T: flex::Prop + 'static> Paragraph<T> {
    pub fn new(id: Rc<SourceID>, props: T) -> Self {
        Self {
            id,
            props: props.into(),
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
            let text = Text::<MinimalFlexChild> {
                id: gen_id!().into(),
                props: MinimalFlexChild {
                    grow: if fullwidth { 1.0 } else { 0.0 },
                }
                .into(),
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

impl<T: flex::Prop + 'static> super::Outline<T> for Paragraph<T> {
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
    ) -> Box<dyn Layout<T>> {
        let map = VectorMap::new(
            |child: &Option<Box<dyn OutlineWrap<<dyn flex::Prop as Desc>::Child>>>| -> Option<Box<dyn LayoutWrap<<dyn flex::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(state, driver, config))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<T, dyn flex::Prop> {
            props: self.props.clone(),
            children,
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}

crate::gen_outline_wrap!(Paragraph, flex::Prop);
