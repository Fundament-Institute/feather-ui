// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::component::ComponentFrom;
use crate::component::text::Text;
use crate::layout::{Desc, Layout, base, flex, leaf};
use crate::persist::{FnPersist, VectorMap};
use crate::{SourceID, UNSIZED_AXIS, gen_id, layout};
use core::f32;
use derive_where::derive_where;
use std::rc::Rc;

#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct Paragraph<T: flex::Prop + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub children: im::Vector<Option<Box<ComponentFrom<dyn flex::Prop>>>>,
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

    fn basis(&self) -> crate::DValue {
        crate::DValue {
            dp: 0.0,
            px: 0.0,
            rel: UNSIZED_AXIS,
        }
    }
}

impl base::Area for MinimalFlexChild {
    fn area(&self) -> &crate::DRect {
        &crate::AUTO_DRECT
    }
}

impl base::Anchor for MinimalFlexChild {}
impl base::Order for MinimalFlexChild {}
impl base::Margin for MinimalFlexChild {}
impl base::RLimits for MinimalFlexChild {}
impl base::Limits for MinimalFlexChild {}
impl base::Padding for MinimalFlexChild {}
impl leaf::Prop for MinimalFlexChild {}
impl leaf::Padded for MinimalFlexChild {}

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
        font: cosmic_text::FamilyOwned,
        color: cosmic_text::Color,
        weight: cosmic_text::Weight,
        style: cosmic_text::Style,
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
                wrap: cosmic_text::Wrap::None,
            };
            self.children.push_back(Some(Box::new(text)));
        }
    }
}

impl<T: flex::Prop + 'static> super::Component<T> for Paragraph<T> {
    fn layout(
        &self,
        state: &crate::StateManager,
        graphics: &crate::graphics::Driver,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let map = VectorMap::new(
            |child: &Option<Box<ComponentFrom<dyn flex::Prop>>>| -> Option<Box<dyn Layout<<dyn flex::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(state, graphics, window, config))
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

crate::gen_component_wrap!(Paragraph, flex::Prop);
