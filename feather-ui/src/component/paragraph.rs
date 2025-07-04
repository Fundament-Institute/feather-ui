// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;
use crate::component::text::Text;
use crate::component::{ChildOf, set_child_parent};
use crate::layout::{Desc, Layout, base, flex, leaf};
use crate::persist::{FnPersist, VectorMap};
use crate::{SourceID, UNSIZED_AXIS, gen_id, layout};
use core::f32;
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct Paragraph<T: flex::Prop + 'static> {
    pub id: Arc<SourceID>,
    props: Rc<T>,
    children: im::Vector<Option<Box<ChildOf<dyn flex::Prop>>>>,
}

#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
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
    pub fn new(id: Arc<SourceID>, props: T) -> Self {
        Self {
            id,
            props: props.into(),
            children: im::Vector::new(),
        }
    }

    pub fn append(&mut self, child: Box<ChildOf<dyn flex::Prop>>) {
        set_child_parent(&child.id(), self.id.clone()).unwrap();
        self.children.push_back(Some(child));
    }

    pub fn prepend(&mut self, child: Box<ChildOf<dyn flex::Prop>>) {
        set_child_parent(&child.id(), self.id.clone()).unwrap();
        self.children.push_front(Some(child));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_text(
        &mut self,
        text: &str,
        font_size: f32,
        line_height: f32,
        font: cosmic_text::FamilyOwned,
        color: sRGB,
        weight: cosmic_text::Weight,
        style: cosmic_text::Style,
        fullwidth: bool,
    ) {
        self.children.clear();
        for (i, word) in text.split_ascii_whitespace().enumerate() {
            let text = Text::<MinimalFlexChild>::new(
                gen_id!(gen_id!(self.id), i),
                MinimalFlexChild {
                    grow: if fullwidth { 1.0 } else { 0.0 },
                }
                .into(),
                font_size,
                line_height,
                word.to_owned() + " ",
                font.clone(),
                color,
                weight,
                style,
                cosmic_text::Wrap::None,
            );
            self.children.push_back(Some(Box::new(text)));
        }
    }
}

impl<T: flex::Prop + 'static> super::Component for Paragraph<T> {
    type Props = T;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        driver: &crate::graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        let mut map = VectorMap::new(
            |child: &Option<Box<ChildOf<dyn flex::Prop>>>| -> Option<Box<dyn Layout<<dyn flex::Prop as Desc>::Child>>> {
                Some(child.as_ref().unwrap().layout(manager, driver, window))
            },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<T, dyn flex::Prop> {
            props: self.props.clone(),
            children,
            id: Arc::downgrade(&self.id),
            renderable: None,
        })
    }
}
