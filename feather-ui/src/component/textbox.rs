// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::base::TextEditSnapshot;
use crate::layout::{base, leaf, Desc, Layout, LayoutWrap};
use crate::persist::{FnPersist, VectorMap};
use crate::{layout, point_to_pixel, render, DRect, SourceID};
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use std::ops::Range;
use std::rc::Rc;
use ultraviolet::{Vec2, Vec4};

use super::line::Line;
use super::text::Text;
use super::{Component, ComponentFrom};

#[derive(Debug, Dispatch, EnumVariantType, Clone)]
#[evt(derive(Clone), module = "mouse_area_event")]
pub enum TextBoxEvent {
    Edit(Range<usize>, String), // Returns the range of the new string and what the old string used to be.
    Cursor(usize),
    Select(usize, usize),
}

pub trait Prop: leaf::Padded + base::TextEdit {}

#[derive_where(Clone)]
pub struct Textbox<T: Prop + 'static> {
    id: Rc<SourceID>,
    props: Rc<T>,
    pub font_size: f32,
    pub line_height: f32,
    pub font: glyphon::FamilyOwned,
    pub color: glyphon::Color,
    pub weight: glyphon::Weight,
    pub style: glyphon::Style,
    pub wrap: glyphon::Wrap,
}

impl<T: Prop + 'static> Textbox<T> {
    pub fn new(
        id: Rc<SourceID>,
        props: T,
        font_size: f32,
        line_height: f32,
        font: glyphon::FamilyOwned,
        color: glyphon::Color,
        weight: glyphon::Weight,
        style: glyphon::Style,
        wrap: glyphon::Wrap,
    ) -> Self {
        Self {
            id: id.clone(),
            props: props.into(),
            font_size,
            line_height,
            font,
            color,
            weight,
            style,
            wrap,
        }
    }
}

impl<T: Prop + 'static> super::Component<T> for Textbox<T> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, manager: &mut crate::StateManager) -> eyre::Result<()> {
        Ok(())
    }

    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &crate::DriverState,
        dpi: crate::Vec2,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let text_system = driver.text().expect("driver.text not initialized");
        let mut text_buffer = glyphon::Buffer::new(
            &mut text_system.borrow_mut().font_system,
            glyphon::Metrics::new(
                point_to_pixel(self.font_size, dpi.x),
                point_to_pixel(self.line_height, dpi.x),
            ),
        );

        let snapshot: TextEditSnapshot<T> = self.props.clone().into();
        text_buffer.set_wrap(&mut text_system.borrow_mut().font_system, self.wrap);
        text_buffer.set_text(
            &mut text_system.borrow_mut().font_system,
            &snapshot.obj.textedit().text.borrow(),
            &glyphon::Attrs::new()
                .family(self.font.as_family())
                .color(self.color)
                .weight(self.weight)
                .style(self.style),
            glyphon::Shaping::Advanced,
        );

        let renderer = glyphon::TextRenderer::new(
            &mut text_system.borrow_mut().atlas,
            &driver.device,
            wgpu::MultisampleState::default(),
            None,
        );

        let textrender = Rc::new_cyclic(|this| render::text::Pipeline {
            this: this.clone(),
            text_buffer: text_buffer.into(),
            renderer: renderer.into(),
            padding: self.props.padding().resolve(dpi).into(),
        });

        Box::new(layout::text::Node::<T> {
            props: self.props.clone(),
            id: Rc::downgrade(&self.id),
            text_render: textrender.clone(),
            renderable: textrender.clone(),
        })
    }
}

crate::gen_component_wrap!(Textbox, Prop);
