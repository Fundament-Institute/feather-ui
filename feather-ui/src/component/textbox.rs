// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::input::{RawEvent, RawEventKind};
use crate::layout::{Layout, base, leaf};
use crate::{SourceID, layout, point_to_pixel, render};
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use std::ops::Range;
use std::rc::Rc;

use super::StateMachine;

#[derive(Debug, Dispatch, EnumVariantType, Clone, PartialEq, Eq)]
#[evt(derive(Clone), module = "mouse_area_event")]
pub enum TextBoxEvent {
    Edit(Range<usize>, String), // Returns the range of the new string and what the old string used to be.
    Cursor(usize),
    Select(usize, usize),
}

#[derive(Default, Clone, PartialEq)]
struct TextBoxState {
    last_x_offset: f32, // Last cursor x offset when something other than up or down navigation happened
    history: Vec<TextBoxEvent>,
    count: u32,
}

pub trait Prop: leaf::Padded + base::TextEdit {}

#[derive_where(Clone)]
pub struct TextBox<T: Prop + 'static> {
    id: Rc<SourceID>,
    props: Rc<T>,
    pub font_size: f32,
    pub line_height: f32,
    pub font: glyphon::FamilyOwned,
    pub color: glyphon::Color,
    pub weight: glyphon::Weight,
    pub style: glyphon::Style,
    pub wrap: glyphon::Wrap,
    pub slots: [Option<crate::Slot>; 3], // TODO: TextBoxEvent::SIZE
}

impl<T: Prop + 'static> TextBox<T> {
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
            slots: [None, None, None],
        }
    }
}

impl<T: Prop + 'static> super::Component<T> for TextBox<T> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, _: &mut crate::StateManager) -> eyre::Result<()> {
        Ok(())
    }

    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        let props = self.props.clone();
        let oninput = Box::new(crate::wrap_event::<RawEvent, TextBoxEvent, TextBoxState>(
            move |e, area, _dpi, mut data| {
                match e {
                    RawEvent::Key {
                        down, logical_key, ..
                    } => match logical_key {
                        winit::keyboard::Key::Named(named_key) => (),
                        winit::keyboard::Key::Character(c) => {
                            if down {
                                props.textedit().obj.edit(&[(0..0, c.to_string())]);
                                // TODO: This hack may need to be replaced with an explicit method of saying "assume a change happened"
                                data.count += 1;
                                return Ok((data, vec![TextBoxEvent::Edit(0..1, String::new())]));
                            }
                        }
                        _ => (),
                    },
                    RawEvent::Mouse { .. } => return Ok((data, vec![])),
                    _ => (),
                }
                Err((data, vec![]))
            },
        ));

        Ok(Box::new(StateMachine {
            state: Some(Default::default()),
            input: [(
                RawEventKind::Mouse as u64 | RawEventKind::Touch as u64 | RawEventKind::Key as u64,
                oninput,
            )],
            output: self.slots.clone(),
        }))
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

        text_buffer.set_wrap(&mut text_system.borrow_mut().font_system, self.wrap);
        text_buffer.set_text(
            &mut text_system.borrow_mut().font_system,
            &self.props.textedit().obj.text.borrow(),
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

crate::gen_component_wrap!(TextBox, Prop);
