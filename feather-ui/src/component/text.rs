// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::sRGB;
use crate::component::{EventRouter, StateMachine};
use crate::graphics::point_to_pixel;
use crate::layout::{self, Layout, leaf};
use crate::{SourceID, graphics};
use cosmic_text::{LineIter, Metrics};
use derive_where::derive_where;
use std::cell::RefCell;
use std::convert::Infallible;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone)]
pub struct TextState {
    buffer: Rc<RefCell<cosmic_text::Buffer>>,
    text: String,
    align: Option<cosmic_text::Align>,
}

impl EventRouter for TextState {
    type Input = Infallible;
    type Output = Infallible;
}

impl PartialEq for TextState {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.buffer, &other.buffer)
            && self.text == other.text
            && self.align == other.align
    }
}

#[derive_where(Clone)]
pub struct Text<T> {
    pub id: Arc<SourceID>,
    pub props: Rc<T>,
    pub font_size: f32,
    pub line_height: f32,
    pub text: String,
    pub font: cosmic_text::FamilyOwned,
    pub color: sRGB,
    pub weight: cosmic_text::Weight,
    pub style: cosmic_text::Style,
    pub wrap: cosmic_text::Wrap,
    pub align: Option<cosmic_text::Align>, /* Alignment overrides whether text is LTR or RTL so
                                            * we usually only want to set it if we're centering
                                            * text */
}

impl<T: leaf::Padded + 'static> Text<T> {
    pub fn new(
        id: Arc<SourceID>,
        props: T,
        font_size: f32,
        line_height: f32,
        text: String,
        font: cosmic_text::FamilyOwned,
        color: sRGB,
        weight: cosmic_text::Weight,
        style: cosmic_text::Style,
        wrap: cosmic_text::Wrap,
        align: Option<cosmic_text::Align>,
    ) -> Self {
        Self {
            id,
            props: props.into(),
            font_size,
            line_height,
            text,
            font,
            color,
            weight,
            style,
            wrap,
            align,
        }
    }
}

impl<T: leaf::Padded + 'static> crate::StateMachineChild for Text<T> {
    fn id(&self) -> Arc<SourceID> {
        self.id.clone()
    }

    fn init(
        &self,
        _: &std::sync::Weak<graphics::Driver>,
    ) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        let statemachine: StateMachine<TextState, 0> = StateMachine {
            state: TextState {
                buffer: Rc::new(RefCell::new(cosmic_text::Buffer::new_empty(Metrics::new(
                    point_to_pixel(self.font_size, 1.0),
                    point_to_pixel(self.line_height, 1.0),
                )))),
                text: String::new(),
                align: None,
            },
            input_mask: 0,
            output: [],
            changed: true,
        };
        Ok(Box::new(statemachine))
    }
}

impl<T: Default + leaf::Padded + 'static> Default for Text<T> {
    fn default() -> Self {
        Self {
            id: Default::default(),
            props: Default::default(),
            font_size: Default::default(),
            line_height: Default::default(),
            text: Default::default(),
            font: cosmic_text::FamilyOwned::SansSerif,
            color: sRGB::new(1.0, 1.0, 1.0, 1.0),
            weight: Default::default(),
            style: Default::default(),
            wrap: cosmic_text::Wrap::None,
            align: None,
        }
    }
}

fn buffer_eq(s: &str, b: &cosmic_text::Buffer) -> bool {
    let mut ranges = LineIter::new(s);
    let mut lines = b.lines.iter();
    loop {
        match (lines.next(), ranges.next()) {
            (Some(line), Some((r, _))) => {
                if &s[r] != line.text() {
                    return false;
                }
            }
            (None, None) => return true,
            _ => return false,
        }
    }
}

impl<T: leaf::Padded + 'static> super::Component for Text<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    type Props = T;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        driver: &graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        let dpi = manager
            .get::<super::window::WindowStateMachine>(window)
            .map(|x| x.state.dpi)
            .unwrap_or(crate::BASE_DPI);
        let mut font_system = driver.font_system.write();

        let metrics = cosmic_text::Metrics::new(
            point_to_pixel(self.font_size, dpi.width),
            point_to_pixel(self.line_height, dpi.height),
        );

        let textstate = manager
            .get_mut::<StateMachine<TextState, 0>>(&self.id)
            .unwrap();
        let textstate = &mut textstate.state;
        textstate
            .buffer
            .borrow_mut()
            .set_metrics(&mut font_system, metrics);
        textstate
            .buffer
            .borrow_mut()
            .set_wrap(&mut font_system, self.wrap);

        if self.align != textstate.align || !buffer_eq(&self.text, &textstate.buffer.borrow()) {
            textstate.buffer.borrow_mut().set_text(
                &mut font_system,
                &self.text,
                &cosmic_text::Attrs::new()
                    .family(self.font.as_family())
                    .color(self.color.into())
                    .weight(self.weight)
                    .style(self.style),
                cosmic_text::Shaping::Advanced,
                self.align,
            );

            textstate.text = self.text.clone();
            textstate.align = self.align;
        }

        let render = Rc::new(crate::render::text::Instance {
            text_buffer: textstate.buffer.clone(),
            padding: self.props.padding().as_perimeter(dpi).into(),
        });

        Box::new(layout::text::Node::<T> {
            props: self.props.clone(),
            id: Arc::downgrade(&self.id),
            buffer: textstate.buffer.clone(),
            renderable: render.clone(),
            realign: self.align.is_some_and(|x| x != cosmic_text::Align::Left),
        })
    }
}
