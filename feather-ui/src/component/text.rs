// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;
use crate::component::{EventStream, StateMachine};
use crate::graphics::point_to_pixel;
use crate::layout::{self, Layout, leaf};
use crate::{SourceID, WindowStateMachine, graphics};
use cosmic_text::Metrics;
use derive_where::derive_where;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone)]
pub struct TextState(Rc<RefCell<cosmic_text::Buffer>>);

impl EventStream for TextState {
    type Input = ();
    type Output = ();
}

impl PartialEq for TextState {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

#[derive_where(Clone)]
pub struct Text<T: leaf::Padded + 'static> {
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
}

impl<T: leaf::Padded + 'static> Text<T> {
    pub fn new(
        id: Arc<SourceID>,
        props: Rc<T>,
        font_size: f32,
        line_height: f32,
        text: String,
        font: cosmic_text::FamilyOwned,
        color: sRGB,
        weight: cosmic_text::Weight,
        style: cosmic_text::Style,
        wrap: cosmic_text::Wrap,
    ) -> Self {
        Self {
            id,
            props,
            font_size,
            line_height,
            text,
            font,
            color,
            weight,
            style,
            wrap,
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
            state: Some(TextState(Rc::new(RefCell::new(
                cosmic_text::Buffer::new_empty(Metrics::new(
                    point_to_pixel(self.font_size, 1.0),
                    point_to_pixel(self.line_height, 1.0),
                )),
            )))),
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
        let winstate: &WindowStateMachine = manager.get(window).unwrap();
        let winstate = winstate.state.as_ref().expect("No window state available");
        let dpi = winstate.dpi;
        let mut font_system = driver.font_system.write();

        let metrics = cosmic_text::Metrics::new(
            point_to_pixel(self.font_size, dpi.x),
            point_to_pixel(self.line_height, dpi.y),
        );

        let textstate: &StateMachine<TextState, 0> = manager.get(&self.id).unwrap();
        let textstate = textstate.state.as_ref().expect("No text state available");
        textstate
            .0
            .borrow_mut()
            .set_metrics(&mut font_system, metrics);
        textstate
            .0
            .borrow_mut()
            .set_wrap(&mut font_system, self.wrap);
        textstate.0.borrow_mut().set_text(
            &mut font_system,
            &self.text,
            &cosmic_text::Attrs::new()
                .family(self.font.as_family())
                .color(self.color.into())
                .weight(self.weight)
                .style(self.style),
            cosmic_text::Shaping::Advanced,
        );

        let render = Rc::new(crate::render::text::Instance {
            text_buffer: textstate.0.clone(),
            padding: self.props.padding().resolve(dpi).into(),
        });

        Box::new(layout::text::Node::<T> {
            props: self.props.clone(),
            id: Arc::downgrade(&self.id),
            buffer: textstate.0.clone(),
            renderable: render.clone(),
        })
    }
}
