// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB;
use crate::component::StateMachine;
use crate::graphics::point_to_pixel;
use crate::layout::{self, Layout, leaf};
use crate::{SourceID, WindowStateMachine, graphics};
use cosmic_text::Metrics;
use derive_where::derive_where;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub struct TextState(Rc<RefCell<cosmic_text::Buffer>>);

impl PartialEq for TextState {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

#[derive_where(Clone)]
pub struct Text<T: leaf::Padded + 'static> {
    pub id: Rc<SourceID>,
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

impl<T: leaf::Padded + 'static> crate::StateMachineChild for Text<T> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init(
        &self,
        _: &std::sync::Weak<graphics::Driver>,
    ) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        let statemachine: StateMachine<(), TextState, 0, 0> = StateMachine {
            state: Some(TextState(Rc::new(RefCell::new(
                cosmic_text::Buffer::new_empty(Metrics::new(
                    point_to_pixel(self.font_size, 1.0),
                    point_to_pixel(self.line_height, 1.0),
                )),
            )))),
            input: [],
            output: [],
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

impl<T: leaf::Padded + 'static> super::Component<T> for Text<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &graphics::Driver,
        window: &Rc<SourceID>,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let winstate: &WindowStateMachine = state.get(window).unwrap();
        let winstate = winstate.state.as_ref().expect("No window state available");
        let dpi = winstate.dpi;
        let mut font_system = driver.font_system.write();

        let metrics = cosmic_text::Metrics::new(
            point_to_pixel(self.font_size, dpi.x),
            point_to_pixel(self.line_height, dpi.y),
        );

        let textstate: &StateMachine<(), TextState, 0, 0> = state.get(&self.id).unwrap();
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
            id: Rc::downgrade(&self.id),
            text_render: render.clone(),
            renderable: render.clone(),
        })
    }
}

crate::gen_component_wrap!(Text, leaf::Padded);
