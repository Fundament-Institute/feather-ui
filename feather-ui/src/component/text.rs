// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::graphics::point_to_pixel;
use crate::layout::{self, Layout, leaf};
use crate::{SourceID, WindowStateMachine, graphics};
use derive_where::derive_where;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct Text<T: leaf::Padded + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub font_size: f32,
    pub line_height: f32,
    pub text: String,
    pub font: cosmic_text::FamilyOwned,
    pub color: cosmic_text::Color,
    pub weight: cosmic_text::Weight,
    pub style: cosmic_text::Style,
    pub wrap: cosmic_text::Wrap,
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
            color: cosmic_text::Color::rgba(255, 255, 255, 255),
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
        graphics: &graphics::Driver,
        window: &Rc<SourceID>,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let winstate: &WindowStateMachine = state.get(window).unwrap();
        let winstate = winstate.state.as_ref().expect("No window state available");
        let dpi = winstate.dpi;
        let mut font_system = graphics.font_system.write();
        let mut text_buffer = cosmic_text::Buffer::new(
            &mut font_system,
            cosmic_text::Metrics::new(
                point_to_pixel(self.font_size, dpi.x),
                point_to_pixel(self.line_height, dpi.x),
            ),
        );

        text_buffer.set_wrap(&mut font_system, self.wrap);
        text_buffer.set_text(
            &mut font_system,
            &self.text,
            &cosmic_text::Attrs::new()
                .family(self.font.as_family())
                .color(self.color)
                .weight(self.weight)
                .style(self.style),
            cosmic_text::Shaping::Advanced,
        );

        let render = Rc::new(crate::render::text::Instance {
            text_buffer: Rc::new(RefCell::new(Some(text_buffer))),
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
