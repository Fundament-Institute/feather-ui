// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{self, Layout, leaf};
use crate::{DriverState, SourceID, WindowStateMachine, point_to_pixel};
use derive_where::derive_where;
use std::cell::RefCell;
use std::rc::Rc;

#[derive_where(Clone)]
pub struct Text<T: leaf::Padded + 'static> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub font_size: f32,
    pub line_height: f32,
    pub text: String,
    pub font: glyphon::FamilyOwned,
    pub color: glyphon::Color,
    pub weight: glyphon::Weight,
    pub style: glyphon::Style,
    pub wrap: glyphon::Wrap,
}

impl<T: Default + leaf::Padded + 'static> Default for Text<T> {
    fn default() -> Self {
        Self {
            id: Default::default(),
            props: Default::default(),
            font_size: Default::default(),
            line_height: Default::default(),
            text: Default::default(),
            font: glyphon::FamilyOwned::SansSerif,
            color: glyphon::Color::rgba(255, 255, 255, 255),
            weight: Default::default(),
            style: Default::default(),
            wrap: glyphon::Wrap::None,
        }
    }
}

impl<T: leaf::Padded + 'static> crate::StateMachineChild for Text<T> {
    fn id(&self) -> std::rc::Rc<SourceID> {
        self.id.clone()
    }
}

impl<T: leaf::Padded + 'static> super::Component<T> for Text<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &DriverState,
        window: &Rc<SourceID>,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let winstate: &WindowStateMachine = state.get(window).unwrap();
        let winstate = winstate.state.as_ref().expect("No window state available");
        let dpi = winstate.dpi;
        let mut font_system = driver.font_system.write();
        let mut text_buffer = glyphon::Buffer::new(
            &mut font_system,
            glyphon::Metrics::new(
                point_to_pixel(self.font_size, dpi.x),
                point_to_pixel(self.line_height, dpi.x),
            ),
        );

        text_buffer.set_wrap(&mut font_system, self.wrap);
        text_buffer.set_text(
            &mut font_system,
            &self.text,
            &glyphon::Attrs::new()
                .family(self.font.as_family())
                .color(self.color)
                .weight(self.weight)
                .style(self.style),
            glyphon::Shaping::Advanced,
        );

        let renderer = glyphon::TextRenderer::new(
            &mut winstate.atlas.borrow_mut(),
            &driver.device,
            wgpu::MultisampleState::default(),
            None,
        );

        let render = Rc::new_cyclic(|this| crate::render::text::Pipeline {
            this: this.clone(),
            text_buffer: Rc::new(RefCell::new(Some(text_buffer))),
            renderer: renderer.into(),
            padding: self.props.padding().resolve(dpi).into(),
            atlas: winstate.atlas.clone(),
            viewport: winstate.viewport.clone(),
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
