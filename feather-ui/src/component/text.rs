// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{self, Layout, leaf};
use crate::{BASE_DPI, DriverState, SourceID, WindowStateMachine, point_to_pixel};
use derive_where::derive_where;
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

impl<T: leaf::Padded + 'static> super::Component<T> for Text<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    fn id(&self) -> std::rc::Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, _: &mut crate::StateManager) -> eyre::Result<()> {
        Ok(())
    }

    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &DriverState,
        window: &Rc<SourceID>,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let winstate: &WindowStateMachine = state.get(window).unwrap();
        let dpi = winstate.state.as_ref().map(|x| x.dpi).unwrap_or(BASE_DPI);
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
            &self.text,
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

        let render = Rc::new_cyclic(|this| crate::render::text::Pipeline {
            this: this.clone(),
            text_buffer: text_buffer.into(),
            renderer: renderer.into(),
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
