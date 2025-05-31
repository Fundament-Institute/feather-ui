// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::cell::RefCell;
use std::rc::Rc;

use glyphon::Viewport;

use crate::{AbsRect, RenderLambda};

pub struct Pipeline {
    pub this: std::rc::Weak<Pipeline>,
    pub renderer: RefCell<glyphon::TextRenderer>,
    pub text_buffer: Rc<RefCell<Option<glyphon::Buffer>>>,
    pub padding: std::cell::Cell<AbsRect>,
    pub atlas: Rc<RefCell<glyphon::TextAtlas>>,
    pub viewport: Rc<RefCell<Viewport>>,
}

impl super::Renderable for Pipeline {
    fn render(
        &self,
        area: AbsRect,
        driver: &crate::DriverState,
    ) -> im::Vector<crate::RenderInstruction> {
        let mut font_system = driver.font_system.write();
        let padding = self.padding.get();

        self.renderer
            .borrow_mut()
            .prepare(
                &driver.device,
                &driver.queue,
                &mut font_system,
                &mut self.atlas.borrow_mut(),
                &self.viewport.borrow(),
                [glyphon::TextArea {
                    buffer: self.text_buffer.borrow().as_ref().unwrap(),
                    left: area.topleft().x + padding.topleft().x,
                    top: area.topleft().y + padding.topleft().y,
                    scale: 1.0,
                    bounds: glyphon::TextBounds {
                        left: area.topleft().x as i32,
                        top: area.topleft().y as i32,
                        right: area.bottomright().x as i32,
                        bottom: area.bottomright().y as i32,
                    },
                    default_color: glyphon::Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                }],
                &mut driver.swash_cache.write(),
            )
            .unwrap();

        let mut result = im::Vector::new();
        let weak = self.this.clone();
        result.push_back(Some(Box::new(move |pass: &mut wgpu::RenderPass| {
            if let Some(this) = weak.upgrade() {
                this.renderer
                    .borrow()
                    .render(&this.atlas.borrow(), &this.viewport.borrow(), pass)
                    .unwrap();
            }
        }) as Box<dyn RenderLambda>));
        result
    }
}
