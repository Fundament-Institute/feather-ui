use std::cell::RefCell;

use crate::{AbsRect, RenderLambda};

pub struct Pipeline {
    pub this: std::rc::Weak<Pipeline>,
    pub renderer: RefCell<glyphon::TextRenderer>,
    pub text_buffer: crate::Rc<RefCell<Option<glyphon::Buffer>>>,
    pub padding: std::cell::Cell<AbsRect>,
}

impl super::Renderable for Pipeline {
    fn render(
        &self,
        area: AbsRect,
        driver: &crate::DriverState,
    ) -> im::Vector<crate::RenderInstruction> {
        let text_system = driver.text().expect("driver.text not initialized");
        let mut borrow = text_system.borrow_mut();
        let padding = self.padding.get();

        let (viewport, atlas, font_system, swash_cache) = borrow.split_borrow();
        self.renderer
            .borrow_mut()
            .prepare(
                &driver.device,
                &driver.queue,
                font_system,
                atlas,
                viewport,
                [glyphon::TextArea {
                    buffer: &self.text_buffer.borrow().as_ref().unwrap(),
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
                swash_cache,
            )
            .unwrap();

        let mut result = im::Vector::new();
        let text_system = text_system.clone();
        let weak = self.this.clone();
        result.push_back(Some(Box::new(move |pass: &mut wgpu::RenderPass| {
            if let Some(this) = weak.upgrade() {
                let text = text_system.borrow();
                let renderer = this.renderer.borrow();
                // SAFETY: This borrow of text and renderer is actually safe, but Glyphon incorrectly requires
                // overly strict lifetimes, so we cast our lifetime away to bypass them.
                let text = unsafe { &*std::ptr::from_ref(&*text) };
                let renderer = unsafe { &*std::ptr::from_ref(&*renderer) };

                renderer.render(&text.atlas, &text.viewport, pass).unwrap();
            }
        }) as Box<dyn RenderLambda>));
        result
    }
}
