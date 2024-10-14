use super::Renderable;
use crate::layout;
use crate::layout::empty::Empty;
use crate::layout::Layout;
use crate::DriverState;
use crate::RenderLambda;
use crate::SourceID;
use std::cell::RefCell;
use std::rc::Rc;

pub struct TextPipeline {
    pub this: std::rc::Weak<TextPipeline>,
    pub renderer: RefCell<glyphon::TextRenderer>,
    pub text_buffer: RefCell<glyphon::Buffer>,
}

impl Renderable for TextPipeline {
    fn render(
        &self,
        area: crate::AbsRect,
        driver: &crate::DriverState,
    ) -> im::Vector<crate::RenderInstruction> {
        let mut borrow = driver.text.borrow_mut();
        self.text_buffer.borrow_mut().set_size(
            &mut borrow.font_system,
            Some(area.width()),
            Some(area.height()),
        );

        self.text_buffer
            .borrow_mut()
            .shape_until_scroll(&mut borrow.font_system, false);

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
                    buffer: &self.text_buffer.borrow(),
                    left: area.topleft.x,
                    top: area.topleft.y,
                    scale: 1.0,
                    bounds: glyphon::TextBounds {
                        left: area.topleft.x as i32,
                        top: area.topleft.y as i32,
                        right: area.bottomright.x as i32,
                        bottom: area.bottomright.y as i32,
                    },
                    default_color: glyphon::Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                }],
                swash_cache,
            )
            .unwrap();

        let mut result = im::Vector::new();
        let text_system = driver.text.clone();
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

#[derive(Clone)]
pub struct Text<Parent: Clone> {
    pub id: Rc<SourceID>,
    pub props: Parent,
    pub font_size: f32,
    pub line_height: f32,
    pub text: String,
    pub font: glyphon::FamilyOwned,
    pub color: glyphon::Color,
    pub weight: glyphon::Weight,
    pub style: glyphon::Style,
}

impl<Parent: Clone + Default> Default for Text<Parent> {
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
        }
    }
}

impl<Parent: Clone + 'static> super::Outline<Parent> for Text<Parent> {
    fn id(&self) -> std::rc::Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, _: &mut crate::StateManager) -> eyre::Result<()> {
        Ok(())
    }

    fn layout(
        &self,
        _: &crate::StateManager,
        driver: &DriverState,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Parent>> {
        let mut text_buffer = glyphon::Buffer::new(
            &mut driver.text.borrow_mut().font_system,
            glyphon::Metrics::new(self.font_size, self.line_height),
        );

        text_buffer.set_text(
            &mut driver.text.borrow_mut().font_system,
            &self.text,
            glyphon::Attrs::new()
                .family(self.font.as_family())
                .color(self.color)
                .weight(self.weight)
                .style(self.style),
            glyphon::Shaping::Advanced,
        );

        let renderer = glyphon::TextRenderer::new(
            &mut driver.text.borrow_mut().atlas,
            &driver.device,
            wgpu::MultisampleState::default(),
            None,
        );

        Box::new(layout::Node::<Empty, Parent> {
            props: (),
            imposed: self.props.clone(),
            children: Default::default(),
            id: Rc::downgrade(&self.id),
            renderable: Some(Rc::new_cyclic(|this| TextPipeline {
                this: this.clone(),
                text_buffer: text_buffer.into(),
                renderer: renderer.into(),
            })),
        })
    }
}
