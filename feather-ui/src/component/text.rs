use crate::layout;
use crate::layout::empty::Empty;
use crate::layout::EventList;
use crate::layout::Layout;
use crate::DriverState;
use crate::RenderLambda;
use std::cell::RefCell;
use std::rc::Rc;

use super::Renderable;

pub struct TextPipeline {
    pub this: std::rc::Weak<TextPipeline>,
    pub text_buffer: RefCell<glyphon::Buffer>,
}

impl<AppData> Renderable<AppData> for TextPipeline {
    fn render(
        &self,
        area: crate::AbsRect,
        driver: &crate::DriverState,
    ) -> im::Vector<crate::RenderInstruction> {
        let mut borrow = driver.text.borrow_mut();
        let (viewport, atlas, text_renderer, font_system, swash_cache) = borrow.split_borrow();
        self.text_buffer.borrow_mut().set_size(
            font_system,
            Some(area.width()),
            Some(area.height()),
        );

        self.text_buffer
            .borrow_mut()
            .shape_until_scroll(font_system, false);

        text_renderer
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
                        left: 0,
                        top: 0,
                        right: area.width() as i32,
                        bottom: area.height() as i32,
                    },
                    default_color: glyphon::Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                }],
                swash_cache,
            )
            .unwrap();

        let mut result = im::Vector::new();
        let text_system = driver.text.clone();
        result.push_back(Some(Box::new(move |pass: &mut wgpu::RenderPass| {
            let text = text_system.borrow();

            text.text_renderer
                .render(&text.atlas, &text.viewport, pass)
                .unwrap();
        }) as Box<dyn RenderLambda>));
        result
    }
}

#[derive(Clone)]
pub struct Text<Parent: Clone> {
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

impl<AppData: 'static, Parent: Clone + 'static> super::Component<AppData, Parent> for Text<Parent> {
    fn layout(
        &self,
        _: &AppData,
        driver: &DriverState,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Parent, AppData>> {
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
        Box::new(layout::Node::<AppData, Empty, Parent> {
            props: (),
            imposed: self.props.clone(),
            children: Default::default(),
            events: std::rc::Weak::<EventList<AppData>>::new(),
            renderable: Some(Rc::new_cyclic(|this| TextPipeline {
                this: this.clone(),
                text_buffer: text_buffer.into(),
            })),
        })
    }
}
