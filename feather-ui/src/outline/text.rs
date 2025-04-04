// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::Renderable;
use crate::layout::leaf;
use crate::layout::Layout;
use crate::rtree;
use crate::AbsRect;
use crate::DriverState;
use crate::RenderLambda;
use crate::SourceID;
use crate::Vec2;
use derive_where::derive_where;
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
        let text_system = driver.text().expect("driver.text not initialized");
        let mut borrow = text_system.borrow_mut();

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

#[derive_where(Clone)]
pub struct TextLayout<T> {
    pub id: std::rc::Weak<SourceID>,
    pub props: Rc<T>,
    pub text_render: Rc<TextPipeline>,
}

#[derive_where(Clone)]
pub struct Text<T> {
    pub id: Rc<SourceID>,
    pub props: Rc<T>,
    pub font_size: f32,
    pub line_height: f32,
    pub text: String,
    pub font: glyphon::FamilyOwned,
    pub color: glyphon::Color,
    pub weight: glyphon::Weight,
    pub style: glyphon::Style,
}

impl<T: Default> Default for Text<T> {
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

impl<T: leaf::Prop + 'static> super::Outline<T> for Text<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Prop + 'static)>,
{
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
    ) -> Box<dyn Layout<T>> {
        let text_system = driver.text().expect("driver.text not initialized");
        let mut text_buffer = glyphon::Buffer::new(
            &mut text_system.borrow_mut().font_system,
            glyphon::Metrics::new(self.font_size, self.line_height),
        );

        text_buffer.set_text(
            &mut text_system.borrow_mut().font_system,
            &self.text,
            glyphon::Attrs::new()
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

        Box::new(TextLayout::<T> {
            props: self.props.clone(),
            id: Rc::downgrade(&self.id),
            text_render: Rc::new_cyclic(|this| TextPipeline {
                this: this.clone(),
                text_buffer: text_buffer.into(),
                renderer: renderer.into(),
            }),
        })
    }
}

crate::gen_outline_wrap!(Text, leaf::Prop);

impl<T> Layout<T> for TextLayout<T> {
    fn get_props(&self) -> &T {
        &self.props
    }
    fn inner_stage<'a>(
        &self,
        mut area: AbsRect,
        driver: &DriverState,
    ) -> Box<dyn crate::outline::Staged + 'a> {
        let text_system = driver.text().expect("driver.text not initialized");
        self.text_render.text_buffer.borrow_mut().set_size(
            &mut text_system.borrow_mut().font_system,
            if area.bottomright.x.is_infinite() {
                None
            } else {
                Some(area.width())
            },
            if area.bottomright.y.is_infinite() {
                None
            } else {
                Some(area.height())
            },
        );

        self.text_render
            .text_buffer
            .borrow_mut()
            .shape_until_scroll(&mut text_system.borrow_mut().font_system, false);

        // If we have indeterminate area, calculate the size
        if area.bottomright.x.is_infinite() || area.bottomright.y.is_infinite() {
            let mut h = 0.0;
            let mut w = 0.0;
            for run in self.text_render.text_buffer.borrow().layout_runs() {
                w += run.line_w;
                h += run.line_height;
            }
            if area.bottomright.x.is_infinite() {
                area.bottomright.x = area.topleft.x + w;
            }
            if area.bottomright.y.is_infinite() {
                area.bottomright.y = area.topleft.y + h;
            }
        }

        Box::new(crate::layout::Concrete {
            area: area,
            render: Some(self.text_render.clone()),
            rtree: Rc::new(rtree::Node::new(
                area,
                None,
                Default::default(),
                self.id.clone(),
            )),
            children: Default::default(),
        })
    }
}
