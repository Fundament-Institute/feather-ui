// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::Renderable;
use crate::layout::{cap_unsized, check_unsized, leaf, limit_area, Layout};
use crate::{point_to_pixel, rtree, AbsRect, DriverState, RenderLambda, SourceID};
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

impl<T: leaf::Padded + 'static> super::Outline<T> for Text<T>
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
        _: &crate::StateManager,
        driver: &DriverState,
        dpi: crate::Vec2,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
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

        Box::new(TextLayout::<T> {
            props: self.props.clone(),
            id: Rc::downgrade(&self.id),
            text_render: Rc::new_cyclic(|this| TextPipeline {
                this: this.clone(),
                text_buffer: text_buffer.into(),
                renderer: renderer.into(),
                padding: self.props.padding().resolve(dpi).into(),
            }),
        })
    }
}

crate::gen_outline_wrap!(Text, leaf::Padded);

#[derive_where(Clone)]
pub struct TextLayout<T: leaf::Padded> {
    pub id: std::rc::Weak<SourceID>,
    pub props: Rc<T>,
    pub text_render: Rc<TextPipeline>,
}

impl<T: leaf::Padded> Layout<T> for TextLayout<T> {
    fn get_props(&self) -> &T {
        &self.props
    }
    fn inner_stage<'a>(
        &self,
        outer_area: AbsRect,
        outer_limits: crate::AbsLimits,
        dpi: crate::Vec2,
        driver: &DriverState,
    ) -> Box<dyn crate::outline::Staged + 'a> {
        let text_system: &Rc<RefCell<crate::TextSystem>> =
            driver.text().expect("driver.text not initialized");
        let mut limits = self.props.limits().resolve(dpi) + outer_limits;
        let myarea = self.props.area().resolve(dpi);
        let (unsized_x, unsized_y) = check_unsized(myarea);
        let padding = self.props.padding().resolve(dpi);
        let allpadding = myarea.bottomright().abs() + padding.topleft() + padding.bottomright();
        let minmax = limits.0.as_array_mut();
        if unsized_x {
            minmax[2] -= allpadding.x;
            minmax[0] -= allpadding.x;
        }
        if unsized_y {
            minmax[3] -= allpadding.y;
            minmax[1] -= allpadding.y;
        }

        let mut evaluated_area = limit_area(
            cap_unsized(myarea * crate::layout::nuetralize_unsized(outer_area)),
            limits,
        );

        let (limitx, limity) = {
            let max = limits.max();
            (
                max.x.is_finite().then_some(max.x),
                max.y.is_finite().then_some(max.y),
            )
        };

        let dim = evaluated_area.dim();
        self.text_render.text_buffer.borrow_mut().set_size(
            &mut text_system.borrow_mut().font_system,
            if unsized_x { limitx } else { Some(dim.0.x) },
            if unsized_y { limity } else { Some(dim.0.y) },
        );

        self.text_render
            .text_buffer
            .borrow_mut()
            .shape_until_scroll(&mut text_system.borrow_mut().font_system, false);

        // If we have indeterminate area, calculate the size
        if unsized_x || unsized_y {
            let mut h = 0.0;
            let mut w: f32 = 0.0;
            for run in self.text_render.text_buffer.borrow().layout_runs() {
                w = w.max(run.line_w);
                h += run.line_height;
            }

            // Apply adjusted limits to inner size calculation
            w = w.max(limits.min().x).min(limits.max().x);
            h = h.max(limits.min().y).min(limits.max().y);
            let ltrb = evaluated_area.0.as_array_mut();
            if unsized_x {
                ltrb[2] = ltrb[0] + w + allpadding.x;
            }
            if unsized_y {
                ltrb[3] = ltrb[1] + h + allpadding.y;
            }
        };

        // We always return the full area so we can correctly capture input, but we need
        // to pass our padding to our renderer so it can pad the text while setting the
        // clip rect to the full area
        self.text_render.padding.set(padding);

        evaluated_area = crate::layout::apply_anchor(
            evaluated_area,
            outer_area,
            self.props.anchor().resolve(dpi) * evaluated_area.dim(),
        );

        Box::new(crate::layout::Concrete::new(
            Some(self.text_render.clone()),
            evaluated_area,
            Rc::new(rtree::Node::new(
                evaluated_area,
                None,
                Default::default(),
                self.id.clone(),
            )),
            Default::default(),
        ))
    }
}

pub struct TextPipeline {
    pub this: std::rc::Weak<TextPipeline>,
    pub renderer: RefCell<glyphon::TextRenderer>,
    pub text_buffer: RefCell<glyphon::Buffer>,
    pub padding: std::cell::Cell<AbsRect>,
}

impl Renderable for TextPipeline {
    fn render(
        &self,
        area: crate::AbsRect,
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
                    buffer: &self.text_buffer.borrow(),
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
