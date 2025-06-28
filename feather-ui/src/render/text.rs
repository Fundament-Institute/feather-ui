// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::cell::RefCell;
use std::rc::Rc;

use cosmic_text::{CacheKey, FontSystem};
use guillotiere::{AllocId, Size};
use ultraviolet::Vec2;
use wgpu::{Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo};

use crate::color::{Premultiplied, sRGB32};
use crate::graphics::{GlyphCache, GlyphRegion};
use crate::render::atlas::Atlas;
use crate::render::compositor::Compositor;
use crate::{AbsRect, Error};

use swash::scale::{Render, ScaleContext, Source, StrikeWith};
use swash::zeno::{Format, Vector};

pub use swash::scale::image::{Content, Image};
pub use swash::zeno::{Angle, Command, Placement, Transform};

pub struct Instance {
    pub text_buffer: Rc<RefCell<cosmic_text::Buffer>>,
    pub padding: std::cell::Cell<AbsRect>,
}

impl Instance {
    pub fn get_glyph(key: CacheKey, glyphs: &GlyphCache) -> Option<&GlyphRegion> {
        glyphs.get(&key)
    }

    pub fn draw_glyph(
        font_system: &mut FontSystem,
        context: &mut ScaleContext,
        cache_key: CacheKey,
    ) -> Option<Image> {
        let font = match font_system.get_font(cache_key.font_id) {
            Some(some) => some,
            None => {
                debug_assert!(false, "did not find font {:?}", cache_key.font_id);
                return None;
            }
        };

        // Build the scaler
        let mut scaler = context
            .builder(font.as_swash())
            .size(f32::from_bits(cache_key.font_size_bits))
            .hint(true)
            .build();

        // Compute the fractional offset-- you'll likely want to quantize this
        // in a real renderer
        let offset = Vector::new(cache_key.x_bin.as_float(), cache_key.y_bin.as_float());

        // Select our source order
        Render::new(&[
            // Color outline with the first palette
            Source::ColorOutline(0),
            // Color bitmap with best fit selection mode
            Source::ColorBitmap(StrikeWith::BestFit),
            // Standard scalable outline
            Source::Outline,
        ])
        // Select a subpixel format
        .format(Format::Alpha)
        // Apply the fractional offset
        .offset(offset)
        // Render the image
        .render(&mut scaler, cache_key.glyph_id)
    }

    pub fn write_glyph(
        key: CacheKey,
        font_system: &mut FontSystem,
        glyphs: &mut GlyphCache,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: &mut Atlas,
        cache: &mut ScaleContext,
    ) -> Result<(), Error> {
        if glyphs.get(&key).is_some() {
            // We can't actually return this borrow because of https://github.com/rust-lang/rust/issues/58910
            return Ok(());
        }

        let Some(mut image) = Self::draw_glyph(font_system, cache, key) else {
            return Err(Error::GlyphRenderFailure);
        };

        let region = if image.data.is_empty() {
            super::atlas::Region {
                id: AllocId::deserialize(u32::MAX),
                uv: guillotiere::euclid::Box2D::zero(),
                index: 0,
            }
        } else {
            // Find a position in the packer
            atlas.reserve(
                device,
                Size::new(image.placement.width as i32, image.placement.height as i32),
            )?
        };

        if !image.data.is_empty() {
            match image.content {
                Content::Mask => {
                    let mask = image.data;
                    image.data = mask
                        .iter()
                        .flat_map(|x| sRGB32::new(255, 255, 255, *x).as_f32().srgb_pre().as_bgra())
                        .collect();
                }
                // This is in sRGB RGBA format but our texture atlas is in pre-multiplied sRGB BGRA format, so swap it
                Content::Color => {
                    for c in image.data.as_mut_slice().chunks_exact_mut(4) {
                        // Pre-multiply color, then extract in BGRA form.
                        c.copy_from_slice(
                            &sRGB32::new(c[0], c[1], c[2], c[3])
                                .as_f32()
                                .srgb_pre()
                                .as_bgra(),
                        );
                    }
                }
                Content::SubpixelMask => {
                    // TODO: wide doesn't implement SSE shuffle instructions yet, which could potentially be faster here
                    let len = image.data.len() / 4;
                    let slice = image.data.as_mut_slice();
                    for i in 0..len {
                        let idx = i * 4;
                        slice.swap(idx, idx + 2);
                        // Don't pre-multiply this because it's already a mask
                    }
                }
            }

            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: atlas.get_texture(),
                    mip_level: 0,
                    origin: Origin3d {
                        x: region.uv.min.x as u32,
                        y: region.uv.min.y as u32,
                        z: region.index as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &image.data,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        image.placement.width * atlas.get_texture().format().components() as u32,
                    ),
                    rows_per_image: None,
                },
                Extent3d {
                    width: image.placement.width,
                    height: image.placement.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        if let Some(mut old) = glyphs.insert(
            key,
            GlyphRegion {
                offset: [image.placement.left, image.placement.top],
                region,
            },
        ) {
            atlas.destroy(&mut old.region);
        }

        Ok(())
    }

    pub fn prepare_glyph(
        x: i32,
        y: i32,
        line_y: f32,
        scale_factor: f32,
        color: cosmic_text::Color,
        bounds_min_x: i32,
        bounds_min_y: i32,
        bounds_max_x: i32,
        bounds_max_y: i32,
        glyph: &GlyphRegion,
    ) -> Result<Option<super::compositor::Data>, Error> {
        if glyph.region.uv.area() == 0 {
            return Ok(None);
        }
        //let atlas_min = region.uv.min;

        let mut x = x + glyph.offset[0];
        let mut y = (line_y * scale_factor).round() as i32 + y - glyph.offset[1];

        let mut u = glyph.region.uv.min.x;
        let mut v = glyph.region.uv.min.y;

        let mut width = glyph.region.uv.width();
        let mut height = glyph.region.uv.height();

        // Starts beyond right edge or ends beyond left edge
        let max_x = x + width;
        if x > bounds_max_x || max_x < bounds_min_x {
            return Ok(None);
        }

        // Starts beyond bottom edge or ends beyond top edge
        let max_y = y + height;
        if y > bounds_max_y || max_y < bounds_min_y {
            return Ok(None);
        }

        // Clip left ege
        if x < bounds_min_x {
            let right_shift = bounds_min_x - x;

            x = bounds_min_x;
            width = max_x - bounds_min_x;
            u += right_shift;
        }

        // Clip right edge
        if x + width > bounds_max_x {
            width = bounds_max_x - x;
        }

        // Clip top edge
        if y < bounds_min_y {
            let bottom_shift = bounds_min_y - y;

            y = bounds_min_y;
            height = max_y - bounds_min_y;
            v += bottom_shift;
        }

        // Clip bottom edge
        if y + height > bounds_max_y {
            height = bounds_max_y - y;
        }

        Ok(Some(super::compositor::Data {
            pos: [x as f32, y as f32].into(),
            dim: [width as f32, height as f32].into(),
            uv: [u as f32, v as f32].into(),
            uvdim: [width as f32, height as f32].into(),
            color: color.0,
            rotation: 0.0,
            texclip: ((glyph.region.index as u32) << 16),
            ..Default::default()
        }))
    }

    fn evaluate(
        buffer: &cosmic_text::Buffer,
        pos: Vec2,
        scale: f32,
        bounds: AbsRect,
        color: cosmic_text::Color,
        compositor: &mut super::compositor::Compositor,
        font_system: &mut FontSystem,
        glyphs: &mut GlyphCache,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: &mut Atlas,
        cache: &mut ScaleContext,
    ) -> Result<(), Error> {
        let bounds_top = bounds.topleft().y as i32;
        let bounds_bottom = bounds.bottomright().y as i32;
        let bounds_min_x = (bounds.topleft().x as i32).max(0);
        let bounds_min_y = bounds_top.max(0);
        let bounds_max_x = bounds.bottomright().x as i32;
        let bounds_max_y = bounds_bottom;

        let is_run_visible = |run: &cosmic_text::LayoutRun| {
            let start_y_physical = (pos.y + (run.line_top * scale)) as i32;
            let end_y_physical = start_y_physical + (run.line_height * scale) as i32;

            start_y_physical <= bounds_max_y && bounds_top <= end_y_physical
        };

        let layout_runs = buffer
            .layout_runs()
            .skip_while(|run| !is_run_visible(run))
            .take_while(is_run_visible);

        for run in layout_runs {
            for glyph in run.glyphs.iter() {
                let physical_glyph = glyph.physical((pos.x, pos.y), scale);

                let color = match glyph.color_opt {
                    Some(some) => some,
                    None => color,
                };

                Self::write_glyph(
                    physical_glyph.cache_key,
                    font_system,
                    glyphs,
                    device,
                    queue,
                    atlas,
                    cache,
                )?;

                if let Some(data) = Self::prepare_glyph(
                    physical_glyph.x,
                    physical_glyph.y,
                    run.line_y,
                    scale,
                    color,
                    bounds_min_x,
                    bounds_min_y,
                    bounds_max_x,
                    bounds_max_y,
                    Self::get_glyph(physical_glyph.cache_key, glyphs)
                        .ok_or(Error::GlyphCacheFailure)?,
                )? {
                    compositor.append(&data);
                }
            }
        }

        Ok(())
    }
}

impl super::Renderable for Instance {
    fn render(
        &self,
        area: AbsRect,
        driver: &crate::graphics::Driver,
        compositor: &mut Compositor,
    ) -> Result<(), Error> {
        let padding = self.padding.get();

        Self::evaluate(
            &self.text_buffer.borrow(),
            area.topleft() + padding.topleft(),
            1.0,
            area,
            cosmic_text::Color::rgb(255, 255, 255),
            compositor,
            &mut driver.font_system.write(),
            &mut driver.glyphs.write(),
            &driver.device,
            &driver.queue,
            &mut driver.atlas.write(),
            &mut driver.swash_cache.write(),
        )
    }
}
