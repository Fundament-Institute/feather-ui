// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::cell::RefCell;
use std::rc::Rc;

use cosmic_text::{CacheKey, FontSystem, SwashCache, SwashContent};
use guillotiere::AllocId;
use guillotiere::Size;
use ultraviolet::Vec2;
use wgpu::Extent3d;
use wgpu::Origin3d;
use wgpu::TexelCopyBufferLayout;
use wgpu::TexelCopyTextureInfo;

use crate::Error;
use crate::color::Premultiplied;
use crate::color::sRGB32;
use crate::graphics::GlyphCache;
use crate::graphics::GlyphRegion;
use crate::render::compositor::Compositor;
use crate::{AbsRect, render::atlas::Atlas};

pub struct Instance {
    pub text_buffer: Rc<RefCell<Option<cosmic_text::Buffer>>>,
    pub padding: std::cell::Cell<AbsRect>,
}

impl Instance {
    fn get_glyph(key: CacheKey, glyphs: &GlyphCache) -> Option<&GlyphRegion> {
        glyphs.get(&key)
    }

    pub fn write_glyph<'a>(
        key: CacheKey,
        font_system: &mut FontSystem,
        glyphs: &'a mut GlyphCache,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: &mut Atlas,
        cache: &mut SwashCache,
    ) -> Result<(), Error> {
        if let Some(_) = glyphs.get(&key) {
            // We can't actually return this borrow because of https://github.com/rust-lang/rust/issues/58910
            return Ok(());
        }

        let Some(mut image) = cache.get_image_uncached(font_system, key) else {
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
                SwashContent::Mask => {
                    let mask = image.data;
                    image.data = mask
                        .iter()
                        .flat_map(|x| sRGB32::new(255, 255, 255, *x).as_f32().srgb_pre().as_bgra())
                        .collect();
                }
                // This is in sRGB RGBA format but our texture atlas is in pre-multiplied sRGB BGRA format, so swap it
                SwashContent::Color => {
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
                SwashContent::SubpixelMask => {
                    // TODO: wide doesn't implement SSE shuffle instructions yet, which could potentially be faster here
                    let len = image.data.len() / 4;
                    let slice = image.data.as_mut_slice();
                    for i in 0..len {
                        let idx = i * 4;
                        slice.swap(idx + 0, idx + 2);
                        // Don't pre-multiply this because it's already a mask
                    }
                }
            }

            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &atlas.get_texture(),
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
                    width: image.placement.width as u32,
                    height: image.placement.height as u32,
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
        key: CacheKey,
        font_system: &mut FontSystem,
        glyphs: &mut GlyphCache,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: &mut Atlas,
        cache: &mut SwashCache,
    ) -> Result<Option<super::compositor::Data>, Error> {
        Self::write_glyph(key, font_system, glyphs, device, queue, atlas, cache)?;
        let glyph = Self::get_glyph(key, glyphs).ok_or(Error::GlyphCacheFailure)?;
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
            texclip: ((glyph.region.index as u32) << 16) | 0,
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
        cache: &mut SwashCache,
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

            start_y_physical <= bounds_bottom && bounds_top <= end_y_physical
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
                    physical_glyph.cache_key,
                    font_system,
                    glyphs,
                    device,
                    queue,
                    atlas,
                    cache,
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
            self.text_buffer
                .borrow()
                .as_ref()
                .ok_or(Error::InternalFailure)?,
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
