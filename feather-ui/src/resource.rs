// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::any::Any;
use std::fs::File;

use resvg::{tiny_skia, usvg};

use crate::Error;
use crate::color::sRGB64;
use crate::render::atlas;
use fast_image_resize::{IntoImageView, ResizeOptions};
use std::hash::Hash;

pub(crate) const MIN_AREA: i32 = 64 * 64;
pub(crate) const MAX_VARIANCE: f32 = 0.1;

pub(crate) fn within_variance(l: i32, r: i32, range: f32) -> bool {
    let diff = l as f32 / r as f32;
    diff > (1.0 - range) && diff < (1.0 + range)
}

pub trait ResourceLocation: crate::DynHashEq + Any + Send + Sync {
    fn fetch(&self) -> Result<Box<dyn ResourceLoader>, Error>;
}

dyn_clone::clone_trait_object!(ResourceLocation);

pub trait ResourceLoader: std::fmt::Debug + Send + Sync {
    fn load(
        &self,
        driver: &crate::graphics::Driver,
        size: &guillotiere::Size,
        dpi: f32,
        resize: bool,
    ) -> Result<atlas::Region, Error>;
}

impl Hash for dyn ResourceLocation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}

impl PartialEq for dyn ResourceLocation {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other as &dyn Any)
    }
}

impl Eq for dyn ResourceLocation {}

#[derive(Debug)]
// resvg requires the DPI when it parses the XML, and we can't store the XML tree directly, so we store the string instead.
struct SVGXML(String);

impl ResourceLocation for std::path::PathBuf {
    fn fetch(&self) -> Result<Box<dyn ResourceLoader>, Error> {
        use std::io::Read;

        if let Some(extension) = self.extension() {
            let ext = extension.to_str().unwrap().to_ascii_lowercase();
            if &ext == "svgz" {
                let mut buf = Vec::new();
                {
                    let mut f = File::open(&self)?;
                    f.read_to_end(&mut buf)?;
                }

                if buf.starts_with(&[0x1f, 0x8b]) {
                    buf = usvg::decompress_svgz(&buf)
                        .map_err(|e| Error::ResourceError(Box::new(e)))?;
                }

                return Ok(Box::new(SVGXML(
                    String::from_utf8(buf).map_err(|e| Error::ResourceError(Box::new(e)))?,
                )));
            } else if &ext == "svg" {
                let mut buf = String::new();
                File::open(&self)?.read_to_string(&mut buf)?;
                return Ok(Box::new(SVGXML(buf)));
            }
        }
        {
            let mut f = File::open(&self)?;
            let mut header = [0_u8; 2];
            if f.read(&mut header)? == 2 && header == [0x1f_u8, 0x8b_u8] {
                let mut buf: Vec<u8> = header.into();
                f.read_to_end(&mut buf)?;
                buf = usvg::decompress_svgz(&buf).map_err(|e| Error::ResourceError(Box::new(e)))?;

                return Ok(Box::new(SVGXML(
                    String::from_utf8(buf).map_err(|e| Error::ResourceError(Box::new(e)))?,
                )));
            }
        }
        Ok(Box::new(
            image::ImageReader::open(&self)?
                .with_guessed_format()?
                .decode()
                .map_err(|e| Error::ResourceError(Box::new(e)))?,
        ))
    }
}

impl ResourceLoader for SVGXML {
    fn load(
        &self,
        driver: &crate::graphics::Driver,
        size: &guillotiere::Size,
        dpi: f32,
        _resize: bool,
    ) -> Result<atlas::Region, Error> {
        use crate::color::Premultiplied;
        use crate::color::sRGB32;

        let xml_opt = usvg::roxmltree::ParsingOptions {
            allow_dtd: true,
            ..Default::default()
        };
        let xml_tree = usvg::roxmltree::Document::parse_with_options(&self.0, xml_opt)
            .map_err(|e| Error::ResourceError(Box::new(e)))?;

        let svg_opt = usvg::Options {
            dpi,
            font_size: 12.0, // TODO: change this based on system text-scaling property.
            shape_rendering: usvg::ShapeRendering::CrispEdges,
            default_size: tiny_skia::Size::from_wh(size.width as f32, size.height as f32).unwrap(),
            ..Default::default()
        };

        let svg = usvg::Tree::from_xmltree(&xml_tree, &svg_opt)
            .map_err(|e| Error::ResourceError(Box::new(e)))
            .map(|x| Box::new(x))?;

        let svg_size = svg.size().to_int_size();
        // If we provided an intended draw size, smoosh the SVG into that size.
        let (t, w, h) = match (size.width, size.height) {
            (0, 0) => (
                tiny_skia::Transform::identity(),
                svg_size.width(),
                svg_size.height(),
            ),
            (x, 0) => {
                let scale = x as f32 / svg.size().width();
                (
                    tiny_skia::Transform::from_scale(scale, scale),
                    x as u32,
                    (svg.size().height() * scale) as u32,
                )
            }
            (0, y) => {
                let scale = y as f32 / svg.size().height();
                (
                    tiny_skia::Transform::from_scale(scale, scale),
                    (svg.size().width() * scale) as u32,
                    y as u32,
                )
            }
            (x, y) => (
                tiny_skia::Transform::from_scale(
                    x as f32 / svg.size().width(),
                    y as f32 / svg.size().height(),
                ),
                x as u32,
                y as u32,
            ),
        };

        let mut pixmap = tiny_skia::Pixmap::new(w, h).unwrap();

        resvg::render(&svg, t, &mut pixmap.as_mut());

        // Here we flip this into our BGRA representation
        for c in pixmap.data_mut().chunks_exact_mut(4) {
            // Pre-multiply color, then extract in BGRA form.
            c.copy_from_slice(
                &sRGB32::new(c[0], c[1], c[2], c[3])
                    .as_f32()
                    .srgb_pre()
                    .as_bgra(),
            );
        }

        let region = driver
            .atlas
            .write()
            .reserve(&driver.device, guillotiere::Size::new(w as i32, h as i32))?;

        driver.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: driver.atlas.read().get_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region.uv.min.x as u32,
                    y: region.uv.min.y as u32,
                    z: region.index as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            pixmap.data(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(
                    w * driver.atlas.read().get_texture().format().components() as u32,
                ),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        // TODO: generate mipmaps
        Ok(region)
    }
}

impl ResourceLoader for image::DynamicImage {
    fn load(
        &self,
        driver: &crate::graphics::Driver,
        size: &guillotiere::Size,
        _: f32,
        _resize: bool,
    ) -> Result<atlas::Region, Error> {
        use crate::color::Premultiplied;
        use crate::color::sRGB32;

        // If we're too close to the native size of the image, skip resizing it and simply store the native size to the atlas.
        let force_native = within_variance(size.height, self.height() as i32, 0.05)
            && within_variance(size.width, self.width() as i32, 0.05);

        let aspect = self.width() as f32 / self.height() as f32;
        let mut size = *size;

        if size.width == 0 {
            size.width = (size.height as f32 / aspect).round() as i32;
        }
        if size.height == 0 {
            size.height = (size.width as f32 * aspect).round() as i32;
        }
        let (raw, w, h) = if !force_native
            && !(size.width as u32 > self.width() && size.height as u32 > self.height())
        {
            let srgb_map = if self.color().has_color() {
                fast_image_resize::create_srgb_mapper()
            } else {
                fast_image_resize::create_gamma_22_mapper()
            };

            let inner_format = match self.color().channel_count() {
                4 => fast_image_resize::PixelType::U16x4,
                3 => fast_image_resize::PixelType::U16x3,
                2 => fast_image_resize::PixelType::U16x2,
                1 => fast_image_resize::PixelType::U16,
                _ => return Err(Error::InternalFailure),
            };

            let mut inner_src_image =
                fast_image_resize::images::Image::new(self.width(), self.height(), inner_format);

            srgb_map
                .forward_map(self, &mut inner_src_image)
                .map_err(|e| Error::ResourceError(Box::new(e)))?;

            let mut dst_image = fast_image_resize::images::Image::new(
                size.width as u32,
                size.height as u32,
                inner_format,
            );

            fast_image_resize::Resizer::new()
                .resize(
                    &inner_src_image,
                    &mut dst_image,
                    &ResizeOptions::new().use_alpha(true).resize_alg(
                        fast_image_resize::ResizeAlg::Convolution(
                            fast_image_resize::FilterType::CatmullRom,
                        ),
                    ),
                )
                .map_err(|e| Error::ResourceError(Box::new(e)))?;

            srgb_map
                .backward_map_inplace(&mut dst_image)
                .map_err(|e| Error::ResourceError(Box::new(e)))?;

            let mut output = unsafe {
                vec![
                    std::mem::zeroed::<u8>();
                    (dst_image.width() * dst_image.height() * 4) as usize
                ]
            };

            match inner_format {
                fast_image_resize::PixelType::U16 => {
                    for (p, c) in dst_image
                        .typed_image::<fast_image_resize::pixels::U16>()
                        .unwrap()
                        .pixels()
                        .iter()
                        .zip(output.as_mut_slice().chunks_exact_mut(4))
                    {
                        c.copy_from_slice(
                            &sRGB64::new(p.0, p.0, p.0, u16::MAX)
                                .as_f32()
                                .srgb_pre()
                                .as_bgra(),
                        )
                    }
                }
                fast_image_resize::PixelType::U16x2 => {
                    for (p, c) in dst_image
                        .typed_image::<fast_image_resize::pixels::U16x2>()
                        .unwrap()
                        .pixels()
                        .iter()
                        .zip(output.as_mut_slice().chunks_exact_mut(4))
                    {
                        c.copy_from_slice(
                            &sRGB64::new(p.0[0], p.0[0], p.0[0], p.0[1])
                                .as_f32()
                                .srgb_pre()
                                .as_bgra(),
                        )
                    }
                }
                fast_image_resize::PixelType::U16x3 => {
                    for (p, c) in dst_image
                        .typed_image::<fast_image_resize::pixels::U16x3>()
                        .unwrap()
                        .pixels()
                        .iter()
                        .zip(output.as_mut_slice().chunks_exact_mut(4))
                    {
                        c.copy_from_slice(
                            &sRGB64::new(p.0[0], p.0[1], p.0[2], u16::MAX)
                                .as_f32()
                                .srgb_pre()
                                .as_bgra(),
                        )
                    }
                }
                fast_image_resize::PixelType::U16x4 => {
                    for (p, c) in dst_image
                        .typed_image::<fast_image_resize::pixels::U16x4>()
                        .unwrap()
                        .pixels()
                        .iter()
                        .zip(output.as_mut_slice().chunks_exact_mut(4))
                    {
                        c.copy_from_slice(
                            &sRGB64::new(p.0[0], p.0[1], p.0[2], p.0[3])
                                .as_f32()
                                .srgb_pre()
                                .as_bgra(),
                        )
                    }
                }
                _ => return Err(Error::InternalFailure),
            }

            (output, dst_image.width(), dst_image.height())
        } else {
            let mut raw = self.to_rgba8().into_vec();

            // Raw is in sRGB RGBA but our atlas is in pre-multiplied BGRA format
            for c in raw.as_mut_slice().chunks_exact_mut(4) {
                // Pre-multiply color, then extract in BGRA form.
                println!("{c:?}");
                c.copy_from_slice(
                    &sRGB32::new(c[0], c[1], c[2], c[3])
                        .as_f32()
                        .srgb_pre()
                        .as_bgra(),
                );
            }

            (raw, self.width(), self.height())
        };

        let region = driver
            .atlas
            .write()
            .reserve(&driver.device, guillotiere::Size::new(w as i32, h as i32))?;

        driver.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: driver.atlas.read().get_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region.uv.min.x as u32,
                    y: region.uv.min.y as u32,
                    z: region.index as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &raw,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(
                    w * driver.atlas.read().get_texture().format().components() as u32,
                ),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        // TODO: We generate mipmaps by queueing a "generate mipmaps" operation, which is done by queueing up a series of compositor operations
        // that operate on the atlas itself.
        Ok(region)
    }
}
