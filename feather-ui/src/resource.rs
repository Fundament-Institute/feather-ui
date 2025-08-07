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

/// An empty size request equates to requesting the native size of the image. One axis being zero equates to requesting the
/// equivelent aspect ratio from tha native aspect ratio.
#[inline]
pub fn fill_size(size: guillotiere::Size, native: guillotiere::Size) -> guillotiere::Size {
    match (size.width, size.height) {
        (0, 0) => native,
        (x, 0) => guillotiere::Size::new(
            x,
            (native.height as f32 * (x as f32 / native.width as f32)).round() as i32,
        ),
        (0, y) => guillotiere::Size::new(
            (native.width as f32 * (y as f32 / native.height as f32)).round() as i32,
            y,
        ),
        _ => size,
    }
}

pub trait Location: crate::DynHashEq + Any + Send + Sync {
    fn fetch(&self) -> Result<Box<dyn Loader>, Error>;
}

dyn_clone::clone_trait_object!(Location);

pub trait Loader: std::fmt::Debug + Send + Sync {
    fn load(
        &self,
        driver: &crate::graphics::Driver,
        size: guillotiere::Size,
        dpi: f32,
        resize: bool,
    ) -> Result<(atlas::Region, guillotiere::Size), Error>;
}

impl Hash for dyn Location {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}

impl PartialEq for dyn Location {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other as &dyn Any)
    }
}

impl Eq for dyn Location {}

#[derive(Debug)]
// resvg requires the DPI when it parses the XML, and we can't store the XML tree directly, so we store the string instead.
struct SVGXML(String);

impl Location for std::path::PathBuf {
    fn fetch(&self) -> Result<Box<dyn Loader>, Error> {
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

        // We start by guessing the format from ImageReader, because it only reads a maximum of 16 bytes from the file.
        {
            let image = image::ImageReader::open(&self)?.with_guessed_format()?;

            match image.format() {
                Some(image::ImageFormat::Png) | Some(image::ImageFormat::Jpeg) => (),
                _ => {
                    return Ok(Box::new(
                        image
                            .decode()
                            .map_err(|e| Error::ResourceError(Box::new(e)))?,
                    ));
                }
            }
        }

        // If we get here, it's a PNG or JPEG, so we drop the image to close the file handle, then load it with load_image instead
        // so we can correctly convert it to sRGB color space using the embedded color profile information.
        Ok(Box::new(
            load_image::load_path(&self).map_err(|e| Error::ResourceError(Box::new(e)))?,
        ))
    }
}

impl Loader for SVGXML {
    fn load(
        &self,
        driver: &crate::graphics::Driver,
        size: guillotiere::Size,
        dpi: f32,
        _resize: bool,
    ) -> Result<(atlas::Region, guillotiere::Size), Error> {
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

        queue_atlas_data(
            pixmap.data(),
            &region,
            &driver.queue,
            w,
            h,
            &driver.atlas.read(),
        );

        // TODO: generate mipmaps
        Ok((
            region,
            guillotiere::Size::new(svg_size.width() as i32, svg_size.height() as i32),
        ))
    }
}

struct LoadImageView<'a>(&'a load_image::Image);

fn image_data_as_bytes(data: &load_image::ImageData) -> &[u8] {
    use load_image::export::rgb::ComponentBytes;
    match data {
        load_image::ImageData::RGB8(rgbs) => rgbs.as_slice().as_bytes(),
        load_image::ImageData::RGBA8(rgbas) => rgbas.as_slice().as_bytes(),
        load_image::ImageData::RGB16(rgbs) => rgbs.as_slice().as_bytes(),
        load_image::ImageData::RGBA16(rgbas) => rgbas.as_slice().as_bytes(),
        load_image::ImageData::GRAY8(gray_v08s) => gray_v08s.as_slice().as_bytes(),
        load_image::ImageData::GRAY16(gray_v08s) => gray_v08s.as_slice().as_bytes(),
        load_image::ImageData::GRAYA8(gray_alpha_v08s) => gray_alpha_v08s.as_slice().as_bytes(),
        load_image::ImageData::GRAYA16(gray_alpha_v08s) => gray_alpha_v08s.as_slice().as_bytes(),
    }
}

impl<'a> IntoImageView for LoadImageView<'a> {
    fn pixel_type(&self) -> Option<fast_image_resize::PixelType> {
        use fast_image_resize::PixelType;
        use load_image::ImageData;
        Some(match self.0.bitmap {
            ImageData::RGB8(_) => PixelType::U8x3,
            ImageData::RGBA8(_) => PixelType::U8x4,
            ImageData::GRAY8(_) => PixelType::U8,
            ImageData::GRAYA8(_) => PixelType::U8x2,
            ImageData::RGB16(_) => PixelType::U16x3,
            ImageData::RGBA16(_) => PixelType::U16x4,
            ImageData::GRAY16(_) => PixelType::U16,
            ImageData::GRAYA16(_) => PixelType::U16x2,
        })
    }

    fn width(&self) -> u32 {
        self.0.width as u32
    }

    fn height(&self) -> u32 {
        self.0.height as u32
    }

    fn image_view<P: fast_image_resize::PixelTrait>(
        &self,
    ) -> Option<impl fast_image_resize::ImageView<Pixel = P>> {
        if P::pixel_type() == self.pixel_type().unwrap() {
            return fast_image_resize::images::TypedImageRef::<P>::from_buffer(
                self.width(),
                self.height(),
                image_data_as_bytes(&self.0.bitmap),
            )
            .ok();
        }
        None
    }
}

fn process_pixels<P: fast_image_resize::pixels::InnerPixel>(
    output: &mut [u8],
    dst_image: &fast_image_resize::images::Image<'_>,
    convert: fn(p: &P) -> crate::color::sRGB,
) {
    use crate::color::Premultiplied;
    for (p, c) in dst_image
        .typed_image::<P>()
        .unwrap()
        .pixels()
        .iter()
        .zip(output.chunks_exact_mut(4))
    {
        c.copy_from_slice(&convert(p).srgb_pre().as_bgra())
    }
}

fn process_load_pixels<T>(
    output: &mut [u8],
    source: &Vec<T>,
    convert: fn(p: &T) -> crate::color::sRGB,
) {
    use crate::color::Premultiplied;
    for (p, c) in source.iter().zip(output.chunks_exact_mut(4)) {
        // Pre-multiply color, then extract in BGRA form.
        c.copy_from_slice(&convert(p).srgb_pre().as_bgra());
    }
}

fn image_resize_loader(
    inner_format: fast_image_resize::PixelType,
    srgb_map: fast_image_resize::PixelComponentMapper,
    source: &impl IntoImageView,
    size: guillotiere::Size,
) -> Result<(Vec<u8>, u32, u32), Error> {
    let mut inner_src_image =
        fast_image_resize::images::Image::new(source.width(), source.height(), inner_format);

    srgb_map
        .forward_map(source, &mut inner_src_image)
        .map_err(|e| Error::ResourceError(Box::new(e)))?;

    let mut dst_image =
        fast_image_resize::images::Image::new(size.width as u32, size.height as u32, inner_format);

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
        vec![std::mem::zeroed::<u8>(); (dst_image.width() * dst_image.height() * 4) as usize]
    };

    match inner_format {
        fast_image_resize::PixelType::U16 => process_pixels::<fast_image_resize::pixels::U16>(
            output.as_mut_slice(),
            &dst_image,
            |p| sRGB64::new(p.0, p.0, p.0, u16::MAX).as_f32(),
        ),
        fast_image_resize::PixelType::U16x2 => process_pixels::<fast_image_resize::pixels::U16x2>(
            output.as_mut_slice(),
            &dst_image,
            |p| sRGB64::new(p.0[0], p.0[0], p.0[0], p.0[1]).as_f32(),
        ),
        fast_image_resize::PixelType::U16x3 => process_pixels::<fast_image_resize::pixels::U16x3>(
            output.as_mut_slice(),
            &dst_image,
            |p| sRGB64::new(p.0[0], p.0[1], p.0[2], u16::MAX).as_f32(),
        ),
        fast_image_resize::PixelType::U16x4 => process_pixels::<fast_image_resize::pixels::U16x4>(
            output.as_mut_slice(),
            &dst_image,
            |p| sRGB64::new(p.0[0], p.0[1], p.0[2], p.0[3]).as_f32(),
        ),
        _ => return Err(Error::InternalFailure),
    }

    Ok((output, dst_image.width(), dst_image.height()))
}

impl Loader for load_image::Image {
    fn load(
        &self,
        driver: &crate::graphics::Driver,
        mut size: guillotiere::Size,
        _: f32,
        resize: bool,
    ) -> Result<(atlas::Region, guillotiere::Size), Error> {
        use crate::color::sRGB32;

        let native = guillotiere::Size::new(self.width as i32, self.height as i32);
        size = fill_size(
            size,
            guillotiere::Size::new(self.width as i32, self.height as i32),
        );

        // If we're too close to the native size of the image, skip resizing it and simply store the native size to the atlas.
        let force_native = within_variance(size.height, self.height as i32, 0.05)
            && within_variance(size.width, self.width as i32, 0.05);
        let (raw, w, h) = if !force_native
            && !(size.width as usize > self.width && size.height as usize > self.height)
        {
            use load_image::ImageData;

            let srgb_map = match self.bitmap {
                ImageData::RGB8(_)
                | ImageData::RGBA8(_)
                | ImageData::RGB16(_)
                | ImageData::RGBA16(_) => fast_image_resize::create_srgb_mapper(),
                ImageData::GRAY8(_)
                | ImageData::GRAY16(_)
                | ImageData::GRAYA8(_)
                | ImageData::GRAYA16(_) => fast_image_resize::create_gamma_22_mapper(),
            };

            let inner_format = match self.bitmap {
                ImageData::RGB8(_) | ImageData::RGB16(_) => fast_image_resize::PixelType::U16x3,
                ImageData::RGBA8(_) | ImageData::RGBA16(_) => fast_image_resize::PixelType::U16x4,
                ImageData::GRAY8(_) | ImageData::GRAY16(_) => fast_image_resize::PixelType::U16,
                ImageData::GRAYA8(_) | ImageData::GRAYA16(_) => fast_image_resize::PixelType::U16x2,
            };

            image_resize_loader(inner_format, srgb_map, &LoadImageView(&self), size)?
        } else {
            let mut output =
                unsafe { vec![std::mem::zeroed::<u8>(); self.width * self.height * 4] };

            match &self.bitmap {
                load_image::ImageData::RGB8(rgbs) => {
                    process_load_pixels(output.as_mut_slice(), rgbs, |p| {
                        sRGB32::new(p.r, p.g, p.b, u8::MAX).as_f32()
                    })
                }
                load_image::ImageData::RGBA8(rgbas) => {
                    process_load_pixels(output.as_mut_slice(), rgbas, |p| {
                        sRGB32::new(p.r, p.g, p.b, p.a).as_f32()
                    })
                }
                load_image::ImageData::RGB16(rgbs) => {
                    process_load_pixels(output.as_mut_slice(), rgbs, |p| {
                        sRGB64::new(p.r, p.g, p.b, u16::MAX).as_f32()
                    })
                }
                load_image::ImageData::RGBA16(rgbas) => {
                    process_load_pixels(output.as_mut_slice(), rgbas, |p| {
                        sRGB64::new(p.r, p.g, p.b, p.a).as_f32()
                    })
                }
                load_image::ImageData::GRAY8(gray_v08s) => {
                    process_load_pixels(output.as_mut_slice(), gray_v08s, |p| {
                        sRGB32::new(p.value(), p.value(), p.value(), u8::MAX).as_f32()
                    })
                }
                load_image::ImageData::GRAY16(gray_v08s) => {
                    process_load_pixels(output.as_mut_slice(), gray_v08s, |p| {
                        sRGB64::new(p.value(), p.value(), p.value(), u16::MAX).as_f32()
                    })
                }
                load_image::ImageData::GRAYA8(gray_alpha_v08s) => {
                    process_load_pixels(output.as_mut_slice(), gray_alpha_v08s, |p| {
                        sRGB32::new(p.v, p.v, p.v, p.a).as_f32()
                    })
                }
                load_image::ImageData::GRAYA16(gray_alpha_v08s) => {
                    process_load_pixels(output.as_mut_slice(), gray_alpha_v08s, |p| {
                        sRGB64::new(p.v, p.v, p.v, p.a).as_f32()
                    })
                }
            };

            (output, self.width as u32, self.height as u32)
        };

        let region = driver
            .atlas
            .write()
            .reserve(&driver.device, guillotiere::Size::new(w as i32, h as i32))?;

        queue_atlas_data(&raw, &region, &driver.queue, w, h, &driver.atlas.read());

        // TODO: We generate mipmaps by queueing a "generate mipmaps" operation, which is done by queueing up a series of compositor operations
        // that operate on the atlas itself.
        Ok((region, native))
    }
}

impl Loader for image::DynamicImage {
    fn load(
        &self,
        driver: &crate::graphics::Driver,
        mut size: guillotiere::Size,
        _: f32,
        _resize: bool,
    ) -> Result<(atlas::Region, guillotiere::Size), Error> {
        use crate::color::Premultiplied;
        use crate::color::sRGB32;

        let native = guillotiere::Size::new(self.width() as i32, self.height() as i32);
        size = fill_size(size, native);

        // If we're too close to the native size of the image, skip resizing it and simply store the native size to the atlas.
        let force_native = within_variance(size.height, self.height() as i32, 0.05)
            && within_variance(size.width, self.width() as i32, 0.05);

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

            image_resize_loader(inner_format, srgb_map, self, size)?
        } else {
            let mut raw = self.to_rgba8().into_vec();

            // Raw is in sRGB RGBA but our atlas is in pre-multiplied BGRA format
            for c in raw.as_mut_slice().chunks_exact_mut(4) {
                // Pre-multiply color, then extract in BGRA form.
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

        queue_atlas_data(&raw, &region, &driver.queue, w, h, &driver.atlas.read());

        // TODO: We generate mipmaps by queueing a "generate mipmaps" operation, which is done by queueing up a series of compositor operations
        // that operate on the atlas itself.
        Ok((region, native))
    }
}

pub(crate) fn queue_atlas_data(
    data: &[u8],
    region: &atlas::Region,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    atlas: &atlas::Atlas,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: atlas.get_texture(),
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: region.uv.min.x as u32,
                y: region.uv.min.y as u32,
                z: region.index as u32,
            },
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * atlas.get_texture().format().components() as u32),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}
