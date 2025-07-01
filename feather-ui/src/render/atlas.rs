// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::collections::HashMap;
use std::rc::Rc;

use guillotiere::{AllocId, AllocatorOptions, AtlasAllocator, Size};
use wgpu::util::DeviceExt;
use wgpu::{Extent3d, TextureDescriptor, TextureFormat, TextureUsages};

use crate::Error;

/// Array of 2D textures, along with an array of guillotine allocators to go along with them. We use an array of mid-size
/// textures so we don't have to resize the atlas allocator or adjust any UV coordinates that way.
#[derive_where::derive_where(Debug)]
pub struct Atlas {
    extent: u32,
    pub extent_buf: wgpu::Buffer,
    pub mvp: wgpu::Buffer, // Matrix for rendering *onto* the texture atlas (the compositor has it's own for rendering to the screen)
    texture: wgpu::Texture,
    #[derive_where(skip)]
    allocators: Vec<AtlasAllocator>,
    cache: HashMap<Rc<crate::SourceID>, Region>,
    pub view: Rc<wgpu::TextureView>, // Stores a view into the atlas texture. Compositors take a weak reference to this that is invalidated when they need to rebind
}

// TODO: Should be possible to define an HDR pipeline with 16-bit channels
pub const ATLAS_FORMAT: TextureFormat = TextureFormat::Bgra8UnormSrgb;

unsafe impl Send for Atlas {}
unsafe impl Sync for Atlas {}

impl Drop for Atlas {
    fn drop(&mut self) {
        for (_, mut r) in self.cache.drain() {
            r.id = AllocId::deserialize(u32::MAX);
        }
    }
}

impl Atlas {
    fn create_view(t: &wgpu::Texture) -> wgpu::TextureView {
        t.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Atlas View"),
            format: Some(crate::render::atlas::ATLAS_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_array_layer: 0,
            array_layer_count: None,
            ..Default::default()
        })
    }

    /// Creates an encoder that resizes the texture atlas, submits this to the work queue. Does not wait until
    /// queue finishes processing as this shouldn't be necessary.
    pub fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layers: u32) {
        let mut texture = Self::create_texture(device, self.extent, layers);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Atlas Resize Encoder"),
        });

        encoder.copy_texture_to_texture(
            self.texture.as_image_copy(),
            texture.as_image_copy(),
            self.texture.size(),
        );

        std::mem::swap(&mut texture, &mut self.texture);
        // We swap the actual Rc object here to ensure the weak references get destroyed
        let mut view = Rc::new(Self::create_view(&self.texture));
        std::mem::swap(&mut self.view, &mut view);

        queue.write_buffer(
            &self.mvp,
            0,
            crate::graphics::mat4_ortho(
                0.0,
                self.texture.height() as f32,
                self.texture.width() as f32,
                -(self.texture.height() as f32),
                1.0,
                10000.0,
            )
            .as_byte_slice(),
        );

        queue.write_buffer(&self.extent_buf, 0, &self.extent.to_ne_bytes());
        queue.submit(Some(encoder.finish()));
        // device.poll(PollType::Wait); // shouldn't be needed as long as queues after this refer to the correct texture
        texture.destroy(); // TODO: This *should* be legal as long as we submit the queue without needing to wait
        assert_eq!(Rc::strong_count(&view), 1); // This MUST be dropped because we rely on this to signal the compositors to rebind
    }

    pub fn new(device: &wgpu::Device, extent: u32) -> Self {
        let extent = device.limits().max_texture_dimension_2d.min(extent);
        let texture = Self::create_texture(device, extent, 1);
        let allocator = Self::create_allocator(extent);

        let mvp = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Atlas MVP"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: crate::graphics::mat4_ortho(
                0.0,
                texture.height() as f32,
                texture.width() as f32,
                -(texture.height() as f32),
                1.0,
                10000.0,
            )
            .as_byte_slice(),
        });

        let extent_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Extent"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(&[extent]),
        });

        Self {
            extent,
            view: Rc::new(Self::create_view(&texture)),
            texture,
            allocators: vec![allocator],
            extent_buf,
            mvp,
            cache: HashMap::new(),
        }
    }

    fn create_allocator(extent: u32) -> AtlasAllocator {
        AtlasAllocator::with_options(
            Size::new(extent as i32, extent as i32),
            &AllocatorOptions {
                large_size_threshold: 512,
                small_size_threshold: 16,
                ..Default::default()
            },
        )
    }

    pub fn get_texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    fn create_texture(device: &wgpu::Device, extent: u32, count: u32) -> wgpu::Texture {
        device.create_texture(&TextureDescriptor {
            label: Some("Feather Atlas"),
            size: Extent3d {
                width: extent,
                height: extent,
                depth_or_array_layers: count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ATLAS_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[TextureFormat::Bgra8UnormSrgb, TextureFormat::Bgra8Unorm],
        })
    }

    fn create_region(&self, idx: usize, r: guillotiere::Allocation, dim: Size) -> Region {
        let mut uv = r.rectangle;
        uv.set_size(dim);
        Region {
            id: r.id,
            uv,
            index: idx as u16,
        }
    }

    pub fn get_region(&self, id: Rc<crate::SourceID>) -> Option<&Region> {
        self.cache.get(&id)
    }

    pub fn cache_region(
        &mut self,
        device: &wgpu::Device,
        id: Rc<crate::SourceID>,
        dim: Size,
    ) -> Result<&Region, Error> {
        let uv = self.cache.get(&id).map(|x| x.uv);

        if let Some(old) = uv {
            if old.size() != dim {
                if let Some(mut region) = self.cache.remove(&id) {
                    self.destroy(&mut region);
                }
                let region = self.reserve(device, dim)?;
                self.cache.insert(id.clone(), region);
            }
        } else {
            let region = self.reserve(device, dim)?;
            self.cache.insert(id.clone(), region);
        }

        self.cache.get(&id).ok_or(Error::AtlasCacheFailure)
    }

    pub fn reserve(&mut self, device: &wgpu::Device, dim: Size) -> Result<Region, Error> {
        if dim.height == 0 {
            assert_ne!(dim.height, 0);
        }
        assert_ne!(dim.width, 0);
        assert_ne!(dim.height, 0);

        for (idx, a) in self.allocators.iter_mut().enumerate() {
            if let Some(r) = a.allocate(dim) {
                return Ok(self.create_region(idx, r, dim));
            }
        }

        // If we run out of room, try adding another layer
        self.grow(device)?;

        if dim.width > self.extent as i32 || dim.height > self.extent as i32 {
            panic!(
                "Requested reservation size ({},{}) exceeds size of texture atlas ({},{})!",
                dim.width, dim.height, self.extent, self.extent
            )
        }

        if let Some(r) = self
            .allocators
            .last_mut()
            .ok_or(Error::InternalFailure)?
            .allocate(dim)
        {
            return Ok(self.create_region(self.allocators.len() - 1, r, dim));
        }

        Err(Error::AtlasReservationFailure)
    }

    pub fn destroy(&mut self, region: &mut Region) {
        self.allocators[region.index as usize].deallocate(region.id);
        region.id = AllocId::deserialize(u32::MAX);
    }

    // This always triggers a resize error telling us to abort the current frame render
    fn grow(&mut self, device: &wgpu::Device) -> Result<u32, Error> {
        if (self.extent * 2) <= device.limits().max_texture_dimension_2d {
            self.extent *= 2;
            self.allocators
                .first_mut()
                .ok_or(Error::InternalFailure)?
                .grow(Size::new(self.extent as i32, self.extent as i32));
        } else {
            self.allocators.push(Self::create_allocator(self.extent));
        }

        Err(Error::ResizeTextureAtlas(self.allocators.len() as u32))
    }

    pub fn draw(&self, driver: &crate::graphics::Driver, encoder: &mut wgpu::CommandEncoder) {
        // We create one render pass for each layer of the atlas
        for i in 0..self.texture.depth_or_array_layers() {
            let view = self.texture.create_view(&wgpu::wgt::TextureViewDescriptor {
                label: Some("Atlas Layer View"),
                format: Some(ATLAS_FORMAT),
                dimension: Some(wgpu::TextureViewDimension::D2),
                usage: Some(TextureUsages::RENDER_ATTACHMENT),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: i,
                array_layer_count: Some(1),
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Atlas Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_viewport(
                0.0,
                0.0,
                self.texture.width() as f32,
                self.texture.height() as f32,
                0.0,
                1.0,
            );

            for (_, pipeline) in driver.pipelines.write().iter_mut() {
                pipeline.draw(driver, &mut pass, i as u16);
            }
        }
    }
}

#[derive(Debug)]
/// A single allocated region on a particular texture atlas. Because we never resize our texture atlases, we can simply
/// store the UV rect directly
pub struct Region {
    pub id: AllocId,
    pub uv: guillotiere::Rectangle,
    pub index: u16,
}

impl Drop for Region {
    fn drop(&mut self) {
        if self.id != AllocId::deserialize(u32::MAX) {
            panic!(
                "dropped a region without deallocating it! Regions can't automatically deallocate themselves, put them inside an object that can store a reference to the Atlas!"
            )
        }
    }
}
