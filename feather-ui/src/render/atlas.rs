// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::collections::HashMap;
use std::num::NonZero;
use std::sync::Arc;

use guillotiere::{AllocId, AllocatorOptions, AtlasAllocator, Size};
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayoutEntry, Extent3d, TextureDescriptor, TextureFormat, TextureUsages};

use crate::Error;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum AtlasKind {
    Primary,
    Layer0,
    Layer1,
}

/// Array of 2D textures, along with an array of guillotine allocators to go along with them. We use an array of mid-size
/// textures so we don't have to resize the atlas allocator or adjust any UV coordinates that way.
#[derive_where::derive_where(Debug)]
pub struct Atlas {
    extent: u32,
    pub extent_buf: wgpu::Buffer,
    pub mvp: wgpu::Buffer, // Matrix for rendering *onto* the texture atlas (the compositor has it's own for rendering to the screen)
    pub(crate) texture: wgpu::Texture,
    #[derive_where(skip)]
    allocators: Vec<AtlasAllocator>,
    cache: HashMap<Arc<crate::SourceID>, Region>,
    pub view: Arc<wgpu::TextureView>, // Stores a view into the atlas texture. Compositors take a weak reference to this that is invalidated when they need to rebind
    pub targets: Vec<wgpu::TextureView>,
    kind: AtlasKind,
    mipgen: HashMap<u8, (wgpu::Buffer, usize)>,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    mipbindings: HashMap<u32, wgpu::BindGroup>,
    miplayout: wgpu::BindGroupLayout,
    mipsize: wgpu::Buffer,
}

// TODO: Should be possible to define an HDR pipeline with 16-bit channels
pub const ATLAS_FORMAT: TextureFormat = TextureFormat::Bgra8UnormSrgb;
const ATLAS_MIP_LEVELS: u32 = 8;

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

    fn create_target(t: &wgpu::Texture, i: u32, mip: u32) -> wgpu::TextureView {
        let name = format!("Atlas Layer {i} Target");
        t.create_view(&wgpu::wgt::TextureViewDescriptor {
            label: Some(&name),
            format: Some(ATLAS_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: mip,
            mip_level_count: Some(1),
            base_array_layer: i,
            array_layer_count: Some(1),
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
        let mut view = Arc::new(Self::create_view(&self.texture));
        std::mem::swap(&mut self.view, &mut view);

        queue.write_buffer(
            &self.mvp,
            0,
            bytemuck::cast_slice(
                &crate::graphics::mat4_ortho(
                    0.0,
                    self.texture.height() as f32,
                    self.texture.width() as f32,
                    -(self.texture.height() as f32),
                    1.0,
                    10000.0,
                )
                .to_array(),
            ),
        );

        queue.write_buffer(
            &self.mipsize,
            0,
            bytemuck::bytes_of(&[self.texture.width() as f32, self.texture.height() as f32]),
        );

        queue.write_buffer(&self.extent_buf, 0, &self.extent.to_ne_bytes());
        queue.submit(Some(encoder.finish()));
        // device.poll(PollType::Wait); // shouldn't be needed as long as queues after this refer to the correct texture
        texture.destroy();
        assert_eq!(Arc::strong_count(&view), 1); // This MUST be dropped because we rely on this to signal the compositors to rebind

        self.targets.clear();
        self.mipbindings.clear();

        for mip in 0..self.texture.mip_level_count() {
            for i in 0..self.texture.depth_or_array_layers() {
                self.targets
                    .push(Self::create_target(&self.texture, i, mip));
            }
        }
    }

    /// Create a standard projection matrix suitable for compositing for a texture.
    pub fn create_mvp_from_texture(
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        name: &'_ str,
    ) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(
                &crate::graphics::mat4_ortho(
                    0.0,
                    texture.height() as f32,
                    texture.width() as f32,
                    -(texture.height() as f32),
                    1.0,
                    10000.0,
                )
                .to_array(),
            ),
        })
    }

    pub fn new(device: &wgpu::Device, extent: u32, kind: AtlasKind) -> Self {
        let extent = device.limits().max_texture_dimension_2d.min(extent);
        let texture = Self::create_texture(device, extent, 1);
        let allocator = Self::create_allocator(extent);

        let mvp = Self::create_mvp_from_texture(device, &texture, "Atlas MVP");
        let mipsize = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mipmap Base Size"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::bytes_of(&[texture.width() as f32, texture.height() as f32]),
        });

        let extent_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Extent"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(&[extent]),
        });

        let mut targets = Vec::new();

        for mip in 0..texture.mip_level_count() {
            for i in 0..texture.depth_or_array_layers() {
                targets.push(Self::create_target(&texture, i, mip));
            }
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mipmapper"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mipmap.wgsl").into()),
        });

        let miplayout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mipmap Bind Group"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZero::new(size_of::<crate::Mat4x4>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZero::new(size_of::<crate::graphics::Vec2f>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mipmap Pipeline"),
            bind_group_layouts: &[&miplayout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ATLAS_FORMAT,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Mipmap Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            border_color: Some(wgpu::SamplerBorderColor::TransparentBlack),
            ..Default::default()
        });

        Self {
            extent,
            view: Arc::new(Self::create_view(&texture)),
            texture,
            allocators: vec![allocator],
            extent_buf,
            mvp,
            cache: HashMap::new(),
            targets,
            kind,
            mipgen: HashMap::new(),
            pipeline,
            sampler,
            mipbindings: HashMap::new(),
            miplayout,
            mipsize,
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
        assert!(count < u8::MAX.into());
        device.create_texture(&TextureDescriptor {
            label: Some("Feather Atlas"),
            size: Extent3d {
                width: extent,
                height: extent,
                depth_or_array_layers: count,
            },
            mip_level_count: ATLAS_MIP_LEVELS,
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
        assert!(idx < u8::MAX.into());
        let mut uv = r.rectangle;
        uv.set_size(dim);
        Region {
            id: r.id,
            uv,
            index: idx as u8,
        }
    }

    pub fn get_region(&self, id: Arc<crate::SourceID>) -> Option<&Region> {
        self.cache.get(&id)
    }

    pub fn remove_cache(&mut self, id: &Arc<crate::SourceID>) -> bool {
        if let Some(mut region) = self.cache.remove(id) {
            self.destroy(&mut region);
            true
        } else {
            false
        }
    }

    pub fn cache_region(
        &mut self,
        device: &wgpu::Device,
        id: &Arc<crate::SourceID>,
        dim: Size,
        mipmap: Option<&wgpu::Queue>,
    ) -> Result<&Region, Error> {
        let uv = self.cache.get(id).map(|x| x.uv);

        if let Some(old) = uv {
            if old.size() != dim {
                if let Some(mut region) = self.cache.remove(id) {
                    self.destroy(&mut region);
                }
                let region = self.reserve(device, dim, mipmap)?;
                self.cache.insert(id.clone(), region);
            }
        } else {
            let region = self.reserve(device, dim, mipmap)?;
            self.cache.insert(id.clone(), region);
        }

        self.cache.get(id).ok_or(Error::AtlasCacheFailure)
    }

    pub fn reserve(
        &mut self,
        device: &wgpu::Device,
        dim: Size,
        mipmap: Option<&wgpu::Queue>,
    ) -> Result<Region, Error> {
        if dim.height == 0 {
            assert_ne!(dim.height, 0);
        }
        assert_ne!(dim.width, 0);
        assert_ne!(dim.height, 0);

        for (idx, a) in self.allocators.iter_mut().enumerate() {
            if let Some(r) = a.allocate(dim) {
                let region = self.create_region(idx, r, dim);
                if let Some(queue) = mipmap {
                    self.queue_mip(&region, device, queue);
                }
                return Ok(region);
            }
        }

        // If we run out of room, try adding another layer
        Err(self.grow(device))
    }

    pub fn destroy(&mut self, region: &mut Region) {
        self.allocators[region.index as usize].deallocate(region.id);
        region.id = AllocId::deserialize(u32::MAX);
    }

    // This always triggers a resize error telling us to abort the current frame render
    fn grow(&mut self, device: &wgpu::Device) -> Error {
        if (self.extent * 2) <= device.limits().max_texture_dimension_2d {
            self.extent *= 2;
            if let Some(allocator) = self.allocators.first_mut() {
                allocator.grow(Size::new(self.extent as i32, self.extent as i32));
            } else {
                return Error::InternalFailure;
            }
        } else {
            self.allocators.push(Self::create_allocator(self.extent));
        }

        Error::ResizeTextureAtlas(self.allocators.len() as u32, self.kind)
    }

    fn queue_mip(&mut self, region: &Region, device: &wgpu::Device, queue: &wgpu::Queue) {
        // TODO: In order for this to actual work reliably, the allocated region must be a multiple
        // of 2^8 (for 8 miplevels), or 256, but our current region allocator can't do this.
        let uv = region.uv.to_f32();
        let rect = [uv.min.x, uv.min.y, uv.max.x, uv.max.y];
        self.mipgen.entry(region.index).or_insert_with(|| {
            (
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mipgen data"),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    size: size_of::<[f32; 1024]>() as u64,
                    mapped_at_creation: false,
                }),
                0,
            )
        });

        self.mipgen.entry(region.index).and_modify(|(buf, count)| {
            assert!(*count < 256);
            queue.write_buffer(
                buf,
                (*count * size_of::<[f32; 4]>()) as u64,
                bytemuck::bytes_of(&rect),
            );
            *count += 1;
        });
    }

    pub fn process_mipmaps(
        &mut self,
        driver: &crate::graphics::Driver,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let layers = self.texture.depth_or_array_layers();
        for i in 0..layers {
            if let Some((buf, count)) = self.mipgen.get_mut(&(i as u8)) {
                if *count == 0 {
                    continue;
                }

                for mip in 0..(self.texture.mip_level_count() - 1) {
                    let src = &self.targets[(i + mip * layers) as usize];
                    let dest = &self.targets[(i + (mip + 1) * layers) as usize];

                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Atlas Mipmap Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: dest,
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

                    /*pass.set_viewport(
                        0.0,
                        0.0,
                        self.texture.width() as f32,
                        self.texture.height() as f32,
                        0.0,
                        1.0,
                    );*/

                    let group = self.mipbindings.entry(i + mip * layers).or_insert_with(|| {
                        let bindings = [
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.mvp.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.mipsize.as_entire_binding(),
                            },
                        ];

                        driver.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &self.miplayout,
                            entries: &bindings,
                            label: None,
                        })
                    });

                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &*group, &[]);
                    pass.draw(0..(*count as u32 * 6), 0..1);
                }
                *count = 0;
            }
        }
    }

    pub fn draw(&self, driver: &crate::graphics::Driver, encoder: &mut wgpu::CommandEncoder) {
        // We create one render pass for each layer of the atlas
        for i in 0..self.texture.depth_or_array_layers() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Atlas Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets[i as usize],
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
                pipeline.draw(driver, &mut pass, i as u8);
            }
        }
    }
}

#[derive(Debug)]
/// A single allocated region on a particular texture atlas. We store the pixel coordinates, not the normalized UV coordinates.
pub struct Region {
    pub id: AllocId,
    pub uv: guillotiere::Rectangle,
    pub index: u8,
}

// Technically, this should implement !Forget (or !Leak), because it should never be put in an Rc<>. If rust ever stabilizes
// that trait, it would be good to add it here. However, using a stale Region will simply render incorrect data, it won't
// access invalid memory, so there is no safety or soundness problem here, only a correctness issue.
impl Drop for Region {
    fn drop(&mut self) {
        if self.id != AllocId::deserialize(u32::MAX) {
            panic!(
                "dropped a region without deallocating it! Regions can't automatically deallocate themselves, put them inside an object that can store a reference to the Atlas!"
            )
        }
    }
}
