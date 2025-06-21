use crate::{AbsRect, graphics::Vec2f};
use std::{collections::HashMap, num::NonZero, rc::Rc};
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, Buffer, BufferDescriptor, BufferUsages, TextureView,
    wgt::SamplerDescriptor,
};

use crate::ZERO_RECT;
use parking_lot::RwLock;

use super::atlas::Atlas;

#[derive(Debug)]
/// Shared resources that are the same between all windows
pub struct Shared {
    layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
    pipelines: RwLock<HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>>,
}

pub const TARGET_STATE: wgpu::ColorTargetState = wgpu::ColorTargetState {
    format: crate::render::atlas::ATLAS_FORMAT,
    blend: Some(wgpu::BlendState::REPLACE),
    write_mask: wgpu::ColorWrites::ALL,
};

impl Shared {
    pub fn new(device: &wgpu::Device, _: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compositor"),
            //    source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compositor.wgsl").into()),
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string(
                    std::env::current_exe()
                        .unwrap()
                        .join("../../../feather-ui/src/shaders/compositor.wgsl"),
                )
                .unwrap()
                .into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compositor Bind Group"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZero::new(size_of::<ultraviolet::Mat4>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZero::new(size_of::<u32>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compositor Pipeline"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Compositor Sampler"),
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
            layout,
            shader,
            sampler,
            pipelines: HashMap::new().into(),
        }
    }

    fn get_pipeline(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        self.pipelines
            .write()
            .entry(format)
            .or_insert_with(|| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&self.layout),
                    vertex: wgpu::VertexState {
                        module: &self.shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &self.shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
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
                })
            })
            .clone()
    }
}

type DeferFn = dyn FnOnce(&crate::graphics::Driver, &mut Data);

/// Fundamentally, the compositor works on a massive set of pre-allocated vertices that it assembles into quads in the vertex
/// shader, which then moves them into position and assigns them UV coordinates. Then the pixel shader checks if it must do
/// per-pixel clipping and discards the pixel if it's out of bounds, then samples a texture from the provided texture bank,
/// does color modulation if applicable, then draws the final pixel to the screen using pre-multiplied alpha blending. This
/// allows the compositor to avoid allocating a vertex buffer, instead using a SSBO (or webgpu storage buffer) to store the
/// per-quad data, which it then accesses from the built-in vertex index.
///
/// The compositor can accept GPU generated instructions written directly into it's buffer using a compute shader, if desired.
///
/// The compositor can also accept custom draw calls that break up the batched compositor instructions, which is intended for
/// situations where rendering to the texture atlas is either impractical, or a different blending operation is required (such
/// as subpixel blended text, which requires the SRC1 dual-source blending mode, instead of standard pre-multiplied alpha).
pub struct Compositor {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    mvp: Buffer,
    buffer: Buffer,
    clip: Buffer,
    clipdata: Vec<AbsRect>,             // Clipping Rectangles
    data: Vec<Data>,                    // CPU-side buffer of all the data
    regions: Vec<std::ops::Range<u32>>, // Target copy ranges for where to map data to the GPU buffer. Assumes contiguous data vector.
    defer: HashMap<u32, Box<DeferFn>>,
    view: std::rc::Weak<TextureView>,
}

impl Compositor {
    fn gen_binding(
        mvp: &Buffer,
        buffer: &Buffer,
        clip: &Buffer,
        shared: &Shared,
        device: &wgpu::Device,
        atlas: &Atlas,
        atlasview: &TextureView,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let bindings = [
            BindGroupEntry {
                binding: 0,
                resource: mvp.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(atlasview),
            },
            BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&shared.sampler),
            },
            BindGroupEntry {
                binding: 4,
                resource: clip.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: atlas.extent_buf.as_entire_binding(),
            },
        ];

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &bindings,
            label: None,
        })
    }

    pub fn new(
        shared: &Shared,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        atlas: &Atlas,
    ) -> Self {
        let pipeline = shared.get_pipeline(device, config.view_formats[0]);

        let mvp = device.create_buffer(&BufferDescriptor {
            label: Some("MVP"),
            size: std::mem::size_of::<ultraviolet::Mat4>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Compositor Data"),
            size: 32 * size_of::<Data>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let clip = device.create_buffer(&BufferDescriptor {
            label: Some("Compositor Clip Data"),
            size: 4 * size_of::<AbsRect>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let group = Self::gen_binding(
            &mvp,
            &buffer,
            &clip,
            shared,
            device,
            atlas,
            &atlas.view,
            &pipeline.get_bind_group_layout(0),
        );

        #[allow(clippy::single_range_in_vec_init)]
        Self {
            pipeline,
            group,
            mvp,
            buffer,
            clip,
            clipdata: vec![ZERO_RECT],
            regions: vec![0..0],
            data: Vec::new(),
            defer: HashMap::new(),
            view: Rc::downgrade(&atlas.view),
        }
    }

    /// Should be called when self.view.strong_count() is 0, or when the data bindings must be rebound
    pub fn rebind(&mut self, shared: &Shared, device: &wgpu::Device, atlas: &Atlas) {
        self.view = Rc::downgrade(&atlas.view);
        self.group = Self::gen_binding(
            &self.mvp,
            &self.buffer,
            &self.clip,
            shared,
            device,
            atlas,
            &atlas.view,
            &self.pipeline.get_bind_group_layout(0),
        );
    }

    fn check_data(&mut self, shared: &Shared, device: &wgpu::Device, atlas: &Atlas) {
        let size = self.data.len() * size_of::<Data>();
        if (self.buffer.size() as usize) < size {
            self.buffer.destroy();
            self.buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Compositor Data"),
                size: size.next_power_of_two() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.rebind(shared, device, atlas);
        }
    }

    fn check_clip(&mut self, shared: &Shared, device: &wgpu::Device, atlas: &Atlas) {
        let size = self.clipdata.len() * size_of::<AbsRect>();
        if (self.clip.size() as usize) < size {
            self.clip.destroy();
            self.clip = device.create_buffer(&BufferDescriptor {
                label: Some("Compositor Clip Data"),
                size: size.next_power_of_two() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.rebind(shared, device, atlas);
        }
    }

    pub fn append_clip(&mut self, clip: AbsRect) -> u16 {
        let n = self.clipdata.len();
        self.clipdata.push(clip);
        n as u16
    }

    /// Returns the GPU buffer and the current offset, which allows a compute shader to accumulate commands
    /// in the GPU buffer directly, provided it calls set_compute_buffer afterwards with the command count.
    /// Attempting to insert a non-GPU command before calling set_compute_buffer will panic.
    pub fn get_compute_buffer(&mut self) -> (&Buffer, u32) {
        let offset = self.regions.last().unwrap().end;
        if offset == u32::MAX {
            panic!(
                "Still processing a compute operation! Finish it by calling set_compute_buffer() first."
            );
        }
        self.regions.push(offset..u32::MAX);
        (&self.buffer, offset)
    }

    /// After executing a compute shader that added a series of compositor commands to the command buffer,
    /// this must be called with the number of commands that were contiguously inserted into the buffer.
    pub fn set_compute_buffer(&mut self, count: u32) {
        let region = self.regions.last_mut().unwrap();
        region.start += count;
        region.end = region.start;
    }

    pub fn prepare(
        &mut self,
        driver: &crate::graphics::Driver,
        _: &mut wgpu::CommandEncoder,
        _: &wgpu::SurfaceConfiguration,
    ) {
        // Check to see if we need to rebind our atlas view
        if self.view.strong_count() == 0 {
            self.rebind(&driver.shared, &driver.device, &driver.atlas.read());
        }

        // Resolve all defers
        for (idx, f) in self.defer.drain() {
            f(driver, &mut self.data[idx as usize]);
        }
    }

    pub fn append(&mut self, data: &Data) -> u32 {
        let region = self.regions.last_mut().unwrap();
        if region.end == u32::MAX {
            panic!(
                "Still processing a compute operation! Finish it by calling set_compute_buffer() first."
            );
        }

        let idx = region.end;
        self.data.push(*data);
        region.end += 1;
        idx
    }

    pub fn defer(&mut self, f: impl FnOnce(&crate::graphics::Driver, &mut Data) + 'static) {
        let region = self.regions.last_mut().unwrap();
        if region.end == u32::MAX {
            panic!(
                "Still processing a compute operation! Finish it by calling set_compute_buffer() first."
            );
        }

        let idx = region.end;
        self.data.push(Default::default());
        self.defer.insert(idx, Box::new(f));
        region.end += 1;
    }

    pub fn draw(
        &mut self,
        driver: &crate::graphics::Driver,
        pass: &mut wgpu::RenderPass<'_>,
        config: &wgpu::SurfaceConfiguration,
    ) {
        driver.queue.write_buffer(
            &self.mvp,
            0,
            crate::graphics::mat4_proj(
                0.0,
                config.height as f32,
                config.width as f32,
                -(config.height as f32),
                0.2,
                10000.0,
            )
            .as_byte_slice(),
        );

        if !self.clipdata.is_empty() {
            self.check_clip(&driver.shared, &driver.device, &driver.atlas.read());
            driver.queue.write_buffer(
                &self.clip,
                0,
                bytemuck::cast_slice(self.clipdata.as_slice()),
            );
        }

        self.check_data(&driver.shared, &driver.device, &driver.atlas.read());

        // TODO turn into write_buffer_with (is that actually going to be faster?)
        let mut offset = 0;
        for range in &self.regions {
            let len = range.end - range.start;
            driver.queue.write_buffer(
                &self.buffer,
                range.start as u64,
                bytemuck::cast_slice(
                    &self.data.as_slice()[offset as usize..(offset + len) as usize],
                ),
            );
            offset += len;
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[0]);
        pass.draw(0..(self.regions.last().unwrap().end * 6), 0..1);

        self.data.clear();
        self.clipdata.clear();
        self.clipdata.push(ZERO_RECT);
        self.regions.clear();
        self.regions.push(0..0);
    }
}

// Renderdoc Format:
// struct Data {
// 	float pos[2];
// 	float dim[2];
//  float uv[2];
//  float uvdim[2];
// 	uint32_t color;
// 	float rotation;
// 	uint32_t texclip;
//  char padding[4];
// };
// Data d[];

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, bytemuck::NoUninit)]
pub struct Data {
    pub pos: Vec2f,
    pub dim: Vec2f,
    pub uv: Vec2f,
    pub uvdim: Vec2f,
    pub color: u32, // Encoded as a non-linear, non-premultiplied sRGB32 color
    pub rotation: f32,
    pub texclip: u32,
    pub _padding: [u8; 4], // We have to manually specify this to satisfy bytemuck
}

static_assertions::const_assert_eq!(std::mem::size_of::<Data>(), 48);

impl Data {
    pub fn new(
        pos: ultraviolet::Vec2,
        dim: ultraviolet::Vec2,
        uv: ultraviolet::Vec2,
        uvdim: ultraviolet::Vec2,
        color: u32,
        rotation: f32,
        tex: u16,
        clip: u16,
    ) -> Self {
        Self {
            pos: pos.as_array().into(),
            dim: dim.as_array().into(),
            uv: uv.as_array().into(),
            uvdim: uvdim.as_array().into(),
            color,
            rotation,
            texclip: (((tex & 0x7FFF) as u32) << 16) | clip as u32,
            ..Default::default()
        }
    }
}

// Our shader will assemble a rotation based on this matrix, but transposed:
// [ cos(r) -sin(r) 0 (x - x*cos(r) + y*sin(r)) ]
// [ sin(r)  cos(r) 0 (y - x*sin(r) - y*cos(r)) ]
// [ 0       0      1 0 ]
// [ 0       0      0 1 ] ^ -1
