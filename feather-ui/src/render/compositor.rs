use crate::AbsRect;
use std::{collections::HashMap, num::NonZero, rc::Rc};
use wgpu::{TextureUsages, TextureViewDescriptor, wgt::SamplerDescriptor};

use crate::ZERO_RECT;
use parking_lot::RwLock;

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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compositor.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compositor Bind Group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZero::new(size_of::<ultraviolet::Mat4>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
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
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

/// Fundamentally, the compositor works on a massive set of pre-allocated quads. The vertex shader moves these quads into
/// position and assigns them UV coordinates. The pixel shader first checks if it must do per-pixel clipping and discards
/// the pixel if it's out of bounds, then samples a texture from the provided texture bank, does color modulation if
/// applicable, then draws the final pixel to the screen using pre-multiplied alpha blending.
///
/// This can be achieved in many different ways. Starting from the simplest method:
/// 1. Pre-allocate each triangle individually (requires 6 vertices per quad) and render them all in one draw call,
///    encoding the buffer index into the vertex. Then, pass this index to the pixel shader, along with the UV values,
///    which can then use the index to look up the clip index, texture index, and color modulation to use for that pixel.
/// 2. Use 4 vertices and a triangle strip for the quad, but utilize GPU instancing. The GPU then provides the index of
///    which buffer value needs to be used for each instance of the quad to both the vertex shader and pixel shader. This
///    means the vertex shader only has to move the unit square into position and generate UV coordinates.
/// 3. Use 4 vertices and a triangle strip for the quad, but use command queues to assemble the draw instructions ahead of
///    time, in such a way that the command queue can be re-used because nothing about the command queue actually changes,
///    only the buffers and textures bound to it change. This requires DX12 or vulkan.
/// 4. Use 4 vertices and a triangle strip for the quad, but use GPU indirect calls. This means assembling two buffers,
///    one that contains the draw calls to be issued that won't need to change, and another buffer containing the changing
///    information about what to composite. This still requires DX12 or vulkan, but would allow feather to implement more
///    complex handling of GPU state to minimize memory transfers.
///
/// We currently use (1), as it works on virtually any hardware. We would prefer (2), but WebGL 1.0 didn't support
/// instancing, nor did OpenGL ES 2.0, which means many older devices also don't. (3) or (4) may be implemented as a
/// seperate option later, espiecally if we continue to use wgpu, since wgpu assumes vulkan support.
pub struct Compositor {
    shared: Rc<Shared>,
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    mvp: wgpu::Buffer,
    buffer: wgpu::Buffer,
    clip: wgpu::Buffer,
    clipdata: Vec<AbsRect>,             // Clipping Rectangles
    data: Vec<Data>,                    // CPU-side buffer of all the data
    regions: Vec<std::ops::Range<u32>>, // Target copy ranges for where to map data to the GPU buffer. Assumes contiguous data vector.
    defer: HashMap<u32, Box<dyn FnOnce(&crate::graphics::Driver, &mut Data)>>,
}

impl Compositor {
    pub fn new(
        shared: Rc<Shared>,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        atlas: &super::atlas::Atlas,
    ) -> Self {
        let pipeline = shared.get_pipeline(device, config.view_formats[0]);

        let mvp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MVP"),
            size: std::mem::size_of::<ultraviolet::Mat4>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Data"),
            size: 32 * size_of::<Data>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let clip = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ClipRects"),
            size: 4 * size_of::<AbsRect>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let atlasview = atlas.get_texture().create_view(&TextureViewDescriptor {
            label: Some("Compositor Atlas View"),
            format: Some(crate::render::atlas::ATLAS_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_array_layer: 0,
            array_layer_count: None,
            ..Default::default()
        });

        let bindings = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: mvp.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&atlasview),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&shared.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: clip.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: atlas.extent_buf.as_entire_binding(),
            },
        ];

        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bindings,
            label: None,
        });

        Self {
            shared,
            pipeline,
            group,
            mvp,
            buffer,
            clip,
            clipdata: vec![ZERO_RECT],
            regions: vec![0..0],
            data: Vec::new(),
            defer: HashMap::new(),
        }
    }

    fn check_data(&mut self, device: &wgpu::Device) {
        let size = self.data.len() * size_of::<Data>();
        if (self.buffer.size() as usize) < size {
            self.buffer.destroy();
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Composite Data"),
                size: size.next_power_of_two() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
    }

    fn check_clip(&mut self, device: &wgpu::Device) {
        let size = self.clipdata.len() * size_of::<AbsRect>();
        if (self.clip.size() as usize) < size {
            self.clip.destroy();
            self.clip = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Clip Data"),
                size: size.next_power_of_two() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
    }

    pub(crate) fn append_clip(&mut self, clip: AbsRect) -> u16 {
        let n = self.clipdata.len();
        self.clipdata.push(clip);
        n as u16
    }

    /// Returns the GPU buffer and the current offset, which allows a compute shader to accumulate commands
    /// in the GPU buffer directly, provided it calls set_compute_buffer afterwards with the command count.
    /// Attempting to insert a non-GPU command before calling set_compute_buffer will panic.
    pub(crate) fn get_compute_buffer(&mut self) -> (&wgpu::Buffer, u32) {
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
    pub(crate) fn set_compute_buffer(&mut self, count: u32) {
        let region = self.regions.last_mut().unwrap();
        region.start += count;
        region.end = region.start;
    }

    pub fn prepare(
        &mut self,
        graphics: &crate::graphics::Driver,
        encoder: &mut wgpu::CommandEncoder,
        _: &wgpu::SurfaceConfiguration,
    ) {
        // If we have a pending copy, queue it up.
        graphics
            .atlas
            .borrow_mut()
            .perform_copy(&graphics.device, &graphics.queue, encoder);

        // Resolve all defers
        for (idx, f) in self.defer.drain() {
            f(graphics, &mut self.data[idx as usize]);
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
        graphics: &crate::graphics::Driver,
        pass: &mut wgpu::RenderPass<'_>,
        config: &wgpu::SurfaceConfiguration,
    ) {
        graphics.queue.write_buffer(
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

        if self.clipdata.len() > 0 {
            self.check_clip(&graphics.device);
            graphics.queue.write_buffer(
                &self.clip,
                0,
                bytemuck::cast_slice(self.clipdata.as_slice()),
            );
        }

        self.check_data(&graphics.device);

        // TODO turn into write_buffer_with (is that actually going to be faster?)
        let mut offset = 0;
        for range in &self.regions {
            let len = range.end - range.start;
            graphics.queue.write_buffer(
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

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, bytemuck::NoUninit)]
pub struct Data {
    pub pos: [f32; 2],
    pub dim: [f32; 2],
    pub uv: [i32; 2],
    pub uvdim: [i32; 2],
    pub color: u32,
    pub rotation: f32,
    pub tex: u16,
    pub clip: u16,
}

// Our shader will assemble a rotation based on this matrix, but transposed:
// [ cos(r) -sin(r) 0 (x - x*cos(r) + y*sin(r)) ]
// [ sin(r)  cos(r) 0 (y - x*sin(r) - y*cos(r)) ]
// [ 0       0      1 0 ]
// [ 0       0      0 1 ] ^ -1
