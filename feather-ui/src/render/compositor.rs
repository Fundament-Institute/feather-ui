// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::color::sRGB32;
use crate::graphics::{Driver, Vec2f};
use crate::{AbsDim, AbsRect, SourceID};
use derive_where::derive_where;
use num_traits::Zero;
use std::collections::HashMap;
use std::num::NonZero;
use std::sync::Arc;
use ultraviolet::Vec2;
use wgpu::util::DeviceExt;
use wgpu::wgt::SamplerDescriptor;
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, Buffer, BufferDescriptor, BufferUsages, TextureView,
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
    layers: RwLock<HashMap<Arc<SourceID>, Layer>>,
}

pub const TARGET_STATE: wgpu::ColorTargetState = wgpu::ColorTargetState {
    format: crate::render::atlas::ATLAS_FORMAT,
    blend: Some(wgpu::BlendState::REPLACE),
    write_mask: wgpu::ColorWrites::ALL,
};

impl Shared {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compositor"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compositor.wgsl").into()),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX,
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
            layers: HashMap::new().into(),
        }
    }

    pub fn access_layers(
        &self,
    ) -> parking_lot::lock_api::RwLockReadGuard<
        '_,
        parking_lot::RawRwLock,
        HashMap<Arc<SourceID>, Layer>,
    > {
        self.layers.read()
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

    pub fn create_layer(
        &self,
        _device: &wgpu::Device,
        id: Arc<SourceID>,
        area: AbsRect,
        dest: Option<AbsRect>,
        color: sRGB32,
        rotation: f32,
        force: bool,
    ) -> Option<Layer> {
        let dest = dest.unwrap_or(area);

        // If true, this is a clipping layer, not a texture-backed one
        let target = if color == sRGB32::white() && rotation.is_zero() && !force && dest == area {
            None
        } else {
            Some(RwLock::new(LayerTarget {
                dependents: Default::default(),
            }))
        };

        let layer = Layer {
            area,
            dest,
            color,
            rotation,
            target,
        };

        if let Some(prev) = self.layers.read().get(&id) {
            if *prev == layer {
                return None;
            }
        }

        self.layers.write().insert(id, layer)
    }
}

// This holds the information for rendering to a layer, which can only be done if the layer is texture-backed.
#[derive(Debug)]
pub struct LayerTarget {
    pub dependents: Vec<std::sync::Weak<SourceID>>, // Layers that draw on to this one (does not include fake layers)
}

#[derive(Debug)]
pub struct Layer {
    // Renderable area representing what children draw onto. This corresponds to this layer's compositor viewport, if it has one.
    pub area: AbsRect,
    // destination area that the layer is composited onto. Usually this is the same as area, but can be different when scaling down
    dest: AbsRect,
    color: sRGB32,
    rotation: f32,
    // Layers aren't always texture-backed so this may not exist
    pub target: Option<RwLock<LayerTarget>>,
}

impl PartialEq for Layer {
    fn eq(&self, other: &Self) -> bool {
        self.area == other.area
            && self.dest == other.dest
            && self.color == other.color
            && self.rotation == other.rotation
    }
}

type DeferFn = dyn FnOnce(&Driver, &mut Data) + Send + Sync;
type CustomDrawFn = dyn FnMut(&Driver, &mut wgpu::RenderPass<'_>, Vec2) + Send + Sync;

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
#[derive_where(Debug)]
pub struct Compositor {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    mvp: Buffer,
    buffer: Buffer,
    clip: Buffer,
    passindex: Buffer,
    clipdata: Vec<AbsRect>,             // Clipping Rectangles
    data: Vec<Data>,                    // CPU-side buffer of all the data
    regions: Vec<std::ops::Range<u32>>, // Target copy ranges for where to map data to the GPU buffer. Assumes contiguous data vector.
    #[derive_where(skip)]
    defer: HashMap<u32, (Box<DeferFn>, AbsRect, Vec2, u8)>,
    view: std::sync::Weak<TextureView>,
    layer_view: std::sync::Weak<TextureView>,
    #[derive_where(skip)]
    custom: Vec<(u32, Box<CustomDrawFn>, Vec2, u8)>,
    layer: bool, // Tells us which layer atlas to use (the first or second)
    max_depth: u8,
}

#[derive_where(Debug)]
pub struct Segment {
    buffer: Buffer,
    data: Vec<Data>,
    regions: Vec<std::ops::Range<u32>>,
    #[derive_where(skip)]
    defer: HashMap<u32, (Box<DeferFn>, AbsRect, Vec2)>,
    #[derive_where(skip)]
    custom: Vec<(u32, Box<CustomDrawFn>, Vec2, u8)>,
}

impl Compositor {
    pub fn max_depth(&self) -> u8 {
        self.max_depth
    }

    fn gen_binding(
        mvp: &Buffer,
        buffer: &Buffer,
        clip: &Buffer,
        passindex: &Buffer,
        shared: &Shared,
        device: &wgpu::Device,
        atlasview: &TextureView,
        layerview: &TextureView,
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
                resource: clip.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&shared.sampler),
            },
            BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(atlasview),
            },
            BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(layerview),
            },
            BindGroupEntry {
                binding: 6,
                resource: passindex.as_entire_binding(),
            },
        ];

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &bindings,
            label: None,
        })
    }

    // This cannot take a Driver because we have to create two Compositors before the Driver object is made
    pub fn new(
        device: &wgpu::Device,
        shared: &Shared,
        atlasview: &Arc<TextureView>,
        layerview: &Arc<TextureView>,
        format: wgpu::TextureFormat,
        layer: bool,
    ) -> Self {
        let pipeline = shared.get_pipeline(device, format);

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

        let pass = 0_u32;
        let passindex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pass Index"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(&[pass]),
        });

        let group = Self::gen_binding(
            &mvp,
            &buffer,
            &clip,
            &passindex,
            shared,
            device,
            atlasview,
            layerview,
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
            view: Arc::downgrade(atlasview),
            layer_view: Arc::downgrade(layerview),
            custom: Vec::new(),
            layer,
            passindex,
            max_depth: 0,
        }
    }

    /// Should be called when any external or internal buffer gets invalidated (such as atlas views)
    pub fn rebind(
        &mut self,
        shared: &Shared,
        device: &wgpu::Device,
        atlas: &Atlas,
        layers: &Atlas,
    ) {
        self.view = Arc::downgrade(&atlas.view);
        self.layer_view = Arc::downgrade(&layers.view);
        self.group = Self::gen_binding(
            &self.mvp,
            &self.buffer,
            &self.clip,
            &self.passindex,
            shared,
            device,
            &atlas.view,
            &layers.view,
            &self.pipeline.get_bind_group_layout(0),
        );
    }

    fn check_data(
        &mut self,
        shared: &Shared,
        device: &wgpu::Device,
        atlas: &Atlas,
        layers: &Atlas,
    ) {
        let size = self.data.len() * size_of::<Data>();
        if (self.buffer.size() as usize) < size {
            self.buffer.destroy();
            self.buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Compositor Data"),
                size: size.next_power_of_two() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.rebind(shared, device, atlas, layers);
        }
    }

    fn check_clip(
        &mut self,
        shared: &Shared,
        device: &wgpu::Device,
        atlas: &Atlas,
        layers: &Atlas,
    ) {
        let size = self.clipdata.len() * size_of::<AbsRect>();
        if (self.clip.size() as usize) < size {
            self.clip.destroy();
            self.clip = device.create_buffer(&BufferDescriptor {
                label: Some("Compositor Clip Data"),
                size: size.next_power_of_two() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.rebind(shared, device, atlas, layers);
        }
    }

    fn append_clipdata(&mut self, clip: AbsRect) -> u16 {
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

    #[inline]
    fn scissor_check(dim: &Vec2, clip: AbsRect) -> Result<AbsRect, AbsRect> {
        if AbsRect::new(0.0, 0.0, dim.x, dim.y) == clip {
            Err(clip)
        } else {
            Ok(clip)
        }
    }

    #[inline]
    fn clip_data(
        clipdata: &mut Vec<AbsRect>,
        cliprect: Result<AbsRect, AbsRect>,
        offset: Vec2,
        mut data: Data,
    ) -> Option<Data> {
        match cliprect {
            Ok(clip) => {
                if data.rotation.is_zero() {
                    let (mut x, mut y, mut w, mut h) = (
                        data.pos.0[0] + offset.x,
                        data.pos.0[1] + offset.y,
                        data.dim.0[0],
                        data.dim.0[1],
                    );

                    // If the whole rect is outside the cliprect, don't render it at all.
                    if !clip.collides(&AbsRect::new(x, y, x + w, y + h)) {
                        // TODO: When we start reserving slots, this will need to instead insert a special zero size rect.
                        return None;
                    }

                    let (mut u, mut v, mut uw, mut vh) =
                        (data.uv.0[0], data.uv.0[1], data.uvdim.0[0], data.uvdim.0[1]);

                    // If rotation is zero, we don't need to do per-pixel clipping, we can just modify the rect itself.
                    let bounds = clip.0.as_array_ref();
                    let (min_x, min_y, max_x, max_y) = (bounds[0], bounds[1], bounds[2], bounds[3]);

                    // Get the ratio from our target rect to the source UV sampler rect
                    let uv_ratio = Vec2::new(uw / w, vh / h);

                    // Clip left edge
                    if x < min_x {
                        let right_shift = min_x - x;

                        x += right_shift;
                        u += right_shift * uv_ratio.x;
                        w -= right_shift;
                        uw -= right_shift * uv_ratio.x;
                    }

                    // Clip right edge
                    if x + w > max_x {
                        let right_shift = max_x - (x + w);
                        w -= right_shift;
                        uw -= right_shift * uv_ratio.x;
                    }

                    // Clip top edge
                    if y < min_y {
                        let bottom_shift = min_y - y;

                        y += bottom_shift;
                        v += bottom_shift * uv_ratio.y;
                        h -= bottom_shift;
                        vh -= bottom_shift * uv_ratio.y;
                    }

                    // Clip bottom edge
                    if y + h > max_y {
                        let bottom_shift = max_y - (y + h);
                        h -= bottom_shift;
                        vh -= bottom_shift * uv_ratio.y;
                    }

                    Some(Data {
                        pos: [x, y].into(),
                        dim: [w, h].into(),
                        uv: [u, v].into(),
                        uvdim: [uw, vh].into(),
                        color: data.color,
                        rotation: data.rotation,
                        flags: data.flags,
                        _padding: data._padding,
                    })
                } else {
                    // TODO: Beyond some size, like 32, skip all elements except the last N clipping rects and only check those
                    let idx = if let Some((idx, _)) =
                        clipdata.iter().enumerate().find(|(_, r)| **r == clip)
                    {
                        idx
                    } else {
                        clipdata.push(clip);
                        clipdata.len() - 1
                    };

                    debug_assert!(idx < 0xFFFF);
                    data.flags = DataFlags::from_bits(data.flags)
                        .with_clip(idx as u16)
                        .into();
                    data.pos.0[0] += offset.x;
                    data.pos.0[1] += offset.y;
                    Some(data)
                }
            }
            Err(clip) => {
                // If the current cliprect is just the scissor rect, we do NOT add a custom clipping rect or do further clipping, but we do
                // check to see if we need to bother rendering this at all.
                if data.rotation.is_zero() {
                    let (x, y, w, h) = (data.pos.0[0], data.pos.0[1], data.dim.0[0], data.dim.0[1]);
                    if !clip.collides(&AbsRect::new(x, y, x + w, y + h)) {
                        // TODO: When we start reserving slots, this will need to instead insert a special zero size rect.
                        return None;
                    }
                }
                data.pos.0[0] += offset.x;
                data.pos.0[1] += offset.y;
                Some(data)
            }
        }
    }

    pub fn prepare(
        &mut self,
        driver: &Driver,
        _: &mut wgpu::CommandEncoder,
        config: &wgpu::SurfaceConfiguration,
    ) {
        // Check to see if we need to rebind either atlas view
        if self.view.strong_count() == 0 || self.layer_view.strong_count() == 0 {
            self.rebind(
                &driver.shared,
                &driver.device,
                &driver.atlas.read(),
                &driver.layer_atlas[self.layer as usize].read(),
            );
        }

        let dim = Vec2::new(config.width as f32, config.height as f32);

        // Resolve all defers
        for (idx, (f, clip, offset, pass)) in self.defer.drain() {
            f(driver, &mut self.data[idx as usize]);
            self.data[idx as usize].flags = DataFlags::from_bits(self.data[idx as usize].flags)
                .with_pass(pass)
                .into();
            self.data[idx as usize] = Self::clip_data(
                &mut self.clipdata,
                Self::scissor_check(&dim, clip),
                offset,
                self.data[idx as usize],
            )
            .unwrap_or_default();
        }
    }

    #[inline]
    fn append_internal(
        &mut self,
        clipstack: &[AbsRect],
        surface_dim: Vec2,
        layer_offset: Vec2,
        pos: ultraviolet::Vec2,
        dim: ultraviolet::Vec2,
        uv: ultraviolet::Vec2,
        uvdim: ultraviolet::Vec2,
        color: u32,
        rotation: f32,
        tex: u8,
        pass: u8,
        raw: bool,
        layer: bool,
    ) -> u32 {
        let data = Data {
            pos: pos.as_array().into(),
            dim: dim.as_array().into(),
            uv: uv.as_array().into(),
            uvdim: uvdim.as_array().into(),
            color,
            rotation,
            flags: DataFlags::new()
                .with_tex(tex)
                .with_raw(raw)
                .with_layer(layer)
                .with_pass(pass)
                .into(),
            ..Default::default()
        };

        if let Some(d) = Self::clip_data(
            &mut self.clipdata,
            clipstack.last().ok_or(AbsDim(surface_dim).into()).copied(),
            layer_offset,
            data,
        ) {
            self.preprocessed(d)
        } else {
            u32::MAX // TODO: Once we start reserving slots, we will always need to return a valid one by inserting an empty rect in clip_data
        }
    }

    #[inline]
    fn preprocessed(&mut self, data: Data) -> u32 {
        let region = self.regions.last_mut().unwrap();
        if region.end == u32::MAX {
            panic!(
                "Still processing a compute operation! Finish it by calling set_compute_buffer() first."
            );
        }

        let idx = region.end;
        self.data.push(data);
        region.end += 1;
        idx
    }

    pub fn predraw(&mut self, target_dim: Vec2, driver: &Driver) {
        driver.queue.write_buffer(
            &self.mvp,
            0,
            crate::graphics::mat4_proj(
                0.0,
                target_dim.y,
                target_dim.x,
                -(target_dim.y),
                0.2,
                10000.0,
            )
            .as_byte_slice(),
        );

        if !self.clipdata.is_empty() {
            self.check_clip(
                &driver.shared,
                &driver.device,
                &driver.atlas.read(),
                &driver.layer_atlas[self.layer as usize].read(),
            );
            driver.queue.write_buffer(
                &self.clip,
                0,
                bytemuck::cast_slice(self.clipdata.as_slice()),
            );
        }

        self.check_data(
            &driver.shared,
            &driver.device,
            &driver.atlas.read(),
            &driver.layer_atlas[self.layer as usize].read(),
        );

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
    }

    pub fn draw(&mut self, driver: &Driver, pass: &mut wgpu::RenderPass<'_>, index: u32) {
        driver
            .queue
            .write_buffer(&self.passindex, 0, bytemuck::cast_slice(&[index]));

        let mut last_index = 0;
        for (i, f, draw_offset, pass_index) in &mut self.custom {
            if *pass_index as u32 == index {
                if last_index < *i {
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &self.group, &[0]);
                    pass.draw(last_index..*i, 0..1);
                }
                last_index = *i;
                f(driver, pass, *draw_offset);
            }
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[0]);
        pass.draw(last_index..(self.regions.last().unwrap().end * 6), 0..1);
    }

    pub fn postdraw(&mut self) {
        self.data.clear();
        self.clipdata.clear();
        self.clipdata.push(ZERO_RECT);
        self.regions.clear();
        self.regions.push(0..0);
    }
}

/// A Compositor is associated with a render target, which is usually a window, but can also be an intermediate buffer, used
/// for Layers. As a result, a compositor does not control the clipping rect stack, the window itself does. This associates
/// a compositor with a clipstack and any other information it might need to properly append data during a render, such as
/// the offset. Because of this auxillairy information, you cannot append directly to a compositor, only to a CompositorView.
pub struct CompositorView<'a> {
    pub index: u8, // While we carry mutable references of all 3 possible compositors, this tells us which we're currently using
    pub window: &'a mut Compositor, // index 0
    pub layer0: &'a mut Compositor, // index 1
    pub layer1: &'a mut Compositor, // index 2
    pub clipstack: &'a mut Vec<AbsRect>,
    pub offset: Vec2,
    pub surface_dim: Vec2, // Dimension of the top-level window surface.
    pub pass: u8,
}

impl<'a> CompositorView<'a> {
    #[inline]
    pub fn push_clip(&mut self, clip: AbsRect) {
        if let Some(prev) = self.clipstack.last() {
            self.clipstack.push(clip.intersect(*prev));
        } else {
            self.clipstack.push(clip);
        }
    }

    #[inline]
    pub fn current_clip(&self) -> AbsRect {
        *self
            .clipstack
            .last()
            .unwrap_or(&AbsDim(self.surface_dim).into())
    }

    #[inline]
    pub fn pop_clip(&mut self) -> AbsRect {
        self.clipstack
            .pop()
            .expect("Tried to pop a clipping rect but the stack was empty!")
    }

    #[inline]
    pub fn compositor(&mut self) -> &mut Compositor {
        match self.index {
            0 => self.window,
            1 => self.layer0,
            2 => self.layer1,
            _ => panic!("Illegal compositor index!"),
        }
    }

    #[inline]
    pub fn append_data(
        &mut self,
        pos: ultraviolet::Vec2,
        dim: ultraviolet::Vec2,
        uv: ultraviolet::Vec2,
        uvdim: ultraviolet::Vec2,
        color: u32,
        rotation: f32,
        tex: u8,
        raw: bool,
    ) -> u32 {
        // I really wish rust had partial borrows
        let compositor = match self.index {
            0 => &mut self.window,
            1 => &mut self.layer0,
            2 => &mut self.layer1,
            _ => panic!("Illegal compositor index!"),
        };
        compositor.append_internal(
            self.clipstack,
            self.surface_dim,
            self.offset,
            pos,
            dim,
            uv,
            uvdim,
            color,
            rotation,
            tex,
            self.pass,
            raw,
            false,
        )
    }

    #[inline]
    pub(crate) fn append_layer(&mut self, layer: &Layer, uv: guillotiere::Rectangle) -> u32 {
        // I really wish rust had partial borrows
        let compositor = match self.index {
            0 => &mut self.window,
            1 => &mut self.layer0,
            2 => &mut self.layer1,
            _ => panic!("Illegal compositor index!"),
        };
        compositor.append_internal(
            self.clipstack,
            self.surface_dim,
            self.offset,
            layer.dest.topleft(),
            layer.dest.dim().0,
            uv.min.to_f32().to_array().into(),
            uv.size().to_f32().to_array().into(),
            layer.color.rgba,
            layer.rotation,
            0,
            self.pass,
            false,
            true,
        )
    }

    #[inline]
    pub fn preprocessed(&mut self, mut data: Data) -> u32 {
        data.pos.0[0] += self.offset.x;
        data.pos.0[1] += self.offset.y;
        data.flags = DataFlags::from_bits(data.flags).with_pass(self.pass).into();
        self.compositor().preprocessed(data)
    }

    #[inline]
    pub fn defer(&mut self, f: impl FnOnce(&Driver, &mut Data) + Send + Sync + 'static) {
        let clip = self.current_clip();
        let offset = self.offset;
        let pass = self.pass;
        let compositor = self.compositor();
        let region = compositor.regions.last_mut().unwrap();
        if region.end == u32::MAX {
            panic!(
                "Still processing a compute operation! Finish it by calling set_compute_buffer() first."
            );
        }

        let idx = region.end;
        compositor.data.push(Default::default());
        compositor
            .defer
            .insert(idx, (Box::new(f), clip, offset, pass));
        region.end += 1;
    }

    pub fn append_custom(
        &mut self,
        f: impl FnMut(&Driver, &mut wgpu::RenderPass<'_>, Vec2) + Send + Sync + 'static,
    ) {
        let index = self.compositor().regions.last().unwrap().end;
        if index == u32::MAX {
            panic!(
                "Still processing a compute operation! Finish it by calling set_compute_buffer() first."
            );
        }
        let offset = self.offset;
        let pass = self.pass;
        self.compositor()
            .custom
            .push((index, Box::new(f), offset, pass));
    }
}

#[bitfield_struct::bitfield(u32)]
pub struct DataFlags {
    #[bits(16)]
    pub clip: u16,
    #[bits(8)]
    pub tex: u8,
    #[bits(6)]
    pub pass: u8,
    pub layer: bool,
    pub raw: bool,
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
#[derive(Debug, Clone, Copy, Default, PartialEq, bytemuck::NoUninit)]
pub struct Data {
    pub pos: Vec2f,
    pub dim: Vec2f,
    pub uv: Vec2f,
    pub uvdim: Vec2f,
    pub color: u32, // Encoded as a non-linear, non-premultiplied sRGB32 color
    pub rotation: f32,
    pub flags: u32,
    pub _padding: [u8; 4], // We have to manually specify this to satisfy bytemuck
}

static_assertions::const_assert_eq!(std::mem::size_of::<Data>(), 48);

// Our shader will assemble a rotation based on this matrix, but transposed:
// [ cos(r) -sin(r) 0 (x - x*cos(r) + y*sin(r)) ]
// [ sin(r)  cos(r) 0 (y - x*sin(r) - y*cos(r)) ]
// [ 0       0      1 0 ]
// [ 0       0      0 1 ] ^ -1
