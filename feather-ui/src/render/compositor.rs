// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::sRGB32;
use crate::graphics::{Driver, Vec2f};
use crate::{AnyDim, AnyPoint, AnyRect, AnyVector, PxDim, RelDim, SourceID};
use derive_where::derive_where;
use guillotiere::euclid;
use num_traits::Zero;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::num::NonZero;
use std::sync::Arc;
use wgpu::wgt::SamplerDescriptor;
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, Buffer, BufferDescriptor, BufferUsages, TextureView,
};

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

pub const TARGET_BLEND: wgpu::ColorTargetState = wgpu::ColorTargetState {
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
                        min_binding_size: NonZero::new(size_of::<crate::Mat4x4>() as u64),
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
        mut area: AnyRect,
        dest: Option<AnyRect>,
        color: sRGB32,
        rotation: f32,
        force: bool,
    ) -> Option<Layer> {
        // Snap the layer area to the nearest pixel. This is necessary because the layer is treated
        // as a compositing target, which is always assumed to be on a pixel grid.
        let array = area.v.as_array_mut();
        array[0] = array[0].floor();
        array[1] = array[1].floor();
        array[2] = array[2].ceil();
        array[3] = array[3].ceil();

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

        if let Some(prev) = self.layers.read().get(&id)
            && *prev == layer
        {
            return None;
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
    pub area: AnyRect,
    // destination area that the layer is composited onto. Usually this is the same as area, but can be different when scaling down
    dest: AnyRect,
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
type CustomDrawFn = dyn FnMut(&Driver, &mut wgpu::RenderPass<'_>, AnyVector) + Send + Sync;

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
#[derive(Debug)]
pub struct Compositor {
    pipeline: wgpu::RenderPipeline,
    mvp: Buffer,
    clip: Buffer,
    clipdata: Vec<AnyRect>, // Clipping Rectangles
    pub(crate) segments: SmallVec<[HashMap<u8, Segment>; 1]>,
    view: std::sync::Weak<TextureView>,
    layer_view: std::sync::Weak<TextureView>,
    layer: bool, // Tells us which layer atlas to use (the first or second)
}

/// This stores the compositing data for a single render pass. The window compositor only ever has one segment, but the
/// compositors for the layer caches can have many different segments for each dependency layer and each target slice in
/// the layer atlas for that dependency layer. Each segment contains a set of CPU-side copy-regions to enable GPU
/// generation of compositing data, it's own deferred rendering queue, and its own list of custom draw commands.
#[derive_where(Debug)]
pub struct Segment {
    group: wgpu::BindGroup,
    buffer: Buffer,
    data: Vec<Data>,
    regions: Vec<std::ops::Range<u32>>,
    #[derive_where(skip)]
    defer: HashMap<u32, (Box<DeferFn>, AnyRect, AnyVector)>,
    #[derive_where(skip)]
    custom: Vec<(u32, Box<CustomDrawFn>, AnyVector)>,
}

impl Compositor {
    fn reserve(&mut self, driver: &Driver, pass: u8, slice: u8) {
        self.segments.resize_with(pass as usize + 1, HashMap::new);
        self.segments[pass as usize]
            .entry(slice)
            .or_insert_with(|| {
                let buffer = driver.device.create_buffer(&BufferDescriptor {
                    label: Some("Compositor Data"),
                    size: 32 * size_of::<Data>() as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let group = Self::gen_binding(
                    &self.mvp,
                    &buffer,
                    &self.clip,
                    &driver.shared,
                    &driver.device,
                    &driver.atlas.read().view,
                    &driver.layer_atlas[self.layer as usize].read().view,
                    &self.pipeline.get_bind_group_layout(0),
                );

                #[allow(clippy::single_range_in_vec_init)]
                Segment {
                    group,
                    buffer,
                    data: Vec::new(),
                    regions: vec![0..0],
                    defer: HashMap::new(),
                    custom: Vec::new(),
                }
            });
    }

    fn gen_binding(
        mvp: &Buffer,
        buffer: &Buffer,
        clip: &Buffer,
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
            size: std::mem::size_of::<crate::Mat4x4>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let clip = device.create_buffer(&BufferDescriptor {
            label: Some("Compositor Clip Data"),
            size: 4 * size_of::<AnyRect>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Compositor Data"),
            size: 32 * size_of::<Data>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let group = Self::gen_binding(
            &mvp,
            &buffer,
            &clip,
            shared,
            device,
            atlasview,
            layerview,
            &pipeline.get_bind_group_layout(0),
        );

        #[allow(clippy::single_range_in_vec_init)]
        let segment = Segment {
            group,
            buffer,
            data: Vec::new(),
            regions: vec![0..0],
            defer: HashMap::new(),
            custom: Vec::new(),
        };

        Self {
            pipeline,
            mvp,
            clip,
            clipdata: vec![AnyRect::zero()],
            segments: SmallVec::from_buf([HashMap::from_iter([(0, segment)])]),
            view: Arc::downgrade(atlasview),
            layer_view: Arc::downgrade(layerview),
            layer,
        }
    }

    /// Should be called when any external or internal buffer gets invalidated (such as atlas views)
    fn rebind(&mut self, shared: &Shared, device: &wgpu::Device, atlas: &Atlas, layers: &Atlas) {
        self.view = Arc::downgrade(&atlas.view);
        self.layer_view = Arc::downgrade(&layers.view);

        for slices in &mut self.segments {
            for segment in slices.values_mut() {
                segment.group = Self::gen_binding(
                    &self.mvp,
                    &segment.buffer,
                    &self.clip,
                    shared,
                    device,
                    &atlas.view,
                    &layers.view,
                    &self.pipeline.get_bind_group_layout(0),
                );
            }
        }
    }

    fn check_data(
        mvp: &Buffer,
        clip: &Buffer,
        layout: &wgpu::BindGroupLayout,
        segment: &mut Segment,
        shared: &Shared,
        device: &wgpu::Device,
        atlas: &Atlas,
        layers: &Atlas,
    ) {
        let size = segment.data.len() * size_of::<Data>();
        if (segment.buffer.size() as usize) < size {
            segment.buffer.destroy();
            segment.buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Compositor Data"),
                size: size.next_power_of_two() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            segment.group = Self::gen_binding(
                mvp,
                &segment.buffer,
                clip,
                shared,
                device,
                &atlas.view,
                &layers.view,
                layout,
            );
        }
    }

    fn check_clip(
        &mut self,
        shared: &Shared,
        device: &wgpu::Device,
        atlas: &Atlas,
        layers: &Atlas,
    ) {
        let size = self.clipdata.len() * size_of::<AnyRect>();
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

    #[inline]
    fn scissor_check(dim: &PxDim, clip: AnyRect) -> Result<AnyRect, AnyRect> {
        if AnyRect::new(0.0, 0.0, dim.width, dim.height) == clip {
            Err(clip)
        } else {
            Ok(clip)
        }
    }

    #[inline]
    fn clip_data(
        clipdata: &mut Vec<AnyRect>,
        cliprect: Result<AnyRect, AnyRect>,
        offset: AnyVector,
        mut data: Data,
    ) -> Option<Data> {
        match cliprect {
            Ok(clip) => {
                if data.rotation.is_zero() {
                    let (mut x, mut y, mut w, mut h) =
                        (data.pos.0[0], data.pos.0[1], data.dim.0[0], data.dim.0[1]);

                    // If the whole rect is outside the cliprect, don't render it at all.
                    if !clip.collides(&AnyRect::new(x, y, x + w, y + h)) {
                        // TODO: When we start reserving slots, this will need to instead insert a special zero size rect.
                        return None;
                    }

                    let (mut u, mut v, mut uw, mut vh) =
                        (data.uv.0[0], data.uv.0[1], data.uvdim.0[0], data.uvdim.0[1]);

                    // If rotation is zero, we don't need to do per-pixel clipping, we can just modify the rect itself.
                    let bounds = clip.v.as_array_ref();
                    let (min_x, min_y, max_x, max_y) = (bounds[0], bounds[1], bounds[2], bounds[3]);

                    // Get the ratio from our target rect to the source UV sampler rect
                    let uv_ratio = RelDim::new(uw / w, vh / h);

                    // Clip left edge
                    if x < min_x {
                        let right_shift = min_x - x;

                        x += right_shift;
                        u += right_shift * uv_ratio.width;
                        w -= right_shift;
                        uw -= right_shift * uv_ratio.width;
                    }

                    // Clip right edge
                    if x + w > max_x {
                        let right_shift = max_x - (x + w);
                        w += right_shift;
                        uw += right_shift * uv_ratio.width;
                    }

                    // Clip top edge
                    if y < min_y {
                        let bottom_shift = min_y - y;

                        y += bottom_shift;
                        v += bottom_shift * uv_ratio.height;
                        h -= bottom_shift;
                        vh -= bottom_shift * uv_ratio.height;
                    }

                    // Clip bottom edge
                    if y + h > max_y {
                        let bottom_shift = max_y - (y + h);
                        h += bottom_shift;
                        vh += bottom_shift * uv_ratio.height;
                    }

                    Some(Data {
                        pos: [x + offset.x, y + offset.y].into(),
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
                    if !clip.collides(&AnyRect::new(x, y, x + w, y + h)) {
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

    pub fn prepare(&mut self, driver: &Driver, _: &mut wgpu::CommandEncoder, surface_dim: PxDim) {
        // Check to see if we need to rebind either atlas view
        if self.view.strong_count() == 0 || self.layer_view.strong_count() == 0 {
            self.rebind(
                &driver.shared,
                &driver.device,
                &driver.atlas.read(),
                &driver.layer_atlas[self.layer as usize].read(),
            );
        }

        // Resolve all defers
        for slices in &mut self.segments {
            for segment in slices.values_mut() {
                for (idx, (f, clip, offset)) in segment.defer.drain() {
                    f(driver, &mut segment.data[idx as usize]);
                    segment.data[idx as usize].flags =
                        DataFlags::from_bits(segment.data[idx as usize].flags).into();
                    segment.data[idx as usize] = Self::clip_data(
                        &mut self.clipdata,
                        Self::scissor_check(&surface_dim, clip),
                        offset,
                        segment.data[idx as usize],
                    )
                    .unwrap_or_default();
                }

                if !segment.data.is_empty() {
                    Self::check_data(
                        &self.mvp,
                        &self.clip,
                        &self.pipeline.get_bind_group_layout(0),
                        segment,
                        &driver.shared,
                        &driver.device,
                        &driver.atlas.read(),
                        &driver.layer_atlas[self.layer as usize].read(),
                    );

                    // TODO turn into write_buffer_with (is that actually going to be faster?)
                    let mut offset = 0;
                    for range in &segment.regions {
                        let len = range.end - range.start;
                        driver.queue.write_buffer(
                            &segment.buffer,
                            range.start as u64,
                            bytemuck::cast_slice(
                                &segment.data.as_slice()[offset as usize..(offset + len) as usize],
                            ),
                        );
                        offset += len;
                    }
                }
            }
        }

        driver.queue.write_buffer(
            &self.mvp,
            0,
            bytemuck::cast_slice(
                &crate::graphics::mat4_proj(
                    0.0,
                    surface_dim.height,
                    surface_dim.width,
                    -(surface_dim.height),
                    0.2,
                    10000.0,
                )
                .to_array(),
            ),
        );

        // Very important that we do this AFTER resolving all defers, since those can add cliprects
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
    }

    #[inline]
    fn append_internal(
        &mut self,
        clipstack: &[AnyRect],
        surface_dim: PxDim,
        layer_offset: AnyVector,
        pos: AnyPoint,
        dim: AnyDim,
        uv: AnyPoint,
        uvdim: AnyDim,
        color: u32,
        rotation: f32,
        tex: u8,
        pass: u8,
        slice: u8,
        raw: bool,
        layer: bool,
    ) -> u32 {
        let data = Data {
            pos: pos.to_array().into(),
            dim: dim.to_array().into(),
            uv: uv.to_array().into(),
            uvdim: uvdim.to_array().into(),
            color,
            rotation,
            flags: DataFlags::new()
                .with_tex(tex)
                .with_raw(raw)
                .with_layer(layer)
                .into(),
            ..Default::default()
        };

        if let Some(d) = Self::clip_data(
            &mut self.clipdata,
            clipstack
                .last()
                .ok_or(surface_dim.to_untyped().into())
                .copied(),
            layer_offset,
            data,
        ) {
            self.preprocessed(d, pass, slice)
        } else {
            u32::MAX // TODO: Once we start reserving slots, we will always need to return a valid one by inserting an empty rect in clip_data
        }
    }

    #[inline]
    fn preprocessed(&mut self, data: Data, index: u8, slice: u8) -> u32 {
        let segment = &mut self.segments[index as usize].get_mut(&slice).unwrap();
        let region = segment.regions.last_mut().unwrap();
        if region.end == u32::MAX {
            panic!(
                "Still processing a compute operation! Finish it by calling set_compute_buffer() first."
            );
        }

        let idx = region.end;
        segment.data.push(data);
        region.end += 1;
        idx
    }

    pub fn draw(&mut self, driver: &Driver, pass: &mut wgpu::RenderPass<'_>, index: u8, slice: u8) {
        let segment = &mut self.segments[index as usize].get_mut(&slice).unwrap();

        let mut last_index = 0;
        for (i, f, draw_offset) in &mut segment.custom {
            if last_index < *i {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &segment.group, &[0]);
                pass.draw(last_index..*i, 0..1);
            }
            last_index = *i;
            f(driver, pass, *draw_offset);
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &segment.group, &[0]);
        pass.draw(last_index..(segment.regions.last().unwrap().end * 6), 0..1);
    }

    pub fn cleanup(&mut self) {
        for slices in &mut self.segments {
            for segment in slices.values_mut() {
                segment.data.clear();
                segment.regions.clear();
                segment.regions.push(0..0);
            }
        }

        self.clipdata.clear();
        self.clipdata.push(AnyRect::zero());
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
    pub clipstack: &'a mut Vec<AnyRect>,
    pub offset: AnyVector,
    pub surface_dim: PxDim, // Dimension of the top-level window surface.
    pub pass: u8,
    pub slice: u8, // This is the atlas slice index that this is being rendered to
}

impl<'a> CompositorView<'a> {
    #[inline]
    pub fn with_clip<T>(
        &mut self,
        clip: AnyRect,
        f: impl FnOnce(&mut Self) -> Result<T, crate::Error>,
    ) -> Result<T, crate::Error> {
        if let Some(prev) = self.clipstack.last() {
            self.clipstack.push(clip.intersect(*prev));
        } else {
            self.clipstack.push(clip);
        }
        let r = f(self);
        self.clipstack
            .pop()
            .expect("Tried to pop a clipping rect but the stack was empty!");
        r
    }

    #[inline]
    pub fn current_clip(&self) -> AnyRect {
        *self
            .clipstack
            .last()
            .unwrap_or(&self.surface_dim.to_untyped().into())
    }

    #[inline]
    pub fn segment(&mut self) -> &mut Segment {
        let pass = self.pass;
        let slice = self.slice;
        self.compositor().segments[pass as usize]
            .get_mut(&slice)
            .unwrap()
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
        pos: AnyPoint,
        dim: AnyDim,
        uv: AnyPoint,
        uvdim: AnyDim,
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
            self.slice,
            raw,
            false,
        )
    }

    #[inline]
    pub(crate) fn append_layer(
        &mut self,
        layer: &Layer,
        parent_pos: AnyPoint,
        uv: guillotiere::Rectangle,
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
            layer.dest.topleft() + parent_pos.to_vector(),
            layer.dest.dim(),
            uv.min.to_f32().to_array().into(),
            uv.size().to_f32().to_array().into(),
            layer.color.rgba,
            layer.rotation,
            0,
            self.pass,
            self.slice,
            false,
            true,
        )
    }

    #[inline]
    pub fn preprocessed(&mut self, mut data: Data) -> u32 {
        data.pos.0[0] += self.offset.x;
        data.pos.0[1] += self.offset.y;
        data.flags = DataFlags::from_bits(data.flags).into();
        let pass = self.pass;
        let slice = self.slice;
        self.compositor().preprocessed(data, pass, slice)
    }

    #[inline]
    pub fn defer(&mut self, f: impl FnOnce(&Driver, &mut Data) + Send + Sync + 'static) {
        let clip = self.current_clip();
        let offset = self.offset;
        let segment = self.segment();
        let region = segment.regions.last_mut().unwrap();
        if region.end == u32::MAX {
            panic!(
                "Still processing a compute operation! Finish it by calling set_compute_buffer() first."
            );
        }

        let idx = region.end;
        segment.data.push(Default::default());
        segment.defer.insert(idx, (Box::new(f), clip, offset));
        region.end += 1;
    }

    pub fn append_custom(
        &mut self,
        f: impl FnMut(&Driver, &mut wgpu::RenderPass<'_>, AnyVector) + Send + Sync + 'static,
    ) {
        let index = self.segment().regions.last().unwrap().end;
        if index == u32::MAX {
            panic!(
                "Still processing a compute operation! Finish it by calling set_compute_buffer() first."
            );
        }
        let offset = self.offset;
        self.segment().custom.push((index, Box::new(f), offset));
    }

    /// Returns the GPU buffer and the current offset, which allows a compute shader to accumulate commands
    /// in the GPU buffer directly, provided it calls set_compute_buffer afterwards with the command count.
    /// Attempting to insert a non-GPU command before calling set_compute_buffer will panic.
    pub fn get_compute_buffer(&mut self) -> (&Buffer, u32) {
        let offset = self.segment().regions.last().unwrap().end;
        if offset == u32::MAX {
            panic!(
                "Still processing a compute operation! Finish it by calling set_compute_buffer() first."
            );
        }
        self.segment().regions.push(offset..u32::MAX);
        (&self.segment().buffer, offset)
    }

    /// After executing a compute shader that added a series of compositor commands to the command buffer,
    /// this must be called with the number of commands that were contiguously inserted into the buffer.
    pub fn set_compute_buffer(&mut self, count: u32) {
        let region = self.segment().regions.last_mut().unwrap();
        region.start += count;
        region.end = region.start;
    }

    pub fn reserve(&mut self, driver: &Driver) {
        let pass = self.pass;
        let slice = self.slice;
        self.compositor().reserve(driver, pass, slice);
    }
}

#[bitfield_struct::bitfield(u32)]
pub struct DataFlags {
    #[bits(16)]
    pub clip: u16,
    #[bits(8)]
    pub tex: u8,
    #[bits(6)]
    pub __: u8,
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
