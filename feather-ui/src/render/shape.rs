// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::compositor;
use crate::render::compositor::Compositor;
use crate::{graphics, vec4_to_u32};
use guillotiere::Size;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::num::NonZero;
use ultraviolet::Vec4;

pub struct Instance<PIPELINE: crate::render::Pipeline<Data = Data> + 'static> {
    pub padding: crate::AbsRect,
    pub border: f32,
    pub blur: f32,
    pub fill: Vec4,
    pub outline: Vec4,
    pub corners: Vec4,
    pub id: std::rc::Rc<crate::SourceID>,
    pub phantom: PhantomData<PIPELINE>,
}

impl<PIPELINE: crate::render::Pipeline<Data = Data> + 'static> super::Renderable
    for Instance<PIPELINE>
{
    fn render(
        &self,
        area: crate::AbsRect,
        graphics: &crate::graphics::Driver,
        compositor: &mut Compositor,
    ) {
        let dim = *(area.bottomright() - area.topleft() - self.padding.bottomright()).as_array();
        let (region_uv, region_index) = {
            let mut atlas = graphics.atlas.write();
            let region = atlas.cache_region(
                &graphics.device,
                self.id.clone(),
                Size::new(area.dim().0.x.ceil() as i32, area.dim().0.y.ceil() as i32),
            );
            (region.uv, region.index)
        };

        graphics.with_pipeline::<PIPELINE>(|pipeline| {
            pipeline.append(
                &Data {
                    pos: *(area.topleft() + self.padding.topleft()).as_array(),
                    dim: dim,
                    border: self.border,
                    blur: self.blur,
                    corners: *self.corners.as_array(),
                    fill: vec4_to_u32(&self.fill),
                    outline: vec4_to_u32(&self.outline),
                },
                region_index,
            )
        });

        compositor.append(&compositor::Data {
            pos: *(area.topleft() + self.padding.topleft()).as_array(),
            dim: dim,
            uv: region_uv.min.to_array(),
            uvdim: region_uv.size().to_array(),
            color: 0xFFFFFFFF,
            rotation: 0.0,
            tex: region_index,
            clip: 0,
        });
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct Data {
    pub pos: [f32; 2],
    pub dim: [f32; 2],
    pub border: f32,
    pub blur: f32,
    pub corners: [f32; 4],
    pub fill: u32,
    pub outline: u32,
}

#[derive(Debug)]
pub struct Shape<const KIND: u8> {
    data: HashMap<u16, Vec<Data>>,
    buffer: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
}

impl<const KIND: u8> super::Pipeline for Shape<KIND> {
    type Data = Data;

    fn append(&mut self, data: &Self::Data, layer: u16) {
        self.data.entry(layer).or_insert_with(Vec::new).push(*data);
    }

    fn draw(&mut self, graphics: &graphics::Driver, pass: &mut wgpu::RenderPass<'_>, layer: u16) {
        if let Some(data) = self.data.get_mut(&layer) {
            graphics
                .queue
                .write_buffer(&self.buffer, 0, bytemuck::cast_slice(data.as_slice()));

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[0]);
            pass.draw(0..(data.len() as u32 * 6), 0..1);
            data.clear();
        }
    }
}

impl<const KIND: u8> Shape<KIND> {
    pub fn layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
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

        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shape Pipeline"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        })
    }

    pub fn shader(device: &wgpu::Device) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shape"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shape.wgsl").into()),
        })
    }

    fn pipeline(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        device: &wgpu::Device,
        entry_point: &str,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                targets: &[Some(compositor::TARGET_STATE)],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        graphics: &graphics::Driver,
        entry_point: &str,
    ) -> Self {
        let buffer = graphics.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Data"),
            size: 32 * size_of::<Data>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let atlas = graphics.atlas.read();

        let bindings = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: atlas.mvp.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: atlas.extent_buf.as_entire_binding(),
            },
        ];

        let pipeline = Self::pipeline(layout, shader, &graphics.device, entry_point);
        let group = graphics
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &pipeline.get_bind_group_layout(0),
                entries: &bindings,
                label: None,
            });

        Self {
            data: HashMap::new(),
            buffer,
            pipeline,
            group,
        }
    }
}

impl Shape<0> {
    pub fn new(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        graphics: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::create(layout, shader, graphics, "rectangle"))
    }
}

impl Shape<1> {
    pub fn new(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        graphics: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::create(layout, shader, graphics, "triangle"))
    }
}

impl Shape<2> {
    pub fn new(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        graphics: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::create(layout, shader, graphics, "circle"))
    }
}

impl Shape<3> {
    pub fn new(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        graphics: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::create(layout, shader, graphics, "arcs"))
    }
}
