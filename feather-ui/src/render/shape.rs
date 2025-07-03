// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::compositor;
use crate::color::sRGB;
use crate::graphics::{self, Vec2f, Vec4f};
use crate::render::atlas::Atlas;
use crate::render::compositor::Compositor;
use crate::shaders;
use guillotiere::Size;
use num_traits::Zero;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::num::NonZero;
use ultraviolet::Vec4;
use wgpu::BindGroupLayout;

pub struct Instance<PIPELINE: crate::render::Pipeline<Data = Data> + 'static> {
    pub padding: crate::AbsRect,
    pub border: f32,
    pub blur: f32,
    pub fill: sRGB,
    pub outline: sRGB,
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
        driver: &crate::graphics::Driver,
        compositor: &mut Compositor,
    ) -> Result<(), crate::Error> {
        let dim = area.bottomright() - area.topleft() - self.padding.bottomright();
        debug_assert!(dim.x.is_zero() || dim.x.is_sign_positive());
        debug_assert!(dim.y.is_zero() || dim.y.is_sign_positive());

        let (region_uv, region_index) = {
            let mut atlas = driver.atlas.write();
            let region = atlas.cache_region(
                &driver.device,
                self.id.clone(),
                Size::new(area.dim().0.x.ceil() as i32, area.dim().0.y.ceil() as i32),
            )?;
            (region.uv, region.index)
        };

        // The region dimensions here can be wrong, because the region is rounded up to the nearest pixel.
        // However, properly fixing this requires changing how the SDF shader works so it can properly
        // emulate conservative rasterization. For now, we keep our original behavior of rounding up and
        // then letting the compositor squish the result slightly, which is actually pretty accurate.
        // TODO: Change this to be pixel-perfect by outputting the exact dimensions instead of rounded ones.

        driver.with_pipeline::<PIPELINE>(|pipeline| {
            pipeline.append(
                &Data {
                    pos: region_uv.min.to_f32().to_array().into(),
                    dim: region_uv.size().to_f32().to_array().into(),
                    border: self.border,
                    blur: self.blur,
                    corners: self.corners.as_array().into(),
                    fill: self.fill.as_32bit().rgba,
                    outline: self.outline.as_32bit().rgba,
                },
                region_index,
            )
        });

        compositor.append_data(
            area.topleft() + self.padding.topleft(),
            dim,
            region_uv.min.to_f32().to_array().into(),
            region_uv.size().to_f32().to_array().into(),
            0xFFFFFFFF,
            0.0,
            region_index,
            false,
        );

        Ok(())
    }
}

// Renderdoc Format:
// struct Data {
// 	float corners[4];
// 	float pos[2];
// 	float dim[2];
// 	float border;
// 	float blur;
// 	uint32_t fill;
// 	uint32_t outline;
// };
// Data d[];

#[derive(Debug, Clone, Copy, Default, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct Data {
    pub corners: Vec4f,
    pub pos: Vec2f,
    pub dim: Vec2f,
    pub border: f32,
    pub blur: f32,
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
        self.data.entry(layer).or_default().push(*data);
    }

    fn draw(&mut self, driver: &graphics::Driver, pass: &mut wgpu::RenderPass<'_>, layer: u16) {
        if let Some(data) = self.data.get_mut(&layer) {
            let size = data.len() * size_of::<Data>();
            if (self.buffer.size() as usize) < size {
                self.buffer.destroy();
                self.buffer = driver.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Shape Data"),
                    size: size.next_power_of_two() as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.group = Self::rebind(
                    &self.buffer,
                    &self.pipeline.get_bind_group_layout(0),
                    &driver.device,
                    &driver.atlas.read(),
                );
            }

            driver
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
            label: Some("Shape Bind Group"),
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
        shaders::load_wgsl(device, "Shape", shaders::get("shape.wgsl").unwrap())
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
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                targets: &[Some(compositor::TARGET_STATE)],
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
    }

    fn rebind(
        buffer: &wgpu::Buffer,
        layout: &BindGroupLayout,
        device: &wgpu::Device,
        atlas: &Atlas,
    ) -> wgpu::BindGroup {
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

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &bindings,
            label: None,
        })
    }

    fn new(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
        entry_point: &str,
    ) -> Self {
        let buffer = driver.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shape Data"),
            size: 32 * size_of::<Data>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = Self::pipeline(layout, shader, &driver.device, entry_point);

        let group = Self::rebind(
            &buffer,
            &pipeline.get_bind_group_layout(0),
            &driver.device,
            &driver.atlas.read(),
        );

        Self {
            data: HashMap::new(),
            buffer,
            pipeline,
            group,
        }
    }
}

impl Shape<0> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::new(layout, shader, driver, "rectangle"))
    }
}

impl Shape<1> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::new(layout, shader, driver, "triangle"))
    }
}

impl Shape<2> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::new(layout, shader, driver, "circle"))
    }
}

impl Shape<3> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::new(layout, shader, driver, "arcs"))
    }
}
