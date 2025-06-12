// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::collections::HashMap;
use ultraviolet::{Mat4, Vec4};
use wgpu::util::DeviceExt;

// This maps x and y to the viewpoint size, maps input_z from [n,f] to [0,1], and sets
// output_w = input_z for perspective. Requires input_w = 1
pub fn mat4_proj(x: f32, y: f32, w: f32, h: f32, n: f32, f: f32) -> Mat4 {
    Mat4 {
        cols: [
            Vec4::new(2.0 / w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0 / (f - n), 1.0),
            Vec4::new(-(2.0 * x + w) / w, -(2.0 * y + h) / h, -n / (f - n), 0.0),
        ],
    }
}

// Orthographic projection matrix
#[allow(dead_code)]
pub fn mat4_ortho(x: f32, y: f32, w: f32, h: f32, n: f32, f: f32) -> Mat4 {
    Mat4 {
        cols: [
            Vec4::new(2.0 / w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -2.0 / (f - n), 0.0),
            Vec4::new(
                -(2.0 * x + w) / w,
                -(2.0 * y + h) / h,
                (f + n) / (f - n),
                1.0,
            ),
        ],
    }
}

pub fn gen_uniform(graphics: &crate::graphics::State, name: &str, buffer: &[u8]) -> wgpu::Buffer {
    graphics
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: buffer,
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

pub fn default_vertex_buffer(graphics: &crate::graphics::State) -> wgpu::Buffer {
    graphics
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VertBuffer"),
            contents: to_bytes(&[
                Vertex { pos: [0.0, 0.0] },
                Vertex { pos: [1.0, 0.0] },
                Vertex { pos: [0.0, 1.0] },
                Vertex { pos: [1.0, 1.0] },
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        })
}

pub fn to_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 2],
}

#[derive(Debug)]
pub struct ShaderCache {
    shaders: Vec<wgpu::ShaderModule>,
    layouts: Vec<wgpu::PipelineLayout>,
    //_pipelines: HashMap<(usize, wgpu::SurfaceConfiguration), std::rc::Weak<wgpu::RenderPipeline>>,
    shader_hash: HashMap<String, usize>,
    pub basic_vs: usize,
    pub line_vs: usize,
    pub basic_pipeline: usize,
    pub line_pipeline: usize,
}
static_assertions::assert_impl_all!(ShaderCache: Send, Sync);

impl ShaderCache {
    pub fn new(device: &wgpu::Device) -> Self {
        let mut this = ShaderCache {
            shaders: Default::default(),
            layouts: Default::default(),
            //_pipelines: Default::default(),
            shader_hash: Default::default(),
            basic_pipeline: 0,
            line_pipeline: 0,
            basic_vs: 0,
            line_vs: 0,
        };

        this.basic_vs = this.register_shader(device, "Standard VS", include_str!("standard.wgsl"));

        this.line_vs = this.register_shader(device, "Line VS", include_str!("Line.vert.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[0, 1, 2, 3, 4, 5].map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: if i < 2 {
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
                } else {
                    wgpu::ShaderStages::FRAGMENT
                },
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });

        this.basic_pipeline = this.register_pipeline_layout(
            device,
            &wgpu::PipelineLayoutDescriptor {
                label: Some("StandardPipeline"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            },
        );

        let line_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[0, 1, 2].map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: if i < 2 {
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
                } else {
                    wgpu::ShaderStages::FRAGMENT
                },
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });

        this.line_pipeline = this.register_pipeline_layout(
            device,
            &wgpu::PipelineLayoutDescriptor {
                label: Some("LinePipeline"),
                bind_group_layouts: &[&line_group_layout],
                push_constant_ranges: &[],
            },
        );

        this
    }

    pub fn register_shader(&mut self, device: &wgpu::Device, label: &str, shader: &str) -> usize {
        if let Some(idx) = self.shader_hash.get(shader) {
            *idx
        } else {
            let idx = self.shaders.len();
            self.shaders
                .push(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(shader.into()),
                }));
            self.shader_hash.insert(shader.into(), idx);
            idx
        }
    }

    pub fn register_pipeline_layout(
        &mut self,
        device: &wgpu::Device,
        desc: &wgpu::PipelineLayoutDescriptor<'_>,
    ) -> usize {
        let idx = self.layouts.len();
        self.layouts.push(device.create_pipeline_layout(desc));
        idx
    }

    pub fn standard_pipeline(
        &self,
        device: &wgpu::Device,
        fragment: usize,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&self.layouts[self.basic_pipeline]),
            vertex: wgpu::VertexState {
                module: &self.shaders[self.basic_vs],
                entry_point: None,
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shaders[fragment],
                entry_point: None,
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.view_formats[0],
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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

    pub fn line_pipeline(
        &self,
        device: &wgpu::Device,
        fragment: usize,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&self.layouts[self.line_pipeline]),
            vertex: wgpu::VertexState {
                module: &self.shaders[self.basic_vs],
                entry_point: None,
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shaders[fragment],
                entry_point: None,
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.view_formats[0],
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                topology: wgpu::PrimitiveTopology::LineStrip,
                //conservative: true,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }
}
