use crate::outline::Renderable;
use crate::DriverState;
use crate::RenderLambda;
use std::borrow::Cow;
use ultraviolet::{Mat4, Vec4};
use wgpu::util::DeviceExt;

// This maps x and y to the viewpoint size, maps input_z from [n,f] to [0,1], and sets
// output_w = input_z for perspective. Requires input_w = 1
pub fn mat4_proj(x: f32, y: f32, w: f32, h: f32, n: f32, f: f32) -> Mat4 {
    Mat4 {
        cols: [
            Vec4::new(2.0 / w, 0.0, 0.0, 0.0).into(),
            Vec4::new(0.0, 2.0 / h, 0.0, 0.0).into(),
            Vec4::new(0.0, 0.0, 1.0 / (f - n), 1.0).into(),
            Vec4::new(-(2.0 * x + w) / w, -(2.0 * y + h) / h, -n / (f - n), 0.0).into(),
        ],
    }
}

// Orthographic projection matrix
pub fn mat4_ortho(x: f32, y: f32, w: f32, h: f32, n: f32, f: f32) -> Mat4 {
    Mat4 {
        cols: [
            Vec4::new(2.0 / w, 0.0, 0.0, 0.0).into(),
            Vec4::new(0.0, 2.0 / h, 0.0, 0.0).into(),
            Vec4::new(0.0, 0.0, -2.0 / (f - n), 0.0).into(),
            Vec4::new(
                -(2.0 * x + w) / w,
                -(2.0 * y + h) / h,
                (f + n) / (f - n),
                1.0,
            )
            .into(),
        ],
    }
}

pub fn gen_uniform(driver: &DriverState, name: &str, buffer: &[u8]) -> wgpu::Buffer {
    driver
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: buffer,
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

pub fn default_vertex_buffer(driver: &DriverState) -> wgpu::Buffer {
    driver
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
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<T>())
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 2],
}

pub fn standard_pipeline(
    device: &wgpu::Device,
    fragment: wgpu::ShaderModule,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::RenderPipeline {
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("RoundRectPipeline"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let standard_vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Standard VS"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/standard.wgsl"))),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &standard_vs,
            entry_point: "main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment,
            entry_point: "main",
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

pub struct StandardPipeline {
    pub this: std::rc::Weak<StandardPipeline>,
    pub pipeline: wgpu::RenderPipeline,
    pub group: wgpu::BindGroup,
    pub vertices: wgpu::Buffer,
    pub buffers: [wgpu::Buffer; 6],
}

impl Renderable for StandardPipeline {
    fn render(
        &self,
        area: crate::AbsRect,
        driver: &DriverState,
    ) -> im::Vector<crate::RenderInstruction> {
        driver.queue.write_buffer(
            &self.buffers[1],
            0,
            Vec4::new(
                area.topleft.x,
                area.topleft.y,
                area.bottomright.x - area.topleft.x,
                area.bottomright.y - area.topleft.y,
            )
            .as_byte_slice(),
        );

        let weak = self.this.clone();
        let mut result = im::Vector::new();
        result.push_back(Some(Box::new(move |pass: &mut wgpu::RenderPass| {
            if let Some(this) = weak.upgrade() {
                pass.set_vertex_buffer(0, this.vertices.slice(..));
                pass.set_bind_group(0, &this.group, &[]);
                pass.set_pipeline(&this.pipeline);
                pass.draw(
                    //0..(this.vertices.size() as u32 / size_of::<Vertex>() as u32),
                    0..4,
                    0..1,
                );
            }
        }) as Box<dyn RenderLambda>));
        result
    }
}
