use crate::component::region::Region;
use crate::layout;
use crate::layout::empty::Empty;
use crate::layout::EventList;
use crate::layout::Layout;
use crate::DriverState;
use crate::RenderLambda;
use std::borrow::Cow;
use std::mem::size_of;
use std::rc::Rc;
use std::rc::Weak;
use ultraviolet::Mat4;
use ultraviolet::Vec2;
use ultraviolet::Vec4;
use wgpu::util::DeviceExt;
use wgpu::Device;
use wgpu::RenderPipeline;

use super::Renderable;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Vertex {
    pos: [f32; 2],
}

pub struct RoundRectPipeline {
    this: Weak<RoundRectPipeline>,
    pipeline: RenderPipeline,
    group: wgpu::BindGroup,
    vertices: wgpu::Buffer,
}

impl<AppData> Renderable<AppData> for RoundRectPipeline {
    fn render(&self, area: crate::AbsRect) -> im::Vector<crate::RenderInstruction> {
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

#[derive(Clone, Default)]
pub struct RoundRect<Parent: Clone> {
    pub props: Parent,
    pub border: f32,
    pub blur: f32,
    pub corners: Vec4,
    pub fill: Vec4,
    pub outline: Vec4,
}

// TODO: this does not work for some reason
fn mat4x4_proj(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4 {
    let mut m = Mat4 {
        cols: [
            Vec4::new(2.0 / (r - l), 0.0, 0.0, 0.0).into(),
            Vec4::new(0.0, 2.0 / (t - b), 0.0, 0.0).into(),
            Vec4::new(0.0, 0.0, -((f + n) / (f - n)), 1.0).into(),
            Vec4::new(
                (r + l) / (r - l),
                (t + b) / (t - b),
                ((2.0 * f * n) / (f - n)) + 1.0,
                1.0,
            )
            .into(),
        ],
    };
    m.transposed()
}

// Orthographic projection matrix
fn mat4_ortho(x: f32, y: f32, w: f32, h: f32, n: f32, f: f32) -> Mat4 {
    let mut m = Mat4 {
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
    };
    m
}

fn gen_uniform(driver: &DriverState, name: &str, buffer: &[u8]) -> wgpu::Buffer {
    driver
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: buffer,
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

fn to_bytes<T>(v: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<T>())
    }
}

impl<AppData: 'static, Parent: Clone + 'static> super::Component<AppData, Parent>
    for RoundRect<Parent>
{
    fn layout(
        &self,
        data: &AppData,
        driver: &DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Parent, AppData>> {
        let pipeline = gen_pipeline(&driver.device, config);

        let mvp = driver
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MVP"),
                contents: mat4_ortho(
                    0.0,
                    config.height as f32,
                    config.width as f32,
                    -(config.height as f32),
                    0.0,
                    1.0,
                )
                .as_byte_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let vertex_buf = driver
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VertBuffer"),
                contents: to_bytes(&[
                    Vertex { pos: [0.0, 0.0] },
                    Vertex { pos: [400.0, 0.0] },
                    Vertex { pos: [0.0, 200.0] },
                    Vertex {
                        pos: [400.0, 200.0],
                    },
                ]),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let inflate = gen_uniform(driver, "Inflate", Vec2::new(1.0, 1.0).as_byte_slice());
        let dimborderblur = gen_uniform(
            driver,
            "DimBorderBlur",
            Vec4::new(200.0, 200.0, self.border, self.blur).as_byte_slice(),
        );
        let corners = gen_uniform(driver, "Corners", self.corners.as_byte_slice());
        let fill = gen_uniform(driver, "Fill", self.fill.as_byte_slice());
        let outline = gen_uniform(driver, "Outline", self.outline.as_byte_slice());
        let buffers = [mvp, inflate, dimborderblur, corners, fill, outline];
        let buffers: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, x)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: x.as_entire_binding(),
            })
            .collect();

        let bind_group = driver.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipeline.get_bind_group_layout(0),
            entries: &buffers,
            label: None,
        });

        Box::new(layout::Node::<AppData, Empty, Parent> {
            props: (),
            imposed: self.props.clone(),
            children: Default::default(),
            events: std::rc::Weak::<EventList<AppData>>::new(),
            renderable: Some(Rc::new_cyclic(|this| RoundRectPipeline {
                this: this.clone(),
                pipeline,
                group: bind_group,
                vertices: vertex_buf,
            })),
        })
    }
}

fn gen_pipeline(device: &Device, config: &wgpu::SurfaceConfiguration) -> RenderPipeline {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[0, 1, 2, 3, 4, 5].map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: if i < 2 {
                wgpu::ShaderStages::VERTEX
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

    let round_rect = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("RoundRect FS"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/RoundRect.wgsl"))),
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
            module: &round_rect,
            entry_point: "main",
            compilation_options: Default::default(),
            targets: &[Some(config.view_formats[0].into())],
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
