use crate::component::region::Region;
use crate::layout::Layout;
use std::borrow::Cow;
use ultraviolet::Mat4x4;
use wgpu::Device;
use wgpu::RenderPipeline;

use super::Renderable;

pub struct RoundRectPipeline {
    pipeline: RenderPipeline,
}

impl<AppData> Renderable<AppData> for RoundRectPipeline {
    fn render(&self, area: crate::AbsRect) -> im::Vector<crate::RenderInstruction> {
        let result = im::Vector::new();
        result.push_back(Box::new(
            |pass: &wgpu::RenderPass, device: &wgpu::Device| pass.set_pipeline(&self.pipeline),
        ));
        result
    }
}

#[derive(Clone)]
pub struct RoundRect<Parent: Clone> {
    props: Parent,
}

impl<AppData, Parent: Clone> super::Component<AppData, Parent> for RoundRect<Parent> {
    fn layout(&self, data: &AppData) -> Box<dyn Layout<Parent, AppData>> {
        let map = VectorMap::new(
            |child: &Box<ComponentFrom<AppData, Basic>>|
             -> Box<dyn Layout<<Basic as Desc<AppData>>::Impose, AppData>> { child.layout(data) },
        );

        let (_, children) = map.call(Default::default(), &self.children);
        Box::new(layout::Node::<AppData, Basic, Parent> {
            props: self.basic.clone(),
            imposed: self.props.clone(),
            children,
            events: std::rc::Weak::<EventList<AppData>>::new(),
            renderable: None,
        })
    }
}

pub fn pipeline(device: Device, config: &wgpu::SurfaceConfiguration) -> RenderPipeline {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[0, 1, 2, 3, 4, 5].map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("RoundRect"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let standard_vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Standard VS"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/standard.wgsl"))),
    });

    let round_rect = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("RoundRect"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/RoundRect.wgsl"))),
    });

    let test: Mat4x4;
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MVP"),
        contents: test.as_byte_slice(),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &standard_vs,
            entry_point: "main",
            buffers: &[],
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
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}
