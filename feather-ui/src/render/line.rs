use crate::{
    shaders::{to_bytes, Vertex},
    DriverState, RenderLambda,
};
use std::rc::Rc;
use ultraviolet::Vec2;
use wgpu::util::DeviceExt;

pub struct Pipeline {
    pub this: std::rc::Weak<Pipeline>,
    pub pipeline: wgpu::RenderPipeline,
    pub group: wgpu::BindGroup,
    pub start: Vec2,
    pub end: Vec2,
    pub _buffers: [wgpu::Buffer; 3],
}

impl super::Renderable for Pipeline {
    fn render(
        &self,
        _: crate::AbsRect,
        driver: &DriverState,
    ) -> im::Vector<crate::RenderInstruction> {
        // We have to put this in an RC because it needs to be cloneable across multiple render passes
        let verts = Rc::new(
            driver
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("LineVertices"),
                    contents: to_bytes(&[
                        Vertex {
                            pos: self.start.into(),
                        },
                        Vertex {
                            pos: self.end.into(),
                        },
                    ]),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
        );

        let weak = self.this.clone();
        let mut result = im::Vector::new();
        result.push_back(Some(Box::new(move |pass: &mut wgpu::RenderPass| {
            if let Some(this) = weak.upgrade() {
                pass.set_vertex_buffer(0, verts.slice(..));
                pass.set_bind_group(0, &this.group, &[]);
                pass.set_pipeline(&this.pipeline);
                pass.draw(
                    //0..(this.vertices.size() as u32 / size_of::<Vertex>() as u32),
                    0..2,
                    0..1,
                );
            }
        }) as Box<dyn RenderLambda>));
        result
    }
}
