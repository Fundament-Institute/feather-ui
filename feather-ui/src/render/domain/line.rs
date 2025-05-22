use crate::{
    shaders::{to_bytes, Vertex},
    CrossReferenceDomain, DriverState, RenderLambda, SourceID,
};
use std::rc::Rc;
use wgpu::util::DeviceExt;

pub struct Pipeline {
    pub this: std::rc::Weak<Pipeline>,
    pub pipeline: wgpu::RenderPipeline,
    pub group: wgpu::BindGroup,
    pub domain: Rc<CrossReferenceDomain>,
    pub start: Rc<SourceID>,
    pub end: Rc<SourceID>,
    pub _buffers: [wgpu::Buffer; 3],
}

impl super::Renderable for Pipeline {
    fn render(
        &self,
        _: crate::AbsRect,
        driver: &DriverState,
    ) -> im::Vector<crate::RenderInstruction> {
        // TODO: This needs to be deferred until the end of the pipeline and then re-inserted back into place to remove dependency cycles
        let start = self.domain.get_area(&self.start).unwrap_or_default();
        let end = self.domain.get_area(&self.end).unwrap_or_default();

        // We have to put this in an RC because it needs to be cloneable across multiple render passes
        let verts = Rc::new(
            driver
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("LineVertices"),
                    contents: to_bytes(&[
                        Vertex {
                            pos: ((start.topleft() + start.bottomright()) * 0.5).into(),
                        },
                        Vertex {
                            pos: ((end.topleft() + end.bottomright()) * 0.5).into(),
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
