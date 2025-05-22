use ultraviolet::Vec4;

use crate::{DriverState, RenderLambda};

pub struct Pipeline {
    pub this: std::rc::Weak<Pipeline>,
    pub pipeline: wgpu::RenderPipeline,
    pub group: wgpu::BindGroup,
    pub vertices: wgpu::Buffer,
    pub buffers: [wgpu::Buffer; 6],
    pub padding: crate::AbsRect,
}

impl super::Renderable for Pipeline {
    fn render(
        &self,
        area: crate::AbsRect,
        driver: &DriverState,
    ) -> im::Vector<crate::RenderInstruction> {
        driver.queue.write_buffer(
            &self.buffers[1],
            0,
            Vec4::new(
                area.topleft().x + self.padding.topleft().x,
                area.topleft().y + self.padding.topleft().y,
                area.bottomright().x - area.topleft().x - self.padding.bottomright().x,
                area.bottomright().y - area.topleft().y - self.padding.bottomright().y,
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
