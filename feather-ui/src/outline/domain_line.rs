// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::CrossReferenceDomain;
use crate::layout;
use crate::layout::base;
use crate::layout::Layout;
use crate::outline::Renderable;
use crate::shaders::gen_uniform;
use crate::shaders::to_bytes;
use crate::shaders::Vertex;
use crate::DriverState;
use crate::RenderLambda;
use crate::SourceID;
use derive_where::derive_where;
use std::rc::Rc;
use ultraviolet::Vec4;
use wgpu::util::DeviceExt;

// This draws a line between two points that were previously stored in a Cross-reference Domain
#[derive_where(Clone)]
pub struct DomainLine<T: base::Empty + 'static> {
    pub id: Rc<SourceID>,
    pub domain: Rc<CrossReferenceDomain>,
    pub start: Rc<SourceID>,
    pub end: Rc<SourceID>,
    pub props: Rc<T>,
    pub fill: Vec4,
}

impl<T: base::Empty + 'static> super::Outline<T> for DomainLine<T>
where
    for<'a> &'a T: Into<&'a (dyn base::Empty + 'static)>,
{
    fn id(&self) -> std::rc::Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, _: &mut crate::StateManager) -> eyre::Result<()> {
        Ok(())
    }

    fn layout(
        &self,
        _: &crate::StateManager,
        driver: &DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let shader_idx = driver.shader_cache.borrow_mut().register_shader(
            &driver.device,
            "Line FS",
            include_str!("../shaders/Line.frag.wgsl"),
        );
        let pipeline =
            driver
                .shader_cache
                .borrow_mut()
                .line_pipeline(&driver.device, shader_idx, config);

        let mvp = gen_uniform(
            driver,
            "MVP",
            crate::shaders::mat4_proj(
                0.0,
                config.height as f32,
                config.width as f32,
                -(config.height as f32),
                0.2,
                10000.0,
            )
            .as_byte_slice(),
        );

        let posdim = driver
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PosDim"),
                contents: Vec4::new(0.0, 0.0, 1.0, 1.0).as_byte_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let fill = gen_uniform(driver, "Fill", self.fill.as_byte_slice());
        let buffers = [mvp, posdim, fill];
        let bindings: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, x)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: x.as_entire_binding(),
            })
            .collect();

        let bind_group = driver.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bindings,
            label: None,
        });

        Box::new(layout::Node::<T, dyn base::Empty> {
            props: self.props.clone(),
            children: Default::default(),
            id: Rc::downgrade(&self.id),
            renderable: Some(Rc::new_cyclic(|this| LinePipeline {
                this: this.clone(),
                pipeline,
                group: bind_group,
                buffers,
                domain: self.domain.clone(),
                start: self.start.clone(),
                end: self.end.clone(),
            })),
        })
    }
}

crate::gen_outline_wrap!(DomainLine, base::Empty);

pub struct LinePipeline {
    pub this: std::rc::Weak<LinePipeline>,
    pub pipeline: wgpu::RenderPipeline,
    pub group: wgpu::BindGroup,
    pub domain: Rc<CrossReferenceDomain>,
    pub start: Rc<SourceID>,
    pub end: Rc<SourceID>,
    pub buffers: [wgpu::Buffer; 3],
}

impl Renderable for LinePipeline {
    fn render(
        &self,
        area: crate::AbsRect,
        driver: &DriverState,
    ) -> im::Vector<crate::RenderInstruction> {
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
                            pos: ((start.topleft + start.bottomright) * 0.5 + area.topleft).into(),
                        },
                        Vertex {
                            pos: ((end.topleft + end.bottomright) * 0.5 + area.topleft).into(),
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
