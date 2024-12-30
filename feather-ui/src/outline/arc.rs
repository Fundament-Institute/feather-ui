// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2024 Fundament Software SPC <https://fundament.software>

use crate::layout;
use crate::layout::empty::Empty;
use crate::layout::Layout;
use crate::shaders::gen_uniform;
use crate::DriverState;
use crate::SourceID;
use crate::URect;
use std::borrow::Cow;
use std::rc::Rc;
use ultraviolet::Vec4;
use wgpu::util::DeviceExt;

#[derive(Clone, Default)]
pub struct Arc<Parent: Clone> {
    pub id: Rc<SourceID>,
    pub props: Parent,
    pub rect: URect,
    pub border: f32,
    pub blur: f32,
    pub arcs: Vec4,
    pub fill: Vec4,
    pub outline: Vec4,
}

impl<Parent: Clone + 'static> super::Outline<Parent> for Arc<Parent> {
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
    ) -> Box<dyn Layout<Parent>> {
        let round_rect = driver
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Arc FS"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/Arc.wgsl"
                ))),
            });

        let pipeline = crate::shaders::standard_pipeline(&driver.device, round_rect, config);

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
                contents: Vec4::one().as_byte_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let dimborderblur = gen_uniform(
            driver,
            "DimBorderBlur",
            Vec4::new(0.0, 0.0, self.border, self.blur).as_byte_slice(),
        );
        let arcs = gen_uniform(driver, "Arcs", self.arcs.as_byte_slice());
        let fill = gen_uniform(driver, "Fill", self.fill.as_byte_slice());
        let outline = gen_uniform(driver, "Outline", self.outline.as_byte_slice());
        let buffers = [mvp, posdim, dimborderblur, arcs, fill, outline];
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

        Box::new(layout::Node::<Empty, Parent> {
            props: self.rect,
            imposed: self.props.clone(),
            children: Default::default(),
            id: Rc::downgrade(&self.id),
            renderable: Some(Rc::new_cyclic(|this| crate::shaders::StandardPipeline {
                this: this.clone(),
                pipeline,
                group: bind_group,
                vertices: crate::shaders::default_vertex_buffer(driver),
                buffers,
            })),
        })
    }
}
