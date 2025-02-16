// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

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
pub struct ShaderStandard<Parent: Clone> {
    pub id: std::rc::Rc<SourceID>,
    pub props: Parent,
    pub rect: URect,
    pub uniforms: [Vec4; 4],
    pub fragment: String,
}

impl<Parent: Clone + 'static> super::Outline<Parent> for ShaderStandard<Parent> {
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
                label: Some("ShaderStandard FS"),
                source: wgpu::ShaderSource::Wgsl(self.fragment.clone().into()),
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
                contents: Vec4::new(0.0, 0.0, 400.0, 200.0).as_byte_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let dimborderblur = gen_uniform(driver, "DimBorderBlur", self.uniforms[0].as_byte_slice());
        let corners = gen_uniform(driver, "Corners", self.uniforms[1].as_byte_slice());
        let fill = gen_uniform(driver, "Fill", self.uniforms[2].as_byte_slice());
        let outline = gen_uniform(driver, "Outline", self.uniforms[3].as_byte_slice());
        let buffers = [mvp, posdim, dimborderblur, corners, fill, outline];
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
