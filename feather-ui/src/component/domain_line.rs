// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Layout, base};
use crate::shaders::gen_uniform;
use crate::{CrossReferenceDomain, DriverState, SourceID, layout, render};
use derive_where::derive_where;
use std::rc::Rc;
use ultraviolet::Vec4;
use wgpu::util::DeviceExt;

// This draws a line between two points that were previously stored in a Cross-reference Domain
#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct DomainLine<T: base::Empty + 'static> {
    pub id: Rc<SourceID>,
    pub domain: Rc<CrossReferenceDomain>,
    pub start: Rc<SourceID>,
    pub end: Rc<SourceID>,
    pub props: Rc<T>,
    pub fill: Vec4,
}

impl<T: base::Empty + 'static> super::Component<T> for DomainLine<T>
where
    for<'a> &'a T: Into<&'a (dyn base::Empty + 'static)>,
{
    fn layout(
        &self,
        _: &crate::StateManager,
        driver: &DriverState,
        _window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let shader_idx = driver.shader_cache.write().register_shader(
            &driver.device,
            "Line FS",
            include_str!("../shaders/Line.frag.wgsl"),
        );
        let pipeline =
            driver
                .shader_cache
                .write()
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
            renderable: Some(Rc::new_cyclic(|this| render::domain::line::Pipeline {
                this: this.clone(),
                pipeline,
                group: bind_group,
                _buffers: buffers,
                domain: self.domain.clone(),
                start: self.start.clone(),
                end: self.end.clone(),
            })),
        })
    }
}

crate::gen_component_wrap!(DomainLine, base::Empty);
