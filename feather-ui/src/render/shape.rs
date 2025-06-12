// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::compositor;
use crate::render::Pipeline;
use crate::render::compositor::Compositor;
use crate::{graphics, vec4_to_u32};
use guillotiere::Size;
use std::collections::HashMap;
use ultraviolet::Vec4;

pub struct Shape {
    data: HashMap<u16, Vec<Data>>,
    buffer: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
}

pub struct Instance {
    pub padding: crate::AbsRect,
    pub border: f32,
    pub blur: f32,
    pub fill: Vec4,
    pub outline: Vec4,
    pub corners: Vec4,
    pub id: std::rc::Rc<crate::SourceID>,
}

impl super::Renderable for Instance {
    fn render(
        &self,
        area: crate::AbsRect,
        graphics: &crate::graphics::State,
        compositor: &mut Compositor,
    ) {
        let pipeline = graphics.mut_pipeline::<Shape>();
        let dim = *(area.bottomright() - area.topleft() - self.padding.bottomright()).as_array();
        let mut atlas = graphics.atlas.write();
        let region = atlas.cache_region(
            &graphics.device,
            self.id.clone(),
            Size::new(area.dim().0.x.ceil() as i32, area.dim().0.y.ceil() as i32),
        );

        pipeline.append(
            &Data {
                pos: *(area.topleft() + self.padding.topleft()).as_array(),
                dim: dim,
                border: self.border,
                blur: self.blur,
                corners: *self.corners.as_array(),
                fill: vec4_to_u32(&self.fill),
                outline: vec4_to_u32(&self.outline),
            },
            region.index,
        );

        compositor.append(&compositor::Data {
            pos: *(area.topleft() + self.padding.topleft()).as_array(),
            dim: dim,
            uv: region.uv.min.to_array(),
            uvdim: region.uv.size().to_array(),
            color: 0xFFFFFFFF,
            rotation: 0.0,
            tex: region.index,
            clip: 0,
        });
    }
}

#[derive(Clone, Copy, Default, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct Data {
    pub pos: [f32; 2],
    pub dim: [f32; 2],
    pub border: f32,
    pub blur: f32,
    pub corners: [f32; 4],
    pub fill: u32,
    pub outline: u32,
}

impl super::Pipeline for Shape {
    type Data = Data;

    fn append(&mut self, data: &Self::Data, layer: u16) {
        self.data.entry(layer).or_insert_with(Vec::new).push(*data);
    }

    fn draw(&mut self, graphics: &graphics::State, pass: &mut wgpu::RenderPass<'_>, layer: u16) {
        if let Some(data) = self.data.get_mut(&layer) {
            graphics
                .queue
                .write_buffer(&self.buffer, 0, bytemuck::cast_slice(data.as_slice()));

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[0]);
            pass.draw(0..(data.len() as u32 * 6), 0..1);
            data.clear();
        }
    }
}
