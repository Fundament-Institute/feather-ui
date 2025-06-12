// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::AbsRect;
use crate::graphics;
use crate::render::compositor::Compositor;
use std::any::Any;
use std::any::TypeId;
use std::collections::HashMap;
use std::rc::Rc;

pub mod atlas;
pub mod compositor;
pub mod domain;
pub mod line;
pub mod shape;
pub mod text;

pub trait Renderable {
    fn render(&self, area: AbsRect, graphics: &crate::graphics::State, compositor: &mut Compositor);
}

pub trait Pipeline {
    type Data: 'static;

    fn append(&mut self, data: &Self::Data, layer: u16);

    #[allow(unused_variables)]
    fn prepare(
        &mut self,
        graphics: &graphics::State,
        encoder: &mut wgpu::CommandEncoder,
        config: &wgpu::SurfaceConfiguration,
    ) {
    }
    fn draw(&mut self, graphics: &graphics::State, pass: &mut wgpu::RenderPass<'_>, layer: u16);
}

pub trait AnyPipeline: Any + std::fmt::Debug + Send + Sync {
    fn append(&mut self, data: &dyn Any, layer: u16);
}

impl<T: Pipeline + std::fmt::Debug + Send + Sync + 'static> AnyPipeline for T {
    fn append(&mut self, data: &dyn Any, layer: u16) {
        Pipeline::append(self, data.downcast_ref().unwrap(), layer)
    }
}

#[derive(Default)]
pub struct PipelineCache {
    pipelines: HashMap<crate::graphics::PipelineID, Box<dyn AnyPipeline>>,
}

impl PipelineCache {
    pub fn get<T: crate::render::Pipeline + 'static>(&self) -> Option<&T> {
        self.pipelines
            .get(&TypeId::of::<T>())
            .and_then(|x| (x.as_ref() as &dyn std::any::Any).downcast_ref())
    }

    pub fn get_mut<T: crate::render::Pipeline + 'static>(&mut self) -> Option<&mut T> {
        self.pipelines
            .get_mut(&TypeId::of::<T>())
            .and_then(|x| (x.as_mut() as &mut dyn std::any::Any).downcast_mut())
    }

    pub(crate) fn insert<T: crate::render::Pipeline + 'static>(
        &mut self,
        pipeline: Box<dyn AnyPipeline>,
    ) {
        self.pipelines.insert(TypeId::of::<T>(), pipeline);
    }
}

pub struct Chain<const N: usize>(pub [Rc<dyn Renderable>; N]);

impl<const N: usize> Renderable for Chain<N> {
    fn render(
        &self,
        area: crate::AbsRect,
        graphics: &crate::graphics::State,
        compositor: &mut Compositor,
    ) {
        for x in &self.0 {
            x.render(area, graphics, compositor)
        }
    }
}
