// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::AbsRect;
use crate::graphics;
use crate::render::compositor::Compositor;
use std::any::Any;
use std::rc::Rc;

pub mod atlas;
pub mod compositor;
pub mod domain;
pub mod line;
pub mod shape;
pub mod text;

pub trait Renderable {
    #[must_use]
    fn render(
        &self,
        area: AbsRect,
        driver: &crate::graphics::Driver,
        compositor: &mut Compositor,
    ) -> Result<(), crate::Error>;
}

pub trait Pipeline: Any + std::fmt::Debug + Send + Sync {
    type Data: 'static;

    fn append(&mut self, data: &Self::Data, layer: u16);

    #[allow(unused_variables)]
    fn prepare(
        &mut self,
        driver: &graphics::Driver,
        encoder: &mut wgpu::CommandEncoder,
        config: &wgpu::SurfaceConfiguration,
    ) {
    }
    fn draw(&mut self, driver: &graphics::Driver, pass: &mut wgpu::RenderPass<'_>, layer: u16);
}

pub trait AnyPipeline: Any + std::fmt::Debug + Send + Sync {
    fn append(&mut self, data: &dyn Any, layer: u16);
    fn prepare(
        &mut self,
        driver: &graphics::Driver,
        encoder: &mut wgpu::CommandEncoder,
        config: &wgpu::SurfaceConfiguration,
    );
    fn draw(&mut self, driver: &graphics::Driver, pass: &mut wgpu::RenderPass<'_>, layer: u16);
}

impl<T: Pipeline + std::fmt::Debug + Send + Sync + 'static> AnyPipeline for T {
    fn append(&mut self, data: &dyn Any, layer: u16) {
        Pipeline::append(self, data.downcast_ref().unwrap(), layer)
    }
    fn prepare(
        &mut self,
        driver: &graphics::Driver,
        encoder: &mut wgpu::CommandEncoder,
        config: &wgpu::SurfaceConfiguration,
    ) {
        Pipeline::prepare(self, driver, encoder, config);
    }
    fn draw(&mut self, driver: &graphics::Driver, pass: &mut wgpu::RenderPass<'_>, layer: u16) {
        Pipeline::draw(self, driver, pass, layer);
    }
}

pub struct Chain<const N: usize>(pub [Rc<dyn Renderable>; N]);

impl<const N: usize> Renderable for Chain<N> {
    fn render(
        &self,
        area: crate::AbsRect,
        driver: &crate::graphics::Driver,
        compositor: &mut Compositor,
    ) -> Result<(), crate::Error> {
        for x in &self.0 {
            x.render(area, driver, compositor)?;
        }
        Ok(())
    }
}
