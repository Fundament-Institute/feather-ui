// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::render::compositor::CompositorView;
use crate::{PxRect, graphics};
use std::any::Any;
use std::rc::Rc;

pub mod atlas;
pub mod compositor;
pub mod domain;
//pub mod image;
pub mod line;
//pub mod shape;
pub mod text;
pub mod textbox;

pub trait Renderable {
    fn render(
        &self,
        area: PxRect,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), crate::Error>;
}

pub trait Pipeline: Any + std::fmt::Debug + Send + Sync {
    #[allow(unused_variables)]
    fn prepare(
        &mut self,
        driver: &graphics::Driver,
        encoder: &mut wgpu::CommandEncoder,
        config: &wgpu::SurfaceConfiguration,
    ) {
    }
    fn draw(&mut self, driver: &graphics::Driver, pass: &mut wgpu::RenderPass<'_>, layer: u8);
    #[allow(unused_variables)]
    fn destroy(&mut self, driver: &graphics::Driver) {}
}

pub struct Chain<const N: usize>(pub [Rc<dyn Renderable>; N]);

impl<const N: usize> Renderable for Chain<N> {
    fn render(
        &self,
        area: PxRect,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), crate::Error> {
        for x in &self.0 {
            x.render(area, driver, compositor)?;
        }
        Ok(())
    }
}
