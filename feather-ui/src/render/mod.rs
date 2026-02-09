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
pub mod shape;
pub mod text;
//pub mod textbox;

/// Used instead of a direct Fn() trait because closure types are unique, which causes problems.
pub trait Prerender {
    type R: Renderable;

    fn prerender(&self, area: crate::DynSignal<PxRect>) -> Self::R;
}

pub trait Renderable {
    fn render(
        &self,
        parent_pos: crate::PxPoint,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), crate::RenderError>;
}

// This implementation is used for components that never actually pass in a Renderable, so prerender never gets called.
impl Prerender for () {
    type R = ();

    fn prerender(&self, _: crate::DynSignal<PxRect>) -> () {
        ()
    }
}

impl Renderable for () {
    fn render(
        &self,
        _: crate::PxPoint,
        _: &crate::graphics::Driver,
        _: &mut CompositorView<'_>,
    ) -> Result<(), crate::RenderError> {
        debug_assert!(false);
        Err(crate::RenderError::InternalFailure)
    }
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

#[repr(transparent)]
pub struct Chain<const N: usize>(pub [Rc<dyn Renderable>; N]);

impl<const N: usize> Renderable for Chain<N> {
    fn render(
        &self,
        parent_pos: crate::PxPoint,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), crate::RenderError> {
        for x in &self.0 {
            x.render(parent_pos, driver, compositor)?;
        }
        Ok(())
    }
}
