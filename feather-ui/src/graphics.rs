// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::collections::HashMap;
use std::rc::Rc;

use crate::render;
use crate::render::atlas;
use crate::render::compositor;
use parking_lot::RwLock;
use std::any::TypeId;
use std::sync::Arc;
use ultraviolet::Vec2;
use wgpu::{PipelineLayout, RenderPipeline, RenderPipelineDescriptor, ShaderModule};
use winit::window::CursorIcon;

// Points are specified as 72 per inch, and a scale factor of 1.0 corresponds to 96 DPI, so we multiply by the
// ratio times the scaling factor.
#[inline]
pub fn point_to_pixel(pt: f32, scale_factor: f32) -> f32 {
    pt * (72.0 / 96.0) * scale_factor // * text_scale_factor
}

#[inline]
pub fn pixel_to_vec(p: winit::dpi::PhysicalPosition<f32>) -> Vec2 {
    Vec2::new(p.x, p.y)
}

pub type PipelineID = TypeId;

struct PipelineState {
    layout: PipelineLayout,
    shader: ShaderModule,
    generator: Box<
        dyn Fn(&PipelineLayout, &ShaderModule, &wgpu::Device) -> Box<dyn render::AnyPipeline>
            + Send
            + Sync,
    >,
}

pub(crate) type GlyphCache = HashMap<cosmic_text::CacheKey, atlas::Region>;

// We want to share our device/adapter state across windows, but can't create it until we have at least one window,
// so we store a weak reference to it in App and if all windows are dropped it'll also drop these, which is usually
// sensible behavior.
#[derive(Debug)]
pub struct State {
    pub(crate) adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) swash_cache: RwLock<glyphon::SwashCache>,
    pub(crate) font_system: RwLock<glyphon::FontSystem>,
    pub(crate) cursor: RwLock<CursorIcon>, // This is a convenient place to track our global expected cursor
    pub(crate) pipelines: RwLock<HashMap<PipelineID, Box<dyn crate::render::AnyPipeline>>>,
    pub(crate) glyphs: RwLock<GlyphCache>,
    registry: HashMap<PipelineID, PipelineState>,
    pub(crate) compositor: Arc<compositor::Shared>,
    pub(crate) atlas: RwLock<atlas::Atlas>,
}

impl std::fmt::Debug for PipelineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PipelineState")
            .field(&self.layout)
            .field(&self.shader)
            .finish()
    }
}

impl State {
    pub async fn new(
        weak: &mut alloc::sync::Weak<Self>,
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'static>,
    ) -> eyre::Result<Arc<Self>> {
        if let Some(driver) = weak.upgrade() {
            return Ok(driver);
        }

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(surface),
                ..Default::default()
            })
            .await?;

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Feather UI wgpu Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await?;

        let compositor = compositor::Shared::new(&device, 512).into();
        let atlas = atlas::Atlas::new(&device, 512).into();

        let driver = Arc::new(Self {
            adapter,
            device,
            queue,
            swash_cache: glyphon::SwashCache::new().into(),
            font_system: glyphon::FontSystem::new().into(),
            cursor: RwLock::new(CursorIcon::Default),
            pipelines: HashMap::new().into(),
            glyphs: HashMap::new().into(),
            registry: HashMap::new(),
            compositor,
            atlas,
        });

        *weak = Arc::downgrade(&driver);
        Ok(driver)
    }

    pub fn register_pipeline<T: 'static>(
        &mut self,
        layout: PipelineLayout,
        shader: ShaderModule,
        generator: impl Fn(
            &PipelineLayout,
            &ShaderModule,
            &wgpu::Device,
        ) -> Box<dyn render::AnyPipeline>
        + Send
        + Sync
        + 'static,
    ) {
        self.registry.insert(
            TypeId::of::<T>(),
            PipelineState {
                layout,
                shader,
                generator: Box::new(generator),
            },
        );
    }

    pub fn get_pipeline<'a, T: crate::render::Pipeline + 'static>(&'a self) -> &'a T {
        let id = TypeId::of::<T>();

        // We can't use the result of this because it makes the lifetimes weird
        if self.pipelines.read().get(&id).is_none() {
            let PipelineState {
                generator,
                layout,
                shader,
            } = &self.registry[&id];

            self.pipelines
                .write()
                .insert(id, generator(&layout, &shader, &self.device));
        }
        (self.pipelines.read().get(&id).unwrap().as_ref() as &dyn std::any::Any)
            .downcast_ref()
            .unwrap()
    }

    pub fn mut_pipeline<'a, T: crate::render::Pipeline + 'static>(&self) -> &'a mut T {
        let id = TypeId::of::<T>();

        // We can't use the result of this because it makes the lifetimes weird
        if self.pipelines.read().get(&id).is_none() {
            let PipelineState {
                generator,
                layout,
                shader,
            } = &self.registry[&id];

            self.pipelines
                .write()
                .insert(id, generator(&layout, &shader, &self.device));
        }
        (self.pipelines.write().get(&id).unwrap().as_mut() as &dyn std::any::Any)
            .downcast_mut()
            .unwrap()
    }
}

static_assertions::assert_impl_all!(State: Send, Sync);
