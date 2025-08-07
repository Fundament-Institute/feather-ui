// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::collections::HashMap;
use std::io::Read;

use crate::render::atlas::ATLAS_FORMAT;
use crate::render::{atlas, compositor};
use crate::resource::{ResourceLoader, ResourceLocation};
use crate::{Error, render};
use guillotiere::AllocId;
use parking_lot::RwLock;
use std::any::TypeId;
use std::sync::Arc;
use swash::scale::ScaleContext;
use ultraviolet::{Mat4, Vec2, Vec4};
use wgpu::{PipelineLayout, ShaderModule};
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

#[derive_where::derive_where(Debug)]
#[allow(clippy::type_complexity)]
pub(crate) struct PipelineState {
    layout: PipelineLayout,
    shader: ShaderModule,
    #[derive_where(skip)]
    generator: Box<
        dyn Fn(&PipelineLayout, &ShaderModule, &Driver) -> Box<dyn render::AnyPipeline>
            + Send
            + Sync,
    >,
}

#[derive(Debug)]
pub struct GlyphRegion {
    pub offset: [i32; 2],
    pub region: atlas::Region,
}

pub(crate) type GlyphCache = HashMap<cosmic_text::CacheKey, GlyphRegion>;

/// Represents a particular realized instance of a resource on the GPU. This includes the target size, DPI, and
/// whether mipmaps have been generated.
#[derive(Debug)]
pub struct ResourceInstance<'a> {
    location: Result<Box<dyn ResourceLocation>, &'a dyn ResourceLocation>,
    /// If finite, this is used for vector resources, which must care about DPI beyond simply changing their size.
    dpi: f32,
    /// If true, mipmaps should be generated for this resource because the user expects to resize it in realtime.
    resizable: bool,
}

impl std::hash::Hash for ResourceInstance<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match &self.location {
            Ok(l) => l.hash(state),
            Err(l) => l.hash(state),
        }
        // This works because DPI is always non-zero
        f32::to_bits(self.dpi).hash(state);
        self.resizable.hash(state);
    }
}

impl PartialEq for ResourceInstance<'_> {
    fn eq(&self, other: &Self) -> bool {
        let l = match &self.location {
            Ok(l) => l.as_ref(),
            Err(l) => *l,
        };

        let r = match &other.location {
            Ok(l) => l.as_ref(),
            Err(l) => *l,
        };

        *l == *r && self.dpi == other.dpi && self.resizable == other.resizable
    }
}

// We don't put NaNs in our DPI float so this is fine.
impl Eq for ResourceInstance<'_> {}

// We want to share our device/adapter state across windows, but can't create it until we have at least one window,
// so we store a weak reference to it in App and if all windows are dropped it'll also drop these, which is usually
// sensible behavior.
#[derive_where::derive_where(Debug)]
pub struct Driver {
    pub(crate) glyphs: RwLock<GlyphCache>,
    pub(crate) resources: RwLock<HashMap<Box<dyn ResourceLocation>, Box<dyn ResourceLoader>>>, // TODO: needs to be cleaned up when we clean up mutable states
    pub(crate) loaders:
        RwLock<HashMap<ResourceInstance<'static>, smallvec::SmallVec<[atlas::Region; 1]>>>,
    pub(crate) atlas: RwLock<atlas::Atlas>,
    pub(crate) layer_atlas: [RwLock<atlas::Atlas>; 2],
    pub(crate) layer_composite: [RwLock<compositor::Compositor>; 2],
    pub(crate) shared: compositor::Shared,
    pub(crate) pipelines: RwLock<HashMap<PipelineID, Box<dyn crate::render::AnyPipeline>>>,
    pub(crate) registry: RwLock<HashMap<PipelineID, PipelineState>>,
    pub(crate) queue: wgpu::Queue,
    pub(crate) device: wgpu::Device,
    pub(crate) adapter: wgpu::Adapter,
    pub(crate) cursor: RwLock<CursorIcon>, // This is a convenient place to track our global expected cursor
    #[derive_where(skip)]
    pub(crate) swash_cache: RwLock<ScaleContext>,
    pub(crate) font_system: RwLock<cosmic_text::FontSystem>,
}

impl Drop for Driver {
    fn drop(&mut self) {
        for (_, mut r) in self.glyphs.get_mut().drain() {
            r.region.id = AllocId::deserialize(u32::MAX);
        }
    }
}

impl Driver {
    pub async fn new(
        weak: &mut std::sync::Weak<Self>,
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'static>,
        on_driver: &mut Option<Box<dyn FnOnce(std::sync::Weak<Driver>) + 'static>>,
    ) -> eyre::Result<Arc<Self>> {
        if let Some(driver) = weak.upgrade() {
            return Ok(driver);
        }

        let adapter = futures_lite::future::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                compatible_surface: Some(surface),
                ..Default::default()
            },
        ))?;

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

        let shared = compositor::Shared::new(&device);
        let atlas = atlas::Atlas::new(&device, 512, atlas::AtlasKind::Primary);
        let layer0 = atlas::Atlas::new(&device, 128, atlas::AtlasKind::Layer0);
        let layer1 = atlas::Atlas::new(&device, 128, atlas::AtlasKind::Layer1);
        let shape_shader = crate::render::shape::Shape::<0>::shader(&device);
        let shape_pipeline = crate::render::shape::Shape::<0>::layout(&device);

        let comp1 = crate::render::compositor::Compositor::new(
            &device,
            &shared,
            &atlas.view,
            &layer1.view,
            ATLAS_FORMAT,
            true,
        );
        let comp2 = crate::render::compositor::Compositor::new(
            &device,
            &shared,
            &atlas.view,
            &layer0.view,
            ATLAS_FORMAT,
            false,
        );

        let mut driver = Self {
            adapter,
            device,
            queue,
            swash_cache: ScaleContext::new().into(),
            resources: HashMap::new().into(),
            loaders: HashMap::new().into(),
            font_system: cosmic_text::FontSystem::new().into(),
            cursor: CursorIcon::Default.into(),
            pipelines: HashMap::new().into(),
            glyphs: HashMap::new().into(),
            registry: HashMap::new().into(),
            shared,
            atlas: atlas.into(),
            layer_atlas: [layer0.into(), layer1.into()],
            layer_composite: [comp1.into(), comp2.into()],
        };

        driver.register_pipeline::<crate::render::shape::Shape<0>>(
            shape_pipeline.clone(),
            shape_shader.clone(),
            crate::render::shape::Shape::<0>::create,
        );
        driver.register_pipeline::<crate::render::shape::Shape<1>>(
            shape_pipeline.clone(),
            shape_shader.clone(),
            crate::render::shape::Shape::<1>::create,
        );
        driver.register_pipeline::<crate::render::shape::Shape<2>>(
            shape_pipeline.clone(),
            shape_shader.clone(),
            crate::render::shape::Shape::<2>::create,
        );
        driver.register_pipeline::<crate::render::shape::Shape<3>>(
            shape_pipeline.clone(),
            shape_shader.clone(),
            crate::render::shape::Shape::<3>::create,
        );

        let driver = Arc::new(driver);
        *weak = Arc::downgrade(&driver);

        if let Some(f) = on_driver.take() {
            f(weak.clone());
        }
        Ok(driver)
    }

    pub fn register_pipeline<T: 'static>(
        &mut self,
        layout: PipelineLayout,
        shader: ShaderModule,
        generator: impl Fn(&PipelineLayout, &ShaderModule, &Self) -> Box<dyn render::AnyPipeline>
        + Send
        + Sync
        + 'static,
    ) {
        self.registry.write().insert(
            TypeId::of::<T>(),
            PipelineState {
                layout,
                shader,
                generator: Box::new(generator),
            },
        );
    }

    /// Allows replacing the shader in a pipeline, for hot-reloading.
    pub fn reload_pipeline<T: 'static>(&self, shader: ShaderModule) {
        let id = TypeId::of::<T>();
        let mut registry = self.registry.write();
        let pipeline = registry
            .get_mut(&id)
            .expect("Tried to reload unregistered pipeline!");
        pipeline.shader = shader;
        self.pipelines.write().remove(&id);
    }
    pub fn with_pipeline<T: crate::render::Pipeline + 'static>(&self, f: impl FnOnce(&mut T)) {
        let id = TypeId::of::<T>();

        // We can't use the result of this because it makes the lifetimes weird
        if self.pipelines.read().get(&id).is_none() {
            let PipelineState {
                generator,
                layout,
                shader,
            } = &self.registry.read()[&id];

            self.pipelines
                .write()
                .insert(id, generator(layout, shader, self));
        }

        f(
            (self.pipelines.write().get_mut(&id).unwrap().as_mut() as &mut dyn std::any::Any)
                .downcast_mut()
                .unwrap(),
        );
    }

    pub fn prefetch(&self, location: &dyn ResourceLocation) -> Result<(), Error> {
        let mut resources = self.resources.write();

        if !resources.contains_key(location) {
            resources.insert(dyn_clone::clone_box(location), location.fetch()?);
        }
        Ok(())
    }

    pub fn load(
        &self,
        location: &dyn ResourceLocation,
        size: &guillotiere::Size,
        dpi: f32,
        resize: bool,
        mut f: impl FnMut(&atlas::Region) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if let Some(regions) = self.loaders.read().get(&ResourceInstance {
            location: Err(location),
            dpi,
            resizable: resize,
        }) {
            // Check if our requested size is within reasonable resize range - slightly bigger or smaller is fine, and
            // much smaller is fine if we have access to mipmaps.
            for r in regions {
                if r.uv.size() == *size {
                    return f(r);
                } else if r.uv.area() >= crate::resource::MIN_AREA
                    && crate::resource::within_variance(
                        size.width,
                        r.uv.width(),
                        crate::resource::MAX_VARIANCE,
                    )
                    && crate::resource::within_variance(
                        size.width,
                        r.uv.width(),
                        crate::resource::MAX_VARIANCE,
                    )
                {
                    return f(r);
                } else if resize && size.width <= r.uv.width() && size.height < r.uv.height() {
                    return f(r);
                }
            }
        }

        // Check for a prefetched resource
        let region = {
            let loader;
            let reader = self.resources.read();
            if let Some(res) = reader.get(location) {
                res
            } else {
                loader = location.fetch()?;
                &loader
            }
            .load(self, size, dpi, resize)?
        };

        f(&self
            .loaders
            .write()
            .entry(ResourceInstance {
                location: Ok(dyn_clone::clone_box(location)),
                dpi,
                resizable: resize,
            })
            .insert_entry(smallvec::SmallVec::from_buf([region]))
            .get()[0])
    }
}

static_assertions::assert_impl_all!(Driver: Send, Sync);

// This maps x and y to the viewpoint size, maps input_z from [n,f] to [0,1], and sets
// output_w = input_z for perspective. Requires input_w = 1
pub fn mat4_proj(x: f32, y: f32, w: f32, h: f32, n: f32, f: f32) -> Mat4 {
    Mat4 {
        cols: [
            Vec4::new(2.0 / w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0 / (f - n), 1.0),
            Vec4::new(-(2.0 * x + w) / w, -(2.0 * y + h) / h, -n / (f - n), 0.0),
        ],
    }
}

// Orthographic projection matrix
pub fn mat4_ortho(x: f32, y: f32, w: f32, h: f32, n: f32, f: f32) -> Mat4 {
    Mat4 {
        cols: [
            Vec4::new(2.0 / w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -2.0 / (f - n), 0.0),
            Vec4::new(
                -(2.0 * x + w) / w,
                -(2.0 * y + h) / h,
                (f + n) / (f - n),
                1.0,
            ),
        ],
    }
}

macro_rules! gen_from_array {
    ($s:path, $t:path, $i:literal) => {
        impl From<[$t; $i]> for $s {
            fn from(value: [$t; $i]) -> Self {
                Self(value)
            }
        }
        impl From<&[$t; $i]> for $s {
            fn from(value: &[$t; $i]) -> Self {
                Self(*value)
            }
        }
    };
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::NoUninit)]
pub struct Vec2f(pub(crate) [f32; 2]);

gen_from_array!(Vec2f, f32, 2);

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::NoUninit)]
pub struct Vec4f(pub(crate) [f32; 4]);

gen_from_array!(Vec4f, f32, 4);

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::NoUninit)]
pub struct Vec2i(pub(crate) [i32; 2]);

gen_from_array!(Vec2i, i32, 2);

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::NoUninit)]
pub struct Vec4i(pub(crate) [i32; 4]);

gen_from_array!(Vec4i, i32, 4);
