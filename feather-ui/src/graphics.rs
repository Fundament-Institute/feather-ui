// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::collections::HashMap;

use crate::render::atlas::{ATLAS_FORMAT, Atlas, AtlasKind};
use crate::render::compositor::Compositor;
use crate::render::shape::Shape;
use crate::render::{atlas, compositor};
use crate::resource::{Loader, Location, MAX_VARIANCE};
use crate::{Error, render};
use guillotiere::AllocId;
use parking_lot::RwLock;
use smallvec::SmallVec;
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
    location: Result<Box<dyn Location>, &'a dyn Location>,
    /// If finite, this is used for vector resources, which must care about DPI beyond simply changing their size.
    dpi: f32,
    /// If true, mipmaps should be generated for this resource because the user expects to resize it in realtime.
    resizable: bool,
}

impl Clone for ResourceInstance<'static> {
    fn clone(&self) -> Self {
        Self {
            location: self.location.clone(),
            dpi: self.dpi,
            resizable: self.resizable,
        }
    }
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
#[allow(clippy::type_complexity)]
#[derive_where::derive_where(Debug)]
pub struct Driver {
    pub(crate) glyphs: RwLock<GlyphCache>,
    pub(crate) prefetch: RwLock<HashMap<Box<dyn Location>, Box<dyn Loader>>>,
    pub(crate) resources: RwLock<
        HashMap<ResourceInstance<'static>, (SmallVec<[atlas::Region; 1]>, guillotiere::Size)>,
    >,
    pub(crate) locations:
        RwLock<HashMap<Box<dyn Location>, SmallVec<[ResourceInstance<'static>; 1]>>>,
    pub(crate) atlas: RwLock<Atlas>,
    pub(crate) layer_atlas: [RwLock<Atlas>; 2],
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

        for (_, (regions, _)) in self.resources.get_mut().drain() {
            for mut region in regions {
                region.id = AllocId::deserialize(u32::MAX);
            }
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
        let atlas = Atlas::new(&device, 512, AtlasKind::Primary);
        let layer0 = Atlas::new(&device, 128, AtlasKind::Layer0);
        let layer1 = Atlas::new(&device, 128, AtlasKind::Layer1);
        let shape_shader = Shape::<0>::shader(&device);
        let shape_pipeline = Shape::<0>::layout(&device);

        let comp1 = Compositor::new(
            &device,
            &shared,
            &atlas.view,
            &layer1.view,
            ATLAS_FORMAT,
            true,
        );
        let comp2 = Compositor::new(
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
            prefetch: HashMap::new().into(),
            resources: HashMap::new().into(),
            locations: HashMap::new().into(),
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

        driver.register_pipeline::<Shape<0>>(
            shape_pipeline.clone(),
            shape_shader.clone(),
            Shape::<0>::create,
        );
        driver.register_pipeline::<Shape<1>>(
            shape_pipeline.clone(),
            shape_shader.clone(),
            Shape::<1>::create,
        );
        driver.register_pipeline::<Shape<2>>(
            shape_pipeline.clone(),
            shape_shader.clone(),
            Shape::<2>::create,
        );
        driver.register_pipeline::<Shape<3>>(
            shape_pipeline.clone(),
            shape_shader.clone(),
            Shape::<3>::create,
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

    pub fn prefetch(&self, location: &dyn Location) -> Result<(), Error> {
        let mut resources = self.prefetch.write();

        if !resources.contains_key(location) {
            resources.insert(dyn_clone::clone_box(location), location.fetch()?);
        }
        Ok(())
    }

    /// This function is called during layout, outside of a render pass, which allows the texture atlas to be
    /// immediately resized to accomdate the new resource. As a result, it assumes you don't need the region,
    /// only the final intrinsic size.
    pub fn load_and_resize(
        &self,
        location: &dyn Location,
        size: guillotiere::Size,
        dpi: f32,
        resize: bool,
    ) -> Result<guillotiere::Size, Error> {
        let mut uvsize = guillotiere::Size::zero();
        match self.load(location, size, dpi, resize, |r| {
            uvsize = r.uv.size();
            Ok(())
        }) {
            Err(Error::ResizeTextureAtlas(layers, kind)) => {
                // Resize the texture atlas with the requested number of layers (the extent has already been changed)
                match kind {
                    AtlasKind::Primary => self.atlas.write(),
                    AtlasKind::Layer0 => self.layer_atlas[0].write(),
                    AtlasKind::Layer1 => self.layer_atlas[1].write(),
                }
                .resize(&self.device, &self.queue, layers);
                self.load_and_resize(location, size, dpi, resize) // Retry load
            }
            Err(e) => Err(e),
            Ok(_) => Ok(uvsize),
        }
    }

    pub fn load(
        &self,
        location: &dyn Location,
        mut size: guillotiere::Size,
        dpi: f32,
        resize: bool,
        mut f: impl FnMut(&atlas::Region) -> Result<(), Error>,
    ) -> Result<(), Error> {
        use crate::resource;

        if let Some((regions, native_size)) = self.resources.read().get(&ResourceInstance {
            location: Err(location),
            dpi: f32::INFINITY,
            resizable: resize,
        }) {
            size = resource::fill_size(size, *native_size);

            // Check if our requested size is within reasonable resize range - slightly bigger or smaller is fine, and
            // much smaller is fine if we have access to mipmaps.
            for r in regions {
                if r.uv.size() == size
                    || (r.uv.area() >= resource::MIN_AREA
                        && resource::within_variance(size.width, r.uv.width(), MAX_VARIANCE)
                        && resource::within_variance(size.height, r.uv.height(), MAX_VARIANCE))
                    || (resize && size.width <= r.uv.width() && size.height < r.uv.height())
                {
                    return f(r);
                }
            }
        }

        // Check for a prefetched resource
        let (region, native) = {
            let loader;
            let reader = self.prefetch.read();
            if let Some(res) = reader.get(location) {
                res
            } else {
                loader = location.fetch()?;
                &loader
            }
            .load(self, size, dpi, resize)?
        };

        let key = ResourceInstance {
            location: Ok(dyn_clone::clone_box(location)),
            dpi: f32::INFINITY,
            resizable: resize,
        };

        self.locations
            .write()
            .entry(dyn_clone::clone_box(location))
            .and_modify(|x| x.push(key.clone()))
            .or_insert(SmallVec::from_buf([key.clone()]));

        if let Some((entry, _)) = self.resources.write().get_mut(&key) {
            entry.push(region);
            f(entry.last().as_ref().ok_or(Error::InternalFailure)?)
        } else {
            f(&self
                .resources
                .write()
                .entry(key)
                .insert_entry((SmallVec::from_buf([region]), native))
                .get()
                .0[0])
        }
    }

    /// Removes all loaded instances of a particular resource location. Generally used for hotloading resources that changed on disk.
    pub fn evict(&self, location: &dyn Location) {
        if let Some(instances) = self.locations.read().get(location) {
            for instance in instances {
                if let Some((regions, _)) = self.resources.write().remove(instance) {
                    for mut region in regions {
                        self.atlas.write().destroy(&mut region);
                    }
                }
            }
        }
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
