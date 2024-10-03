pub mod component;
pub mod input;
pub mod layout;
//mod lua;
pub mod persist;
mod rtree;
mod shaders;

use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use std::sync::Weak;
use winit::window::WindowId;

use crate::component::window::Window;
use crate::input::Event;
use component::Component;
use eyre::OptionExt;
use persist::FnPersist;
use std::cell::RefCell;
use std::sync::Arc;
use ultraviolet::vec::Vec2;

impl Into<Vec2> for AbsDim {
    fn into(self) -> Vec2 {
        self.0
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct AbsDim(Vec2);

#[derive(Copy, Clone, Debug, Default)]
/// Absolutely positioned rectangle
pub struct AbsRect {
    pub topleft: Vec2,
    pub bottomright: Vec2,
}

impl AbsRect {
    pub fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            topleft: Vec2 { x: left, y: top },
            bottomright: Vec2 {
                x: right,
                y: bottom,
            },
        }
    }
    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.topleft.x
            && p.x <= self.bottomright.x
            && p.y >= self.topleft.y
            && p.y <= self.bottomright.y
    }

    pub fn extend(&self, rhs: AbsRect) -> AbsRect {
        AbsRect {
            topleft: self.topleft.min_by_component(rhs.topleft),
            bottomright: self.bottomright.min_by_component(rhs.bottomright),
        }
    }

    pub fn width(&self) -> f32 {
        self.bottomright.x - self.topleft.x
    }
    pub fn height(&self) -> f32 {
        self.bottomright.y - self.topleft.y
    }
    pub fn dim(&self) -> AbsDim {
        AbsDim(self.bottomright - self.topleft)
    }
}

impl Add<Vec2> for AbsRect {
    type Output = Self;

    fn add(self, rhs: Vec2) -> Self::Output {
        Self {
            topleft: self.topleft + rhs,
            bottomright: self.bottomright + rhs,
        }
    }
}

impl AddAssign<Vec2> for AbsRect {
    fn add_assign(&mut self, rhs: Vec2) {
        self.topleft += rhs;
        self.bottomright += rhs;
    }
}

impl Sub<Vec2> for AbsRect {
    type Output = Self;

    fn sub(self, rhs: Vec2) -> Self::Output {
        Self {
            topleft: (self.topleft - rhs).into(),
            bottomright: (self.bottomright - rhs).into(),
        }
    }
}

impl SubAssign<Vec2> for AbsRect {
    fn sub_assign(&mut self, rhs: Vec2) {
        self.topleft -= rhs;
        self.bottomright -= rhs;
    }
}

#[derive(Copy, Clone, Debug, Default)]
/// Relative point
pub struct RelPoint {
    pub x: f32,
    pub y: f32,
}

impl Add for RelPoint {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Mul<AbsDim> for RelPoint {
    type Output = Vec2;

    fn mul(self, rhs: AbsDim) -> Self::Output {
        Vec2 {
            x: self.x * rhs.0.x,
            y: self.y * rhs.0.y,
        }
    }
}

impl From<Vec2> for RelPoint {
    fn from(value: Vec2) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
/// Relative rectangle
pub struct RelRect {
    pub topleft: RelPoint,
    pub bottomright: Vec2,
}

#[derive(Copy, Clone, Debug, Default)]
/// Unified coordinate
pub struct UPoint {
    pub abs: Vec2,
    pub rel: RelPoint,
}

impl Add for UPoint {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            abs: self.abs + other.abs,
            rel: self.rel + other.rel,
        }
    }
}

impl Mul<AbsDim> for UPoint {
    type Output = Vec2;

    fn mul(self, rhs: AbsDim) -> Self::Output {
        self.abs + (self.rel * rhs)
    }
}

impl Mul<AbsRect> for UPoint {
    type Output = Vec2;

    fn mul(self, rhs: AbsRect) -> Self::Output {
        let dim = AbsDim(rhs.bottomright - rhs.topleft);
        rhs.topleft + (self.rel * dim) + self.abs
    }
}

impl From<Vec2> for UPoint {
    fn from(value: Vec2) -> Self {
        Self {
            abs: value,
            rel: Default::default(),
        }
    }
}

pub fn build_aabb(a: Vec2, b: Vec2) -> AbsRect {
    AbsRect {
        topleft: a.min_by_component(b),
        bottomright: a.max_by_component(b),
    }
}

#[derive(Copy, Clone, Debug, Default)]
/// Unified coordinate rectangle
pub struct URect {
    pub topleft: UPoint,
    pub bottomright: UPoint,
}

pub const FILL_URECT: URect = URect {
    topleft: UPoint {
        abs: Vec2 { x: 0.0, y: 0.0 },
        rel: RelPoint { x: 0.0, y: 0.0 },
    },
    bottomright: UPoint {
        abs: Vec2 { x: 0.0, y: 0.0 },
        rel: RelPoint { x: 1.0, y: 1.0 },
    },
};

impl Mul<AbsRect> for URect {
    type Output = AbsRect;

    fn mul(self, rhs: AbsRect) -> Self::Output {
        let dim = AbsDim(rhs.bottomright - rhs.topleft);
        AbsRect {
            topleft: rhs.topleft + (self.topleft * dim),
            bottomright: rhs.topleft + (self.bottomright * dim),
        }
    }
}

impl From<AbsRect> for URect {
    fn from(value: AbsRect) -> Self {
        Self {
            topleft: value.topleft.into(),
            bottomright: value.bottomright.into(),
        }
    }
}

pub trait RenderLambda: Fn(&mut wgpu::RenderPass) + dyn_clone::DynClone {}
impl<T: Fn(&mut wgpu::RenderPass) + ?Sized + dyn_clone::DynClone> RenderLambda for T {}
dyn_clone::clone_trait_object!(RenderLambda);

type EventHandler<AppData> = Box<dyn FnMut(Event, AbsRect, AppData) -> Result<AppData, AppData>>;
// TODO: This only an Option so it can be zeroed. After fixing im::Vector, remove Option.
type RenderInstruction = Option<Box<dyn RenderLambda>>;

pub struct TextSystem {
    viewport: glyphon::Viewport,
    atlas: glyphon::TextAtlas,
    font_system: glyphon::FontSystem,
    swash_cache: glyphon::SwashCache,
}

impl TextSystem {
    pub fn split_borrow(
        &mut self,
    ) -> (
        &mut glyphon::Viewport,
        &mut glyphon::TextAtlas,
        &mut glyphon::FontSystem,
        &mut glyphon::SwashCache,
    ) {
        (
            &mut self.viewport,
            &mut self.atlas,
            &mut self.font_system,
            &mut self.swash_cache,
        )
    }
}

// We want to share our device/adapter state across windows, but can't create it until we have at least one window,
// so we store a weak reference to it in App and if all windows are dropped it'll also drop these, which is usually
// sensible behavior.
pub struct DriverState {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    text: std::rc::Rc<RefCell<TextSystem>>,
}

pub type WindowCollection<AppData> = im::HashMap<WindowId, Window<AppData>>;

pub struct App<AppData: 'static, O: FnPersist<AppData, WindowCollection<AppData>>> {
    pub app_state: Option<AppData>, // Only optional so we can take the value out during processing and put it back in later.
    pub instance: wgpu::Instance,
    windows: WindowCollection<AppData>,
    driver: Weak<DriverState>,
    store: Option<O::Store>,
    outline: Option<O>,
    init: Option<Box<dyn FnOnce(&Self, &winit::event_loop::ActiveEventLoop) -> O>>,
}

#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;

impl<AppData, O: FnPersist<AppData, WindowCollection<AppData>>> App<AppData, O> {
    pub fn new<T: 'static>(
        app_state: AppData,
        init: impl FnOnce(&Self, &winit::event_loop::ActiveEventLoop) -> O + 'static,
    ) -> eyre::Result<(Self, winit::event_loop::EventLoop<T>)> {
        #[cfg(test)]
        let any_thread = true;
        #[cfg(not(test))]
        let any_thread = false;

        #[cfg(target_os = "windows")]
        let event_loop = winit::event_loop::EventLoop::with_user_event()
            .with_any_thread(any_thread)
            .with_dpi_aware(true)
            .build()?;
        #[cfg(not(target_os = "windows"))]
        let event_loop = winit::event_loop::EventLoop::with_user_event().build()?;

        Ok((
            Self {
                app_state: Some(app_state),
                instance: wgpu::Instance::default(),
                windows: im::HashMap::new(),
                driver: Weak::<DriverState>::new(),
                store: Default::default(),
                outline: None,
                init: Some(Box::new(init)),
            },
            event_loop,
        ))
    }

    pub async fn create_driver(
        &mut self,
        surface: &wgpu::Surface<'static>,
    ) -> eyre::Result<Arc<DriverState>> {
        if let Some(driver) = self.driver.upgrade() {
            return Ok(driver);
        }

        let adapter = self
            .instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .ok_or_eyre("Failed to get adapter")?;

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None,
            )
            .await?;

        let cache = glyphon::Cache::new(&device);
        let atlas =
            glyphon::TextAtlas::new(&device, &queue, &cache, wgpu::TextureFormat::Bgra8Unorm);
        let viewport = glyphon::Viewport::new(&device, &cache);

        let driver = Arc::new(crate::DriverState {
            adapter,
            device,
            queue,
            text: std::rc::Rc::new(RefCell::new(TextSystem {
                font_system: glyphon::FontSystem::new(),
                swash_cache: glyphon::SwashCache::new(),
                viewport: viewport,
                atlas,
            })),
        });

        self.driver = Arc::downgrade(&driver);
        Ok(driver)
    }
}

impl<AppData, T: 'static, O: FnPersist<AppData, WindowCollection<AppData>>>
    winit::application::ApplicationHandler<T> for App<AppData, O>
{
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // If this is our first resume, call the start function that can create the necessary graphics context

        if self.outline.is_none() {
            self.outline
                .replace((self.init.take().unwrap())(self, event_loop));
            let (store, windows) = self
                .outline
                .as_ref()
                .unwrap()
                .call(self.store.take().unwrap(), self.app_state.as_ref().unwrap());
            self.store.replace(store);
            self.windows = windows;
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: WindowId,
        event: winit::event::WindowEvent,
    ) {
        let mut delete = None;
        if let Some(window) = self.windows.get_mut(&window_id) {
            let result = match event {
                winit::event::WindowEvent::CloseRequested => window
                    .on_event(event, self.app_state.take().unwrap())
                    .inspect_err(|_| {
                        delete = Some(window_id);
                    }),
                winit::event::WindowEvent::RedrawRequested => {
                    window.layout_all(self.app_state.as_ref().unwrap());
                    window.stage_all();
                    window.render_all();
                    window.on_event(event, self.app_state.take().unwrap())
                }
                _ => window.on_event(event, self.app_state.take().unwrap()),
            };

            self.app_state.replace(match result {
                Ok(data) => data,
                Err(data) => data,
            });
        }

        if let Some(id) = delete {
            self.windows.remove(&id);
        }

        if self.windows.is_empty() {
            event_loop.exit();
        } else if let Some(outline) = self.outline.as_ref() {
            let (store, windows) =
                outline.call(self.store.take().unwrap(), self.app_state.as_ref().unwrap());
            self.store.replace(store);
            self.windows = windows;
        }
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let _ = (event_loop, device_id, event);
    }

    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let _ = event_loop;
    }
}

#[cfg(test)]
struct TestApp {
    window: Arc<winit::window::Window>,
    surface: Arc<wgpu::Surface<'static>>,
    driver: Arc<DriverState>,
}

#[cfg(test)]
impl FnPersist<u8, WindowCollection<u8>> for TestApp {
    type Store = (u8, WindowCollection<u8>);

    fn call(&self, mut store: Self::Store, args: &u8) -> (Self::Store, WindowCollection<u8>) {
        use component::round_rect::RoundRect;
        use layout::root::Root;
        use layout::Desc;
        use ultraviolet::Vec4;

        if store.0 != *args {
            let region = RoundRect::<<Root as Desc<()>>::Impose> {
                fill: Vec4::new(1.0, 0.0, 0.0, 1.0),
                ..Default::default()
            };
            let window_id = self.window.id();
            let window = Window::new::<()>(
                self.window.clone(),
                self.surface.clone(),
                Box::new(region),
                self.driver.clone(),
            )
            .unwrap();

            store.1 = WindowCollection::new();
            store.1.insert(window_id, window);
        }
        let windows = store.1.clone();
        (store, windows)
    }
}

#[test]
fn test_basic() {
    let (mut app, event_loop) = App::new(
        0u8,
        move |app: &App<u8, TestApp>, event_loop: &winit::event_loop::EventLoop<()>| {
            let window = Arc::new(
                event_loop
                    .create_window(
                        winit::window::Window::default_attributes()
                            .with_title("test_blank")
                            .with_resizable(true),
                    )
                    .unwrap(),
            );

            let surface = Arc::new(app.instance.create_surface(window.clone()).unwrap());

            let driver = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(app.create_driver(&surface))
                .unwrap();
        },
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
