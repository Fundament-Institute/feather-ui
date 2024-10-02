pub mod component;
pub mod input;
pub mod layout;
//mod lua;
pub mod persist;
mod rtree;
mod shaders;

use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use std::sync::Weak;

use crate::component::window::Window;
use crate::input::Event;
use component::Component;
use eyre::OptionExt;
use std::cell::RefCell;
use std::collections::HashMap;
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

pub trait RenderLambda: Fn(&mut wgpu::RenderPass) + dyn_clone::DynClone {}
impl<T: Fn(&mut wgpu::RenderPass) + ?Sized + dyn_clone::DynClone> RenderLambda for T {}
dyn_clone::clone_trait_object!(RenderLambda);

type EventHandler<AppData> = Box<dyn FnMut(Event, AbsRect, AppData) -> AppData>;
// TODO: This only an Option so it can be zeroed. After fixing im::Vector, remove Option.
type RenderInstruction = Option<Box<dyn RenderLambda>>;

pub struct TextSystem {
    viewport: glyphon::Viewport,
    atlas: glyphon::TextAtlas,
    text_renderer: glyphon::TextRenderer,
    font_system: glyphon::FontSystem,
    swash_cache: glyphon::SwashCache,
}

impl TextSystem {
    pub fn split_borrow(
        &mut self,
    ) -> (
        &mut glyphon::Viewport,
        &mut glyphon::TextAtlas,
        &mut glyphon::TextRenderer,
        &mut glyphon::FontSystem,
        &mut glyphon::SwashCache,
    ) {
        (
            &mut self.viewport,
            &mut self.atlas,
            &mut self.text_renderer,
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

pub struct App<AppData: 'static> {
    pub app_state: AppData,
    pub instance: wgpu::Instance,
    pub windows: HashMap<winit::window::WindowId, Window<AppData>>,
    driver: Weak<DriverState>,
}

#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;

impl<AppData> App<AppData> {
    pub fn new<T: 'static>(
        app_state: AppData,
    ) -> eyre::Result<(Self, winit::event_loop::EventLoop<T>)> {
        #[cfg(test)]
        let any_thread = true;
        #[cfg(not(test))]
        let any_thread = false;

        #[cfg(target_os = "windows")]
        let event_loop = winit::event_loop::EventLoopBuilder::with_user_event()
            .with_any_thread(any_thread)
            .with_dpi_aware(true)
            .build()?;
        #[cfg(not(target_os = "windows"))]
        let event_loop = winit::event_loop::EventLoopBuilder::with_user_event().build()?;

        Ok((
            Self {
                app_state,
                instance: wgpu::Instance::default(),
                windows: HashMap::new(),
                driver: Weak::<DriverState>::new(),
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
        let mut atlas =
            glyphon::TextAtlas::new(&device, &queue, &cache, wgpu::TextureFormat::Bgra8Unorm);
        let viewport = glyphon::Viewport::new(&device, &cache);
        let text_renderer = glyphon::TextRenderer::new(
            &mut atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );

        let driver = Arc::new(crate::DriverState {
            adapter,
            device,
            queue,
            text: std::rc::Rc::new(RefCell::new(TextSystem {
                font_system: glyphon::FontSystem::new(),
                swash_cache: glyphon::SwashCache::new(),
                viewport: viewport,
                atlas,
                text_renderer: text_renderer,
            })),
        });

        self.driver = Arc::downgrade(&driver);
        Ok(driver)
    }

    pub fn event<T: 'static>(
        &mut self,
        event: winit::event::Event<T>,
        target: &winit::event_loop::EventLoopWindowTarget<T>,
    ) {
        if let winit::event::Event::WindowEvent { window_id, event } = event {
            let mut delete = None;
            if let Some(window) = self.windows.get_mut(&window_id) {
                match event {
                    winit::event::WindowEvent::CloseRequested => {
                        if window.on_event(event) {
                            delete = Some(window_id)
                        }
                    }
                    winit::event::WindowEvent::RedrawRequested => {
                        window.layout_all(&self.app_state);
                        window.stage_all();
                        window.render_all();
                        window.on_event(event);
                    }
                    _ => {
                        window.on_event(event);
                    }
                }
            }

            if let Some(id) = delete {
                self.windows.remove(&id);
            }

            if self.windows.is_empty() {
                target.exit();
            }
        }
    }
}

#[test]
fn test_basic() {
    use component::region::Region;
    use component::round_rect::RoundRect;
    use layout::root::Root;
    use layout::Desc;
    use ultraviolet::Vec4;
    use wgpu::Device;
    use wgpu::RenderPass;
    use winit::window::WindowBuilder;

    let (mut app, event_loop) = App::<()>::new(()).unwrap();

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("test_blank")
            .with_resizable(true)
            .build(&event_loop)
            .unwrap(),
    );

    let surface = app.instance.create_surface(window.clone()).unwrap();

    let driver = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(app.create_driver(&surface))
        .unwrap();

    let mut region = RoundRect::<<Root as Desc<()>>::Impose> {
        fill: Vec4::new(1.0, 0.0, 0.0, 1.0),
        ..Default::default()
    };
    let window_id = window.id();
    let mut window = Window::new::<()>(window, surface, Box::new(region), driver).unwrap();

    app.windows.insert(window_id, window);

    event_loop
        .run(
            move |event: winit::event::Event<()>,
                  target: &winit::event_loop::EventLoopWindowTarget<()>| {
                app.event(event, target)
            },
        )
        .unwrap();
}
