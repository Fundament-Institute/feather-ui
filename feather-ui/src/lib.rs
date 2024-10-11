pub mod input;
pub mod layout;
pub mod outline;
//mod lua;
pub mod persist;
mod rtree;
mod shaders;

use crate::outline::window::Window;
use color_eyre::owo_colors::OwoColorize;
use dyn_clone::DynClone;
use eyre::OptionExt;
use outline::Outline;
use outline::StateMachineWrapper;
use persist::FnPersist;
use smallvec::SmallVec;
use std::any::Any;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::hash::Hasher;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use std::rc::Rc;
use std::sync::Arc;
use ultraviolet::vec::Vec2;
use winit::window::WindowId;

macro_rules! gen_id {
    ($($varargs:pat),* $(,)?) => {
        concat!(file!(), line!(), gen_id!(@stringify_varargs $($varargs),*));
    };

    (@stringify_varargs $($expected:pat),*) => {
        gen_id!(@stringify_varargs_helper $($expected),*)
    };
    (@stringify_varargs_helper $first:pat, $($rest:pat),*) => {
        concat!(stringify!($first), $(", ", stringify!($rest)),*)
    };
    (@stringify_varargs_helper $last:pat) => {
        // Do not print last comma
        stringify!($last)
    };
}

impl Into<Vec2> for AbsDim {
    fn into(self) -> Vec2 {
        self.0
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct AbsDim(Vec2);

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

/// Object-safe version of Hash + PartialEq
trait DynHashEq: DynClone {
    fn dyn_hash(&self, state: &mut dyn Hasher);
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}

dyn_clone::clone_trait_object!(DynHashEq);

impl<H: std::hash::Hash + std::cmp::PartialEq + std::cmp::Eq + 'static + Clone> DynHashEq for H {
    fn dyn_hash(&self, mut state: &mut dyn Hasher) {
        self.hash(&mut state);
    }
    fn dyn_eq(&self, other: &dyn Any) -> bool {
        if let Some(o) = other.downcast_ref::<H>() {
            self == o
        } else {
            false
        }
    }
}

#[derive(Clone)]
pub enum DataID {
    Named(String),
    Int(i64),
    Other(Box<dyn DynHashEq>),
}

impl std::hash::Hash for DataID {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            DataID::Named(s) => s.hash(state),
            DataID::Int(i) => i.hash(state),
            DataID::Other(hash_comparable) => hash_comparable.dyn_hash(state),
        }
    }
}

impl std::cmp::Eq for DataID {}
impl std::cmp::PartialEq for DataID {
    fn eq(&self, other: &Self) -> bool {
        match self {
            DataID::Named(s) => {
                if let DataID::Named(name) = other {
                    name == s
                } else {
                    false
                }
            }
            DataID::Int(i) => {
                if let DataID::Int(integer) = other {
                    integer == i
                } else {
                    false
                }
            }
            DataID::Other(hash_comparable) => {
                if let DataID::Other(h) = other {
                    hash_comparable.dyn_eq(h)
                } else {
                    false
                }
            }
        }
    }
}

#[derive(Clone)]
struct SourceID {
    parent: std::rc::Weak<SourceID>,
    id: DataID,
}
impl std::cmp::Eq for SourceID {}
impl std::cmp::PartialEq for SourceID {
    fn eq(&self, other: &Self) -> bool {
        std::rc::Weak::ptr_eq(&self.parent, &other.parent) && self.id == other.id
    }
}
impl std::hash::Hash for SourceID {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if let Some(parent) = self.parent.upgrade() {
            parent.id.hash(state);
        }
        self.id.hash(state);
    }
}
#[derive(Clone)]
struct Slot(Rc<SourceID>, u64);

type AnyHandler = dyn FnMut(
    &dyn Any,
    AbsRect,
    &mut dyn Any,
) -> Result<Vec<(u64, Box<dyn Any>)>, Vec<(u64, Box<dyn Any>)>>;

pub type EventHandler<Input: Dispatchable, Output: Dispatchable, State> =
    Box<dyn FnMut(Input, AbsRect, State) -> Result<(State, Vec<Output>), (State, Vec<Output>)>>;

pub type EventWrapper<Output: Dispatchable, State> = Box<
    dyn FnMut(DispatchPair, AbsRect, State) -> Result<(State, Vec<Output>), (State, Vec<Output>)>,
>;

#[inline]
fn wrap_event<Input: Dispatchable + 'static, Output: Dispatchable, State>(
    mut typed: impl FnMut(Input, AbsRect, State) -> Result<(State, Vec<Output>), (State, Vec<Output>)>,
) -> impl FnMut(DispatchPair, AbsRect, State) -> Result<(State, Vec<Output>), (State, Vec<Output>)>
{
    move |pair, rect, state| typed(Input::restore(pair).unwrap(), rect, state)
}

type DispatchPair = (u64, Box<dyn Any>);

trait Dispatchable
where
    Self: Sized,
{
    const SIZE: usize;
    fn extract(self) -> DispatchPair;
    fn restore(pair: DispatchPair) -> eyre::Result<Self>;
}

impl Dispatchable for () {
    const SIZE: usize = 0;

    fn extract(self) -> DispatchPair {
        (0, Box::new(self))
    }

    fn restore(_: DispatchPair) -> eyre::Result<Self> {
        Ok(())
    }
}

/*#[proc_macro_derive(Dispatch)]
pub fn dispatchable(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let enum_name = &ast.ident;
    let vis = &ast.vis;

    let mut struct_declarations = proc_macro2::TokenStream::new();

    let ns: Path = parse_quote!(evt);
    let skip: Path = parse_quote!(skip);
    let counter: u64 = 1;
    let struct_declarations_iter = variants.iter()
        .filter(|variant| !proc_macro_roids::contains_tag(&variant.attrs,  &ns, &skip))
        .map(|variant| {

        let variant_name = &variant.ident;
}*/

#[derive(Default)]
pub struct StateManager {
    states: HashMap<Rc<SourceID>, Box<dyn StateMachineWrapper>>,
}

impl StateManager {
    fn init_default<'a, State: 'static + outline::StateMachineWrapper + Default>(
        &'a mut self,
        id: Rc<SourceID>,
    ) -> eyre::Result<&'a mut State> {
        if !self.states.contains_key(&id) {
            self.states.insert(id.clone(), Box::new(State::default()));
        }
        let v = self
            .states
            .get_mut(&id)
            .ok_or_eyre("Failed to insert state!")?;
        v.as_any()
            .downcast_mut()
            .ok_or_eyre("Runtime type mismatch!")
    }
    pub fn init<'a, State: 'static + outline::StateMachineWrapper>(
        &'a mut self,
        id: Rc<SourceID>,
        f: impl FnOnce() -> eyre::Result<State>,
    ) -> eyre::Result<&'a mut State> {
        if !self.states.contains_key(&id) {
            self.states.insert(id.clone(), Box::new(f()?));
        }
        let v = self
            .states
            .get_mut(&id)
            .ok_or_eyre("Failed to insert state!")?;
        v.as_any()
            .downcast_mut()
            .ok_or_eyre("Runtime type mismatch!")
    }
    pub fn get<'a, State: 'static + outline::StateMachineWrapper>(
        &'a mut self,
        id: &SourceID,
    ) -> eyre::Result<&'a mut State> {
        let v = self.states.get_mut(id).ok_or_eyre("State does not exist")?;
        Ok(v.as_any()
            .downcast_mut()
            .ok_or_eyre("Runtime type mismatch!")?)
    }
    pub fn get_trait<'a>(
        &'a mut self,
        id: &SourceID,
    ) -> eyre::Result<&'a mut Box<dyn StateMachineWrapper>> {
        Ok(self.states.get_mut(id).ok_or_eyre("State does not exist")?)
    }

    pub fn process(&mut self, event: DispatchPair, slot: &Slot) -> eyre::Result<()> {
        // We use smallvec here so we can satisfy the borrow checker without making yet another heap allocation in most cases
        let iter: SmallVec<[(Box<dyn Any>, u64, Option<Slot>); 2]> = {
            let state = self.states.get_mut(&slot.0).ok_or_eyre("Invalid slot")?;
            let v = state.process(event, slot.1)?;
            v.into_iter().map(|(i, e)| {
                (
                    e,
                    i,
                    state.output_slot(i.try_into().unwrap()).unwrap().clone(),
                )
            })
        }
        .collect();

        for (e, index, slot) in iter {
            if let Some(s) = slot.as_ref() {
                self.process((index, e), s)?;
            }
        }
        Ok(())
    }
}

pub struct App<
    AppData: 'static + std::cmp::PartialEq,
    O: FnPersist<AppData, im::HashMap<winit::window::WindowId, Option<Window<AppData>>>>,
> {
    pub app_state: Option<AppData>, // Only optional so we can take the value out during processing and put it back in later.
    pub instance: wgpu::Instance,
    pub driver: std::sync::Weak<DriverState>,
    state: StateManager,
    store: Option<O::Store>,
    outline: O,
    parents: BTreeMap<DataID, DataID>,
    root: outline::Root<AppData>, // Root outline node containing all windows
}

#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;

impl<
        AppData: std::cmp::PartialEq,
        O: FnPersist<AppData, im::HashMap<winit::window::WindowId, Option<Window<AppData>>>>,
    > App<AppData, O>
{
    pub fn init<'a, State: 'static + outline::StateMachineWrapper>(
        &'a mut self,
        id: Rc<SourceID>,
        f: impl FnOnce() -> eyre::Result<State>,
    ) -> eyre::Result<&'a mut State> {
        self.state.init(id, f)
    }

    pub fn new<T: 'static>(
        app_state: AppData,
        outline: O,
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
                driver: std::sync::Weak::<DriverState>::new(),
                store: None,
                outline,
                state: Default::default(),
                parents: Default::default(),
                root: outline::Root::new(),
            },
            event_loop,
        ))
    }

    pub async fn create_driver(
        weak: &mut std::sync::Weak<DriverState>,
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'static>,
    ) -> eyre::Result<Arc<DriverState>> {
        if let Some(driver) = weak.upgrade() {
            return Ok(driver);
        }

        let adapter = instance
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

        *weak = Arc::downgrade(&driver);
        Ok(driver)
    }

    fn update_outline(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, store: O::Store) {
        let (store, windows) = self.outline.call(store, self.app_state.as_ref().unwrap());
        self.store.replace(store);
        self.root.children = windows;

        let (root, manager, driver, instance) = (
            &mut self.root,
            &mut self.state,
            &mut self.driver,
            &mut self.instance,
        );

        root.layout_all::<O>(manager, driver, instance, event_loop);

        root.stage_all(manager);
    }
}

impl<
        AppData: std::fmt::Debug + std::cmp::PartialEq,
        T: 'static,
        O: FnPersist<AppData, im::HashMap<winit::window::WindowId, Option<Window<AppData>>>>,
    > winit::application::ApplicationHandler<T> for App<AppData, O>
{
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // If this is our first resume, call the start function that can create the necessary graphics context
        if let Some(store) = self.store.take() {
            self.update_outline(event_loop, store);
        }
    }
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: WindowId,
        event: winit::event::WindowEvent,
    ) {
        let mut delete = None;
        if let Some(window) = self.root.children.get(&window_id) {
            let window = window.as_ref().unwrap();
            match event {
                winit::event::WindowEvent::CloseRequested => Window::<AppData>::on_window_event(
                    window.id(),
                    std::rc::Weak::new(),
                    event,
                    &mut self.state,
                )
                .inspect_err(|_| {
                    delete = Some(window_id);
                }),
                winit::event::WindowEvent::RedrawRequested => {
                    if let Some(root) = self.root.state.get_mut(&window.id()) {
                        Window::render(&window.id(), root, &mut self.state);
                    }
                    Window::<AppData>::on_window_event(
                        window.id(),
                        std::rc::Weak::new(),
                        event,
                        &mut self.state,
                    )
                }
                _ => Window::<AppData>::on_window_event(
                    window.id(),
                    std::rc::Weak::new(),
                    event,
                    &mut self.state,
                ),
            };

            //if self.app_state.is_none() || data != *self.app_state.as_ref().unwrap() {
            //   self.app_state.replace(data);
            let store = self.store.take().unwrap();
            self.update_outline(event_loop, store);
            //}
        }

        if let Some(id) = delete {
            self.root.children.remove(&id);
        }

        if self.root.children.is_empty() {
            event_loop.exit();
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
struct TestApp {}

#[cfg(test)]
impl FnPersist<AppData, Box<dyn Outline<AppData, (), State = ()>>> for TestApp {
    type Store = (u8, Box<dyn Outline<AppData, (), State = ()>>);

    fn init(&self) -> Self::Store {
        let region = RoundRect::<<Root as Desc<()>>::Impose> {
            fill: Vec4::new(1.0, 0.0, 0.0, 1.0),
            ..Default::default()
        };
        let window = Window::new::<()>(
            gen_id!(),
            winit::window::Window::default_attributes()
                .with_title("test_blank")
                .with_resizable(true),
            Box::new(region),
        );

        store.1 = WindowCollection::new();
        store.1.insert(window_id, window);
        store.0 = *args;

        (0, Box::new(window))
    }
    fn call(&self, mut store: Self::Store, args: &u8) -> (Self::Store, WindowCollection<u8>) {
        use layout::root::Root;
        use layout::Desc;
        use outline::round_rect::RoundRect;
        use ultraviolet::Vec4;

        if store.0 != *args {}
        let windows = store.1.clone();
        (store, windows)
    }
}

#[test]
fn test_basic() {
    let (mut app, event_loop): (App<u8, TestApp>, winit::event_loop::EventLoop<()>) = App::new(
        1u8,
        move |app: &mut App<u8, TestApp>, event_loop: &winit::event_loop::ActiveEventLoop| {
            TestApp {}
        },
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
