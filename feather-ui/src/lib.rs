// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

extern crate alloc;

pub mod input;
pub mod layout;
pub mod lua;
pub mod outline;
pub mod persist;
mod propbag;
mod rtree;
mod shaders;

use crate::outline::window::Window;
use core::cell::Cell;
use core::f32;
use dyn_clone::DynClone;
use eyre::OptionExt;
pub use glyphon::Wrap;
use once_cell::unsync::OnceCell;
use outline::window::WindowStateMachine;
use outline::Outline;
use outline::StateMachineWrapper;
use persist::FnPersist;
use shaders::ShaderCache;
use smallvec::SmallVec;
use std::any::Any;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::hash::Hasher;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use std::rc::Rc;
use ultraviolet::vec::Vec2;
use winit::window::WindowId;

/*
/// Allocates `&[T]` on stack space. (currently unused but might be needed depending on persistent data structures)
pub(crate) fn alloca_array<T, R>(n: usize, f: impl FnOnce(&mut [T]) -> R) -> R {
    use std::mem::{align_of, size_of};

    alloca::with_alloca_zeroed(
        (n * size_of::<T>()) + (align_of::<T>() - 1),
        |memory| unsafe {
            let mut raw_memory = memory.as_mut_ptr();
            if raw_memory as usize % align_of::<T>() != 0 {
                raw_memory =
                    raw_memory.add(align_of::<T>() - raw_memory as usize % align_of::<T>());
            }

            f(std::slice::from_raw_parts_mut::<T>(
                raw_memory.cast::<T>(),
                n,
            ))
        },
    )
}
*/

#[macro_export]
macro_rules! gen_id {
    () => {
        $crate::SourceID::new($crate::DataID::Named(concat!(file!(), ":", line!())))
    };
    ($idx:expr) => {
        $idx.child($crate::DataID::Named(concat!(file!(), ":", line!())))
    };
}

use std::any::TypeId;
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Not an error, this outline simply has no layout state.")]
    Stateless,
    #[error("Enun object didn't match tag {0}! Expected {1:?} but got {2:?}")]
    MismatchedEnumTag(u64, TypeId, TypeId),
    #[error("Invalid enum tag: {0}")]
    InvalidEnumTag(u64),
    #[error("Event handler didn't handle this method.")]
    UnhandledEvent,
}

pub const UNSIZED_AXIS: f32 = f32::MAX;
pub const ZERO_POINT: Vec2 = Vec2 { x: 0.0, y: 0.0 };

#[derive(Copy, Clone, Debug, Default)]
pub struct AbsDim(Vec2);

impl From<AbsDim> for Vec2 {
    fn from(val: AbsDim) -> Self {
        val.0
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Absolutely positioned rectangle
pub struct AbsRect {
    pub topleft: Vec2,
    pub bottomright: Vec2,
}

pub const ZERO_RECT: AbsRect = AbsRect {
    topleft: ZERO_POINT,
    bottomright: ZERO_POINT,
};

impl AbsRect {
    #[inline]
    pub fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            topleft: Vec2 { x: left, y: top },
            bottomright: Vec2 {
                x: right,
                y: bottom,
            },
        }
    }
    #[inline]
    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.topleft.x
            && p.x <= self.bottomright.x
            && p.y >= self.topleft.y
            && p.y <= self.bottomright.y
    }

    #[inline]
    pub fn extend(&self, rhs: AbsRect) -> AbsRect {
        AbsRect {
            topleft: self.topleft.min_by_component(rhs.topleft),
            bottomright: self.bottomright.min_by_component(rhs.bottomright),
        }
    }

    #[inline]
    pub fn width(&self) -> f32 {
        self.bottomright.x - self.topleft.x
    }
    #[inline]
    pub fn height(&self) -> f32 {
        self.bottomright.y - self.topleft.y
    }
    #[inline]
    pub fn dim(&self) -> AbsDim {
        AbsDim(self.bottomright - self.topleft)
    }
}

impl Add<Vec2> for AbsRect {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Vec2) -> Self::Output {
        Self {
            topleft: self.topleft + rhs,
            bottomright: self.bottomright + rhs,
        }
    }
}

impl AddAssign<Vec2> for AbsRect {
    #[inline]
    fn add_assign(&mut self, rhs: Vec2) {
        self.topleft += rhs;
        self.bottomright += rhs;
    }
}

impl Sub<Vec2> for AbsRect {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Vec2) -> Self::Output {
        Self {
            topleft: self.topleft - rhs,
            bottomright: self.bottomright - rhs,
        }
    }
}

impl SubAssign<Vec2> for AbsRect {
    #[inline]
    fn sub_assign(&mut self, rhs: Vec2) {
        self.topleft -= rhs;
        self.bottomright -= rhs;
    }
}

impl From<AbsDim> for AbsRect {
    fn from(value: AbsDim) -> Self {
        AbsRect {
            topleft: ZERO_POINT,
            bottomright: value.0,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Relative point
pub struct RelPoint(pub Vec2);

impl From<Vec2> for RelPoint {
    #[inline]
    fn from(value: Vec2) -> Self {
        Self(value)
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Relative rectangle
pub struct RelRect {
    pub topleft: RelPoint,
    pub bottomright: RelPoint,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Unified coordinate
pub struct UPoint {
    pub abs: Vec2,
    pub rel: RelPoint,
}

impl Add for UPoint {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            abs: self.abs + other.abs,
            rel: RelPoint(self.rel.0 + other.rel.0),
        }
    }
}

impl Sub for UPoint {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            abs: self.abs - rhs.abs,
            rel: RelPoint(self.rel.0 - rhs.rel.0),
        }
    }
}

impl Mul<AbsDim> for UPoint {
    type Output = Vec2;

    #[inline]
    fn mul(self, rhs: AbsDim) -> Self::Output {
        self.abs + (self.rel.0 * rhs.0)
    }
}

impl Mul<AbsRect> for UPoint {
    type Output = Vec2;

    #[inline]
    fn mul(self, rhs: AbsRect) -> Self::Output {
        let dim = AbsDim(rhs.bottomright - rhs.topleft);
        rhs.topleft + (self.rel.0 * dim.0) + self.abs
    }
}

impl From<Vec2> for UPoint {
    #[inline]
    fn from(value: Vec2) -> Self {
        Self {
            abs: value,
            rel: Default::default(),
        }
    }
}

impl From<RelPoint> for UPoint {
    #[inline]
    fn from(value: RelPoint) -> Self {
        Self {
            abs: Default::default(),
            rel: value,
        }
    }
}

pub const ZERO_UPOINT: UPoint = UPoint {
    abs: Vec2 { x: 0.0, y: 0.0 },
    rel: RelPoint(Vec2 { x: 0.0, y: 0.0 }),
};

#[derive(Copy, Clone, Debug, Default)]
pub struct UDim(UPoint);

impl From<UDim> for UPoint {
    #[inline]
    fn from(val: UDim) -> Self {
        val.0
    }
}

#[inline]
pub fn build_aabb(a: Vec2, b: Vec2) -> AbsRect {
    AbsRect {
        topleft: a.min_by_component(b),
        bottomright: a.max_by_component(b),
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Unified coordinate rectangle
pub struct URect {
    pub topleft: UPoint,
    pub bottomright: UPoint,
}

pub const ZERO_URECT: URect = URect {
    topleft: ZERO_UPOINT,
    bottomright: ZERO_UPOINT,
};

pub const FILL_URECT: URect = URect {
    topleft: UPoint {
        abs: Vec2 { x: 0.0, y: 0.0 },
        rel: RelPoint(Vec2 { x: 0.0, y: 0.0 }),
    },
    bottomright: UPoint {
        abs: Vec2 { x: 0.0, y: 0.0 },
        rel: RelPoint(Vec2 { x: 1.0, y: 1.0 }),
    },
};

pub const AUTO_URECT: URect = URect {
    topleft: UPoint {
        abs: Vec2 { x: 0.0, y: 0.0 },
        rel: RelPoint(Vec2 { x: 0.0, y: 0.0 }),
    },
    bottomright: UPoint {
        abs: Vec2 { x: 0.0, y: 0.0 },
        rel: RelPoint(Vec2 {
            x: UNSIZED_AXIS,
            y: UNSIZED_AXIS,
        }),
    },
};

impl Mul<AbsRect> for URect {
    type Output = AbsRect;

    #[inline]
    fn mul(self, rhs: AbsRect) -> Self::Output {
        let dim = AbsDim(rhs.bottomright - rhs.topleft);
        AbsRect {
            topleft: rhs.topleft + (self.topleft * dim),
            bottomright: rhs.topleft + (self.bottomright * dim),
        }
    }
}

impl Mul<AbsDim> for URect {
    type Output = AbsRect;

    #[inline]
    fn mul(self, rhs: AbsDim) -> Self::Output {
        AbsRect {
            topleft: self.topleft * rhs,
            bottomright: self.bottomright * rhs,
        }
    }
}

impl From<AbsRect> for URect {
    #[inline]
    fn from(value: AbsRect) -> Self {
        Self {
            topleft: value.topleft.into(),
            bottomright: value.bottomright.into(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AbsLimits(AbsRect);

// It would be cheaper to avoid using actual infinities here but we currently need them to make the math work
pub const DEFAULT_LIMITS: AbsLimits = AbsLimits(AbsRect {
    topleft: Vec2 {
        x: f32::NEG_INFINITY,
        y: f32::NEG_INFINITY,
    },
    bottomright: Vec2 {
        x: f32::INFINITY,
        y: f32::INFINITY,
    },
});

impl AbsLimits {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self(AbsRect {
            topleft: min,
            bottomright: max,
        })
    }
    #[inline]
    pub fn min(&self) -> Vec2 {
        self.0.topleft
    }
    #[inline]
    pub fn max(&self) -> Vec2 {
        self.0.bottomright
    }
}

impl Default for AbsLimits {
    #[inline]
    fn default() -> Self {
        DEFAULT_LIMITS
    }
}

impl Add<AbsLimits> for AbsLimits {
    type Output = Self;

    #[inline]
    fn add(self, rhs: AbsLimits) -> Self::Output {
        Self(AbsRect {
            topleft: self.min().max_by_component(rhs.min()),
            bottomright: self.max().min_by_component(rhs.max()),
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RelLimits(RelRect);

pub const DEFAULT_RLIMITS: RelLimits = RelLimits(RelRect {
    topleft: RelPoint(DEFAULT_LIMITS.0.topleft),
    bottomright: RelPoint(DEFAULT_LIMITS.0.bottomright),
});

impl Default for RelLimits {
    fn default() -> Self {
        DEFAULT_RLIMITS
    }
}

impl RelLimits {
    #[inline]
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self(RelRect {
            topleft: RelPoint(min),
            bottomright: RelPoint(max),
        })
    }
    #[inline]
    fn min(&self) -> &RelPoint {
        &self.0.topleft
    }
    #[inline]
    fn max(&self) -> &RelPoint {
        &self.0.bottomright
    }
}

impl Mul<AbsDim> for RelLimits {
    type Output = AbsLimits;

    #[inline]
    fn mul(self, rhs: AbsDim) -> Self::Output {
        let (unsized_x, unsized_y) = crate::layout::check_unsized_dim(rhs);
        AbsLimits(AbsRect {
            topleft: Vec2 {
                x: if unsized_x {
                    self.min().0.x
                } else {
                    self.min().0.x * rhs.0.x
                },
                y: if unsized_y {
                    self.min().0.y
                } else {
                    self.min().0.y * rhs.0.y
                },
            },
            bottomright: Vec2 {
                x: if unsized_x {
                    self.max().0.x
                } else {
                    self.max().0.x * rhs.0.x
                },
                y: if unsized_y {
                    self.max().0.y
                } else {
                    self.max().0.y * rhs.0.y
                },
            },
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, derive_more::TryFrom)]
#[repr(u8)]
pub enum RowDirection {
    #[default]
    LeftToRight,
    RightToLeft,
    TopToBottom,
    BottomToTop,
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
    format: Cell<Option<wgpu::TextureFormat>>,
    text: OnceCell<std::rc::Rc<RefCell<TextSystem>>>,
    shader_cache: RefCell<ShaderCache>,
}

impl DriverState {
    fn text(&self) -> eyre::Result<&std::rc::Rc<RefCell<TextSystem>>> {
        self.text.get_or_try_init(|| {
            let format = self
                .format
                .get()
                .ok_or_eyre("driver.text initialized called before driver.format is provided")?;
            let cache = glyphon::Cache::new(&self.device);
            let atlas = glyphon::TextAtlas::new(&self.device, &self.queue, &cache, format);
            let viewport = glyphon::Viewport::new(&self.device, &cache);
            let text = std::rc::Rc::new(RefCell::new(TextSystem {
                font_system: glyphon::FontSystem::new(),
                swash_cache: glyphon::SwashCache::new(),
                viewport,
                atlas,
            }));
            Ok(text)
        })
    }
}

/// Object-safe version of Hash + PartialEq
pub trait DynHashEq: DynClone {
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

#[derive(Clone, Default)]
pub enum DataID {
    Named(&'static str),
    Owned(String),
    Int(i64),
    Other(Box<dyn DynHashEq>),
    #[default]
    None, // Marks an invalid default ID, crashes if you ever try to actually use it.
}

impl std::hash::Hash for DataID {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            DataID::Named(s) => s.hash(state),
            DataID::Owned(s) => s.hash(state),
            DataID::Int(i) => i.hash(state),
            DataID::Other(hash_comparable) => hash_comparable.dyn_hash(state),
            DataID::None => {
                panic!("Invalid ID! Did you forget to initialize an outline node's ID field?")
            }
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
            DataID::Owned(s) => {
                if let DataID::Owned(name) = other {
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
            DataID::None => panic!("Invalid ID!"),
        }
    }
}

#[derive(Clone, Default)]
pub struct SourceID {
    parent: Option<std::rc::Rc<SourceID>>,
    id: DataID,
}

impl SourceID {
    pub fn new(id: DataID) -> Self {
        Self { parent: None, id }
    }
    pub fn child(self: &Rc<Self>, id: DataID) -> Self {
        Self {
            parent: Some(self.clone()),
            id,
        }
    }
}
impl std::cmp::Eq for SourceID {}
impl std::cmp::PartialEq for SourceID {
    fn eq(&self, other: &Self) -> bool {
        if let Some(parent) = &self.parent {
            if let Some(pother) = &other.parent {
                std::rc::Rc::ptr_eq(parent, pother) && self.id == other.id
            } else {
                false
            }
        } else {
            other.parent.is_none() && self.id == other.id
        }
    }
}
impl std::hash::Hash for SourceID {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if let Some(parent) = &self.parent {
            parent.id.hash(state);
        }
        self.id.hash(state);
    }
}
#[derive(Clone)]
pub struct Slot(pub Rc<SourceID>, pub u64);

#[allow(dead_code)]
type AnyHandler =
    dyn FnMut(&dyn Any, AbsRect, &mut dyn Any) -> Result<Vec<DispatchPair>, Vec<DispatchPair>>;

pub type EventHandler<Input, Output, State> =
    Box<dyn FnMut(Input, AbsRect, State) -> Result<(State, Vec<Output>), (State, Vec<Output>)>>;

pub type EventWrapper<Output, State> = Box<
    dyn FnMut(DispatchPair, AbsRect, State) -> Result<(State, Vec<Output>), (State, Vec<Output>)>,
>;

pub type AppEvent<State> = Box<dyn FnMut(DispatchPair, State) -> Result<State, State>>;

#[inline]
#[allow(clippy::type_complexity)]
fn wrap_event<Input: Dispatchable + 'static, Output: Dispatchable, State>(
    mut typed: impl FnMut(Input, AbsRect, State) -> Result<(State, Vec<Output>), (State, Vec<Output>)>,
) -> impl FnMut(DispatchPair, AbsRect, State) -> Result<(State, Vec<Output>), (State, Vec<Output>)>
{
    move |pair, rect, state| typed(Input::restore(pair).unwrap(), rect, state)
}

pub trait WrapEventEx<State: 'static + std::cmp::PartialEq, Input: Dispatchable + 'static> {
    fn wrap(self) -> impl FnMut(DispatchPair, State) -> Result<State, State>;
}

impl<AppData: 'static + std::cmp::PartialEq, Input: Dispatchable + 'static, T>
    WrapEventEx<AppData, Input> for T
where
    T: FnMut(Input, AppData) -> Result<AppData, AppData>,
{
    fn wrap(mut self) -> impl FnMut(DispatchPair, AppData) -> Result<AppData, AppData> {
        move |pair, state| (self)(Input::restore(pair).unwrap(), state)
    }
}

pub type DispatchPair = (u64, Box<dyn Any>);

pub trait Dispatchable
where
    Self: Sized,
{
    const SIZE: usize;
    fn extract(self) -> DispatchPair;
    fn restore(pair: DispatchPair) -> Result<Self, Error>;
}

impl Dispatchable for () {
    const SIZE: usize = 0;

    fn extract(self) -> DispatchPair {
        (0, Box::new(self))
    }

    fn restore(_: DispatchPair) -> Result<Self, Error> {
        Ok(())
    }
}

#[derive(Default)]
pub struct StateManager {
    states: HashMap<Rc<SourceID>, Box<dyn StateMachineWrapper>>,
}

impl StateManager {
    #[allow(dead_code)]
    fn init_default<State: 'static + outline::StateMachineWrapper + Default>(
        &mut self,
        id: Rc<SourceID>,
    ) -> eyre::Result<&mut State> {
        if !self.states.contains_key(&id) {
            self.states.insert(id.clone(), Box::new(State::default()));
        }
        let v = self
            .states
            .get_mut(&id)
            .ok_or_eyre("Failed to insert state!")?;
        v.as_any_mut()
            .downcast_mut()
            .ok_or_eyre("Runtime type mismatch!")
    }
    pub fn init(&mut self, id: Rc<SourceID>, state: Box<dyn StateMachineWrapper>) {
        if !self.states.contains_key(&id) {
            self.states.insert(id.clone(), state);
        }
    }
    pub fn get<'a, State: 'static + outline::StateMachineWrapper>(
        &'a self,
        id: &SourceID,
    ) -> eyre::Result<&'a State> {
        let v = self.states.get(id).ok_or_eyre("State does not exist")?;
        v.as_any()
            .downcast_ref()
            .ok_or_eyre("Runtime type mismatch!")
    }
    pub fn get_mut<'a, State: 'static + outline::StateMachineWrapper>(
        &'a mut self,
        id: &SourceID,
    ) -> eyre::Result<&'a mut State> {
        let v = self.states.get_mut(id).ok_or_eyre("State does not exist")?;
        v.as_any_mut()
            .downcast_mut()
            .ok_or_eyre("Runtime type mismatch!")
    }
    #[allow(clippy::borrowed_box)]
    pub fn get_trait<'a>(
        &'a self,
        id: &SourceID,
    ) -> eyre::Result<&'a Box<dyn StateMachineWrapper>> {
        self.states.get(id).ok_or_eyre("State does not exist")
    }

    pub fn process(&mut self, event: DispatchPair, slot: &Slot, area: AbsRect) -> eyre::Result<()> {
        type IterTuple = (Box<dyn Any>, u64, Option<Slot>);

        // We use smallvec here so we can satisfy the borrow checker without making yet another heap allocation in most cases
        let iter: SmallVec<[IterTuple; 2]> = {
            let state = self.states.get_mut(&slot.0).ok_or_eyre("Invalid slot")?;
            let v = state.process(event, slot.1, area)?;
            v.into_iter()
                .map(|(i, e)| (e, i, state.output_slot(i.ilog2() as usize).unwrap().clone()))
        }
        .collect();

        for (e, index, slot) in iter {
            if let Some(s) = slot.as_ref() {
                self.process((index, e), s, area)?;
            }
        }
        Ok(())
    }

    fn init_outline<Parent: ?Sized>(
        &mut self,
        target: &dyn crate::outline::OutlineWrap<Parent>,
    ) -> eyre::Result<()> {
        if !self.states.contains_key(&target.id()) {
            match target.init() {
                Ok(v) => self.init(target.id().clone(), v),
                Err(Error::Stateless) => (),
                Err(e) => return Err(e.into()),
            };
        }
        target.init_all(self)
    }
}

pub const APP_SOURCE_ID: SourceID = SourceID {
    parent: None,
    id: DataID::Named("__fg_AppData_ID__"),
};

pub struct App<
    AppData: 'static + std::cmp::PartialEq,
    O: FnPersist<AppData, im::HashMap<Rc<SourceID>, Option<Window>>>,
> {
    pub instance: wgpu::Instance,
    pub driver: std::rc::Weak<DriverState>,
    state: StateManager,
    store: Option<O::Store>,
    outline: O,
    _parents: BTreeMap<DataID, DataID>,
    root: outline::Root, // Root outline node containing all windows
}

struct AppDataMachine<AppData: 'static + std::cmp::PartialEq> {
    pub state: Option<AppData>,
    input: Vec<AppEvent<AppData>>,
    pub changed: bool,
}

impl<AppData: 'static + std::cmp::PartialEq> StateMachineWrapper for AppDataMachine<AppData> {
    fn process(
        &mut self,
        input: DispatchPair,
        index: u64,
        _: AbsRect,
    ) -> eyre::Result<Vec<DispatchPair>> {
        let f = self
            .input
            .get_mut(index as usize)
            .ok_or_eyre("index out of bounds")?;
        let processed = match f(input, self.state.take().unwrap()) {
            Ok(s) => Some(s),
            Err(s) => Some(s),
        };
        // If it actually changed, set the change marker
        self.changed = self.changed || (processed != self.state);
        self.state = processed;
        Ok(Vec::new())
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn output_slot(&self, _: usize) -> eyre::Result<&Option<Slot>> {
        Ok(&None)
    }

    fn input_masks(&self) -> SmallVec<[u64; 4]> {
        self.input.iter().map(|_| 0).collect()
    }
}
#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;

//  This logic is the same for both X11 and Wayland because the any_thread variable is the same on both
#[cfg(target_os = "linux")]
use winit::platform::x11::EventLoopBuilderExtX11;

impl<
        AppData: std::cmp::PartialEq,
        O: FnPersist<AppData, im::HashMap<Rc<SourceID>, Option<Window>>>,
    > App<AppData, O>
{
    pub fn new<T: 'static>(
        app_state: AppData,
        inputs: Vec<AppEvent<AppData>>,
        outline: O,
    ) -> eyre::Result<(Self, winit::event_loop::EventLoop<T>)> {
        #[cfg(test)]
        let any_thread = true;
        #[cfg(not(test))]
        let any_thread = false;

        Self::new_any_thread(app_state, inputs, outline, any_thread)
    }

    pub fn new_any_thread<T: 'static>(
        app_state: AppData,
        inputs: Vec<AppEvent<AppData>>,
        outline: O,
        any_thread: bool,
    ) -> eyre::Result<(Self, winit::event_loop::EventLoop<T>)> {
        #[cfg(target_os = "windows")]
        let event_loop = winit::event_loop::EventLoop::with_user_event()
            .with_any_thread(any_thread)
            .with_dpi_aware(true)
            .build()?;
        #[cfg(not(target_os = "windows"))]
        let event_loop = winit::event_loop::EventLoop::with_user_event()
            .with_any_thread(any_thread)
            .build()?;
        let mut manager: StateManager = Default::default();
        manager.init(
            Rc::new(APP_SOURCE_ID),
            Box::new(AppDataMachine {
                input: inputs,
                state: Some(app_state),
                changed: false,
            }),
        );

        Ok((
            Self {
                instance: wgpu::Instance::default(),
                driver: std::rc::Weak::<DriverState>::new(),
                store: None,
                outline,
                state: manager,
                _parents: Default::default(),
                root: outline::Root::new(),
            },
            event_loop,
        ))
    }

    pub async fn create_driver(
        weak: &mut std::rc::Weak<DriverState>,
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'static>,
    ) -> eyre::Result<Rc<DriverState>> {
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

        let shader_cache = ShaderCache::new(&device);

        let driver = Rc::new(crate::DriverState {
            adapter,
            device,
            queue,
            format: Cell::new(None),
            text: OnceCell::new(),
            shader_cache: RefCell::new(shader_cache),
        });

        *weak = Rc::downgrade(&driver);
        Ok(driver)
    }

    fn update_outline(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, store: O::Store) {
        let app_state: &AppDataMachine<AppData> = self.state.get(&APP_SOURCE_ID).unwrap();
        let (store, windows) = self.outline.call(store, app_state.state.as_ref().unwrap());
        self.store.replace(store);
        self.root.children = windows;

        let (root, manager, driver, instance) = (
            &mut self.root,
            &mut self.state,
            &mut self.driver,
            &mut self.instance,
        );

        root.layout_all::<AppData, O>(manager, driver, instance, event_loop)
            .unwrap();

        root.stage_all(manager).unwrap();
    }
}

impl<
        AppData: std::cmp::PartialEq,
        T: 'static,
        O: FnPersist<AppData, im::HashMap<Rc<SourceID>, Option<Window>>>,
    > winit::application::ApplicationHandler<T> for App<AppData, O>
{
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // If this is our first resume, call the start function that can create the necessary graphics context
        let store = self.store.take();
        self.update_outline(event_loop, store.unwrap_or_else(|| O::init(&self.outline)));
    }
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: WindowId,
        event: winit::event::WindowEvent,
    ) {
        let mut delete = None;
        if let Some(root) = self.root.states.get_mut(&window_id) {
            let Some(rtree) = root.staging.as_ref().map(|x| x.get_rtree()) else {
                panic!("Got root state without valid staging!");
            };

            if let Some(window) = self.root.children.get(&root.id) {
                let window = window.as_ref().unwrap();
                let mut resized = false;
                let _ = match event {
                    winit::event::WindowEvent::CloseRequested => {
                        Window::on_window_event(window.id(), rtree, event, &mut self.state)
                            .inspect_err(|_| {
                                delete = Some(window.id());
                            })
                    }
                    winit::event::WindowEvent::RedrawRequested => {
                        if let Ok(state) = self.state.get_mut::<WindowStateMachine>(&window.id()) {
                            if let Some(driver) = self.driver.upgrade() {
                                if let Some(staging) = root.staging.as_ref() {
                                    state.state.as_mut().unwrap().draw =
                                        staging.render(Vec2::zero(), &driver);
                                }
                            }
                        }
                        Window::on_window_event(window.id(), rtree, event, &mut self.state)
                    }
                    winit::event::WindowEvent::Resized(_) => {
                        resized = true;
                        Window::on_window_event(window.id(), rtree, event, &mut self.state)
                    }
                    _ => Window::on_window_event(window.id(), rtree, event, &mut self.state),
                };

                let app_state: &mut AppDataMachine<AppData> =
                    self.state.get_mut(&APP_SOURCE_ID).unwrap();
                if app_state.changed || resized {
                    app_state.changed = false;
                    let store = self.store.take().unwrap();
                    self.update_outline(event_loop, store);
                }
            }
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

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, _: T) {
        event_loop.exit();
    }

    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let _ = event_loop;
    }
}
#[cfg(test)]
struct TestApp {}

#[cfg(test)]
impl FnPersist<u8, im::HashMap<Rc<SourceID>, Option<Window>>> for TestApp {
    type Store = (u8, im::HashMap<Rc<SourceID>, Option<Window>>);

    fn init(&self) -> Self::Store {
        use crate::outline::shape::Shape;
        use ultraviolet::Vec4;
        let rect = Shape::<URect>::round_rect(
            gen_id!().into(),
            crate::FILL_URECT.into(),
            0.0,
            0.0,
            Vec4::zero(),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::zero(),
        );
        let window = Window::new(
            gen_id!().into(),
            winit::window::Window::default_attributes()
                .with_title("test_blank")
                .with_resizable(true),
            Box::new(rect),
        );

        let mut hash = im::HashMap::new();
        hash.insert(window.id().clone(), Some(window));

        (0, hash)
    }

    fn call(
        &self,
        store: Self::Store,
        _: &u8,
    ) -> (Self::Store, im::HashMap<Rc<SourceID>, Option<Window>>) {
        let windows = store.1.clone();
        (store, windows)
    }
}

#[test]
fn test_basic() {
    let (mut app, event_loop): (App<u8, TestApp>, winit::event_loop::EventLoop<()>) =
        App::new(0u8, vec![], TestApp {}).unwrap();

    let proxy = event_loop.create_proxy();
    proxy.send_event(()).unwrap();
    event_loop.run_app(&mut app).unwrap();
}
