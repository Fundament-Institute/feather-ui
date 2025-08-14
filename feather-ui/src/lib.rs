// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

extern crate alloc;

pub mod color;
pub mod component;
mod editor;
pub mod graphics;
pub mod input;
pub mod layout;
#[cfg(feature = "lua")]
pub mod lua;
pub mod persist;
mod propbag;
pub mod render;
pub mod resource;
mod rtree;
mod shaders;
pub mod text;
pub mod util;

use crate::component::window::Window;
use crate::graphics::Driver;
use crate::render::atlas::AtlasKind;
use crate::render::compositor::CompositorView;
use bytemuck::NoUninit;
use component::window::WindowStateMachine;
use component::{Component, StateMachineWrapper};
use core::f32;
use dyn_clone::DynClone;
use eyre::OptionExt;
use guillotiere::euclid::{Point2D, Size2D, Vector2D};
use parking_lot::RwLock;
use persist::FnPersist;
use smallvec::SmallVec;
use std::any::Any;
use std::cell::OnceCell;
use std::cmp::PartialEq;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use std::sync::Arc;
use wgpu::{InstanceDescriptor, InstanceFlags};
use wide::f32x4;
use wide::{CmpGe, CmpGt};
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::EventLoop;
use winit::window::WindowId;
pub use {cosmic_text, guillotiere::euclid, im, notify, wgpu, wide, winit};

type Mat4x4 = euclid::default::Transform3D<f32>;

#[cfg(feature = "lua")]
pub use mlua;

#[macro_export]
macro_rules! gen_id {
    () => {
        std::sync::Arc::new($crate::SourceID::new($crate::DataID::Named(concat!(
            file!(),
            ":",
            line!()
        ))))
    };
    ($idx:expr) => {
        $idx.child($crate::DataID::Named(concat!(file!(), ":", line!())))
    };
    ($idx:expr, $i:expr) => {
        $idx.child($crate::DataID::Int($i as i64))
    };
}

use std::any::TypeId;
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Not an error, this component simply has no layout state.")]
    Stateless,
    #[error("Enum object didn't match tag {0}! Expected {1:?} but got {2:?}")]
    MismatchedEnumTag(u64, TypeId, TypeId),
    #[error("Invalid enum tag: {0}")]
    InvalidEnumTag(u64),
    #[error("Event handler didn't handle this method.")]
    UnhandledEvent,
    #[error("Frame aborted due to pending Texture Atlas resize.")]
    ResizeTextureAtlas(u32, crate::render::atlas::AtlasKind),
    #[error("Internal texture atlas reservation failure.")]
    AtlasReservationFailure,
    #[error("Internal texture atlas cache lookup failure.")]
    AtlasCacheFailure,
    #[error("Internal glyph render failure.")]
    GlyphRenderFailure,
    #[error("Internal glyph cache lookup failure.")]
    GlyphCacheFailure,
    #[error("An assumption about internal state was incorrect.")]
    InternalFailure,
    #[error("A filesystem error occurred: {0}")]
    FileError(std::io::Error),
    #[error("An error happened when loading a resource: {0:?}")]
    ResourceError(Box<dyn std::fmt::Debug + Send + Sync>),
    #[error(
        "The resource was in an unrecognized format. Are you sure you enabled the right feature flags?"
    )]
    UnknownResourceFormat,
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::FileError(value)
    }
}

pub const UNSIZED_AXIS: f32 = f32::MAX;
const MINUS_BOTTOMRIGHT: f32x4 = f32x4::new([1.0, 1.0, -1.0, -1.0]);
pub const BASE_DPI: RelDim = RelDim::new(96.0, 96.0);

#[macro_export]
macro_rules! children {
    () => { [] };
    ($prop:path, $($param:expr),+ $(,)?) => { $crate::im::Vector::from_iter([$(Some(Box::new($param) as Box<$crate::component::ChildOf<dyn $prop>>)),+]) };
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
/// Represents display-independent pixels
pub struct DIP {}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
/// Represents relative values
pub struct Relative {}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
/// Represents an actual pixel
pub struct Pixel {}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
/// Represents a combination of DIP and Pixels that have been resolved for the current DPI
pub struct Resolved {}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
/// Represents a fully evaluate rectangle in raw pixels.
pub struct Evaluated {}

pub type AbsPoint = Point2D<f32, DIP>;
pub type PxPoint = Point2D<f32, Pixel>;
pub type RelPoint = Point2D<f32, Relative>;
pub type ResPoint = Point2D<f32, Resolved>;
pub type EPoint = Point2D<f32, Evaluated>;
pub type AnyPoint = Point2D<f32, guillotiere::euclid::UnknownUnit>;

pub type AbsVector = Vector2D<f32, DIP>;
pub type PxVector = Vector2D<f32, Pixel>;
pub type RelVector = Vector2D<f32, Relative>;
pub type ResVector = Vector2D<f32, Resolved>;
pub type EVector = Vector2D<f32, Evaluated>;
pub type AnyVector = Vector2D<f32, guillotiere::euclid::UnknownUnit>;

pub type AbsDim = Size2D<f32, DIP>;
pub type PxDim = Size2D<f32, Pixel>;
pub type RelDim = Size2D<f32, Relative>;
pub type ResDim = Size2D<f32, Resolved>;
pub type EDim = Size2D<f32, Evaluated>;
pub type AnyDim = Size2D<f32, guillotiere::euclid::UnknownUnit>;

pub trait UnResolve<U> {
    fn unresolve(self, dpi: RelDim) -> U;
}

impl UnResolve<AbsVector> for PxVector {
    fn unresolve(self, dpi: RelDim) -> AbsVector {
        AbsVector::new(self.x / dpi.width, self.y / dpi.height)
    }
}

impl UnResolve<AbsPoint> for PxPoint {
    fn unresolve(self, dpi: RelDim) -> AbsPoint {
        AbsPoint::new(self.x / dpi.width, self.y / dpi.height)
    }
}

pub trait Convert<U> {
    fn to(self) -> U;
}

impl Convert<Size2D<u32, Pixel>> for winit::dpi::PhysicalSize<u32> {
    fn to(self) -> Size2D<u32, Pixel> {
        Size2D::<u32, Pixel>::new(self.width, self.height)
    }
}

impl Convert<Size2D<u32, DIP>> for winit::dpi::LogicalSize<u32> {
    fn to(self) -> Size2D<u32, DIP> {
        Size2D::<u32, DIP>::new(self.width, self.height)
    }
}

impl Convert<Point2D<f32, Pixel>> for winit::dpi::PhysicalPosition<f32> {
    fn to(self) -> Point2D<f32, Pixel> {
        Point2D::<f32, Pixel>::new(self.x, self.y)
    }
}

impl Convert<Point2D<f64, Pixel>> for winit::dpi::PhysicalPosition<f64> {
    fn to(self) -> Point2D<f64, Pixel> {
        Point2D::<f64, Pixel>::new(self.x, self.y)
    }
}

// We use our own SSE optimized Rect type instead of the euclid ones
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Rect<U> {
    pub v: f32x4,
    #[doc(hidden)]
    pub _unit: PhantomData<U>,
}

pub type AbsRect = Rect<DIP>;
pub type PxRect = Rect<Pixel>;
pub type RelRect = Rect<Relative>;
pub type ResRect = Rect<Resolved>;
pub type ERect = Rect<Evaluated>;
pub type AnyRect = Rect<guillotiere::euclid::UnknownUnit>;

unsafe impl<U: Copy + 'static> NoUninit for Rect<U> {}

pub const ZERO_RECT: AbsRect = AbsRect::zero();

impl<U> Rect<U> {
    #[inline]
    pub const fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            v: f32x4::new([left, top, right, bottom]),
            _unit: PhantomData,
        }
    }

    pub const fn broadcast(x: f32) -> Self {
        Self {
            v: f32x4::new([x, x, x, x]), // f32x4::splat isn't a constant function (for some reason)
            _unit: PhantomData,
        }
    }

    #[inline]
    pub const fn corners(topleft: Point2D<f32, U>, bottomright: Point2D<f32, U>) -> Self {
        Self {
            v: f32x4::new([topleft.x, topleft.y, bottomright.x, bottomright.y]),
            _unit: PhantomData,
        }
    }

    #[inline]
    pub fn contains(&self, p: Point2D<f32, U>) -> bool {
        //let test: u32x4 = bytemuck::cast(f32x4::new([p.x, p.y, p.x, p.y]).cmp_ge(self.0));

        f32x4::new([p.x, p.y, p.x, p.y]).cmp_ge(self.v).move_mask() == 0b0011

        /*p.x >= self.0[0]
        && p.y >= self.0[1]
        && p.x < self.0[2]
        && p.y < self.0[3]*/
    }

    #[inline]
    pub fn collides(&self, rhs: &Self) -> bool {
        let r = rhs.v.as_array_ref();
        f32x4::new([r[2], r[3], -r[0], -r[1]])
            .cmp_gt(self.v * MINUS_BOTTOMRIGHT)
            .all()

        /*rhs.0[2] > self.0[0]
        && rhs.0[3] > self.0[1]
        && rhs.0[0] < self.0[2]
        && rhs.0[1] < self.0[3]*/
    }

    #[inline]
    pub fn intersect(&self, rhs: Self) -> Self {
        let rect =
            (self.v * MINUS_BOTTOMRIGHT).fast_max(rhs.v * MINUS_BOTTOMRIGHT) * MINUS_BOTTOMRIGHT;

        // This rect is potentially degenerate, where topleft > bottomright, so we have to guard against this.
        let a = rect.to_array();
        Self {
            v: rect.fast_max(f32x4::new([a[0], a[1], a[0], a[1]])),
            _unit: PhantomData,
        }

        /*let r = rhs.0.as_array_ref();
        let l = self.0.as_array_ref();
        AbsRect::new(
            l[0].max(r[0]),
            l[1].max(r[1]),
            l[2].min(r[2]),
            l[3].min(r[3]),
        )*/
    }

    #[inline]
    pub fn extend(&self, rhs: Self) -> Self {
        /*AbsRect {
            topleft: self.topleft().min_by_component(rhs.topleft()),
            bottomright: self.bottomright().max_by_component(rhs.bottomright()),
        }*/
        Self {
            v: (self.v * MINUS_BOTTOMRIGHT).fast_min(rhs.v * MINUS_BOTTOMRIGHT) * MINUS_BOTTOMRIGHT,
            _unit: PhantomData,
        }
    }

    #[inline]
    pub fn topleft(&self) -> Point2D<f32, U> {
        let ltrb = self.v.as_array_ref();
        Point2D::new(ltrb[0], ltrb[1])
    }

    #[inline]
    pub fn set_topleft(&mut self, v: Point2D<f32, U>) {
        let ltrb = self.v.as_array_mut();
        ltrb[0] = v.x;
        ltrb[1] = v.y;
    }

    #[inline]
    pub fn bottomright(&self) -> Point2D<f32, U> {
        let ltrb = self.v.as_array_ref();
        Point2D::new(ltrb[2], ltrb[3])
    }

    #[inline]
    pub fn set_bottomright(&mut self, v: Point2D<f32, U>) {
        let ltrb = self.v.as_array_mut();
        ltrb[2] = v.x;
        ltrb[3] = v.y;
    }

    #[inline]
    pub fn dim(&self) -> Size2D<f32, U> {
        let ltrb = self.v.as_array_ref();
        Size2D::new(ltrb[2] - ltrb[0], ltrb[3] - ltrb[1])
    }

    #[inline]
    pub const fn zero() -> Self {
        Self {
            v: f32x4::ZERO,
            _unit: PhantomData,
        }
    }

    #[inline]
    pub const fn unit() -> Self {
        Self {
            v: f32x4::new([0.0, 0.0, 1.0, 1.0]),
            _unit: PhantomData,
        }
    }

    /// Discard the units
    #[inline]
    pub fn to_untyped(self) -> AnyRect {
        self.cast_unit()
    }

    /// Cast the unit
    #[inline]
    pub fn cast_unit<V>(self) -> Rect<V> {
        Rect::<V> {
            v: self.v,
            _unit: PhantomData,
        }
    }
}

impl<U> Display for Rect<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ltrb = self.v.as_array_ref();
        write!(
            f,
            "Rect[({},{});({},{})]",
            ltrb[0], ltrb[1], ltrb[2], ltrb[3]
        )
    }
}

impl<U> From<[f32; 4]> for Rect<U> {
    #[inline]
    fn from(value: [f32; 4]) -> Self {
        Self {
            v: f32x4::new(value),
            _unit: PhantomData,
        }
    }
}

#[inline]
const fn splat_point<U>(v: Point2D<f32, U>) -> f32x4 {
    f32x4::new([v.x, v.y, v.x, v.y])
}

#[inline]
const fn splat_size<U>(v: Size2D<f32, U>) -> f32x4 {
    f32x4::new([v.width, v.height, v.width, v.height])
}

impl<U> Add<Point2D<f32, U>> for Rect<U> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Point2D<f32, U>) -> Self::Output {
        Self {
            v: self.v + splat_point(rhs),
            _unit: PhantomData,
        }
    }
}

impl<U> AddAssign<Point2D<f32, U>> for AbsRect {
    #[inline]
    fn add_assign(&mut self, rhs: Point2D<f32, U>) {
        self.v += splat_point(rhs)
    }
}

impl<U> Add<Vector2D<f32, U>> for Rect<U> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Vector2D<f32, U>) -> Self::Output {
        Self {
            v: self.v + splat_point(rhs.to_point()),
            _unit: PhantomData,
        }
    }
}

impl<U> AddAssign<Vector2D<f32, U>> for AbsRect {
    #[inline]
    fn add_assign(&mut self, rhs: Vector2D<f32, U>) {
        self.v += splat_point(rhs.to_point())
    }
}

impl<U> Sub<Point2D<f32, U>> for Rect<U> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Point2D<f32, U>) -> Self::Output {
        Self {
            v: self.v - splat_point(rhs),
            _unit: PhantomData,
        }
    }
}

impl<U> SubAssign<Point2D<f32, U>> for AbsRect {
    #[inline]
    fn sub_assign(&mut self, rhs: Point2D<f32, U>) {
        self.v -= splat_point(rhs)
    }
}

impl Add<RelRect> for AbsRect {
    type Output = DRect;

    #[inline]
    fn add(self, rhs: RelRect) -> Self::Output {
        DRect {
            dp: self,
            rel: rhs,
            px: PxRect::zero(),
        }
    }
}

impl<U> From<Size2D<f32, U>> for Rect<U> {
    fn from(value: Size2D<f32, U>) -> Self {
        Self {
            v: f32x4::new([0.0, 0.0, value.width, value.height]),
            _unit: PhantomData,
        }
    }
}

/// A perimeter has the same top/left/right/bottom elements as a rectangle, but when
/// used in calculations, the bottom and right elements are subtracted, not added.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Perimeter<U> {
    pub v: f32x4,
    #[doc(hidden)]
    pub _unit: PhantomData<U>,
}

impl<U> Perimeter<U> {
    #[inline]
    pub fn topleft(&self) -> Size2D<f32, U> {
        let ltrb = self.v.as_array_ref();
        Size2D::<f32, U> {
            width: ltrb[0],
            height: ltrb[1],
            _unit: PhantomData,
        }
    }

    #[inline]
    pub fn bottomright(&self) -> Size2D<f32, U> {
        let ltrb = self.v.as_array_ref();
        Size2D::<f32, U> {
            width: ltrb[2],
            height: ltrb[3],
            _unit: PhantomData,
        }
    }

    /// Discard the units
    #[inline]
    pub fn to_untyped(self) -> AnyPerimeter {
        self.cast_unit()
    }

    /// Cast the unit
    #[inline]
    pub fn cast_unit<V>(self) -> Perimeter<V> {
        Perimeter::<V> {
            v: self.v,
            _unit: PhantomData,
        }
    }
}

pub type EPerimeter = Perimeter<Evaluated>;
pub type AnyPerimeter = Perimeter<guillotiere::euclid::UnknownUnit>;

impl<U> Add<Perimeter<U>> for Rect<U> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Perimeter<U>) -> Self::Output {
        Self {
            v: self.v + (rhs.v * MINUS_BOTTOMRIGHT),
            _unit: PhantomData,
        }
    }
}

impl<U> AddAssign<Perimeter<U>> for AbsRect {
    #[inline]
    fn add_assign(&mut self, rhs: Perimeter<U>) {
        self.v += rhs.v * MINUS_BOTTOMRIGHT
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// A rectangle with both pixel and display independent units, but no relative component.
pub struct DAbsRect {
    dp: AbsRect,
    px: PxRect,
}

pub const ZERO_DABSRECT: DAbsRect = DAbsRect {
    dp: AbsRect::zero(),
    px: PxRect::zero(),
};

impl DAbsRect {
    fn resolve(&self, dpi: RelDim) -> ERect {
        ERect {
            v: self.px.v + (self.dp.v * splat_size(dpi)),
            _unit: PhantomData,
        }
    }

    fn to_perimeter(&self, dpi: RelDim) -> Perimeter<Evaluated> {
        Perimeter::<Evaluated> {
            v: self.resolve(dpi).v,
            _unit: PhantomData,
        }
    }
}

impl From<AbsRect> for DAbsRect {
    fn from(value: AbsRect) -> Self {
        DAbsRect {
            dp: value,
            px: PxRect::zero(),
        }
    }
}

#[inline]
fn resolve_point(px: PxPoint, dp: AbsPoint, dpi: RelDim) -> ResPoint {
    ResPoint {
        x: px.x + (dp.x * dpi.width),
        y: px.y + (dp.y * dpi.height),
        _unit: PhantomData,
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// A point with both pixel and display independent units, but no relative component.
pub struct DAbsPoint {
    dp: AbsPoint,
    px: PxPoint,
}

impl DAbsPoint {
    fn resolve(&self, dpi: RelDim) -> ResPoint {
        ResPoint {
            x: self.px.x + (self.dp.x * dpi.width),
            y: self.px.y + (self.dp.y * dpi.height),
            _unit: PhantomData,
        }
        //self.px + resolve_point(self.dp, dpi)
    }
}

impl From<AbsPoint> for DAbsPoint {
    fn from(value: AbsPoint) -> Self {
        DAbsPoint {
            dp: value,
            px: PxPoint::zero(),
        }
    }
}

#[inline]
pub fn build_aabb<U>(a: Point2D<f32, U>, b: Point2D<f32, U>) -> Rect<U> {
    Rect::<U>::corners(a.min(b), a.max(b))
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Unified coordinate
pub struct UPoint(f32x4);

pub const ZERO_UPOINT: UPoint = UPoint(f32x4::ZERO);

impl UPoint {
    #[inline]
    pub const fn new(abs: ResPoint, rel: RelPoint) -> Self {
        Self(f32x4::new([abs.x, abs.y, rel.x, rel.y]))
    }
    #[inline]
    pub fn abs(&self) -> ResPoint {
        let ltrb = self.0.as_array_ref();
        ResPoint {
            x: ltrb[0],
            y: ltrb[1],
            _unit: PhantomData,
        }
    }
    #[inline]
    pub fn rel(&self) -> RelPoint {
        let ltrb = self.0.as_array_ref();
        RelPoint {
            x: ltrb[2],
            y: ltrb[3],
            _unit: PhantomData,
        }
    }
}

impl Add for UPoint {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for UPoint {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul<EDim> for UPoint {
    type Output = EPoint;

    #[inline]
    fn mul(self, rhs: EDim) -> Self::Output {
        let rel = self.rel();
        self.abs()
            .add_size(&Size2D::<f32, Resolved>::new(
                rel.x * rhs.width,
                rel.y * rhs.height,
            ))
            .cast_unit()
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DPoint {
    dp: AbsPoint,
    px: PxPoint,
    rel: RelPoint,
}

pub const ZERO_DPOINT: DPoint = DPoint {
    px: PxPoint {
        x: 0.0,
        y: 0.0,
        _unit: PhantomData,
    },
    dp: AbsPoint {
        x: 0.0,
        y: 0.0,
        _unit: PhantomData,
    },
    rel: RelPoint {
        x: 0.0,
        y: 0.0,
        _unit: PhantomData,
    },
};

impl DPoint {
    const fn resolve(&self, dpi: RelDim) -> UPoint {
        UPoint(f32x4::new([
            self.px.x + (self.dp.x * dpi.width),
            self.px.y + (self.dp.y * dpi.height),
            self.rel.x,
            self.rel.y,
        ]))
    }
}

impl From<RelPoint> for DPoint {
    fn from(value: RelPoint) -> Self {
        Self {
            dp: AbsPoint::zero(),
            px: PxPoint::zero(),
            rel: value,
        }
    }
}

impl From<AbsPoint> for DPoint {
    fn from(value: AbsPoint) -> Self {
        Self {
            dp: value,
            px: PxPoint::zero(),
            rel: RelPoint::zero(),
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Unified coordinate rectangle
pub struct URect {
    pub abs: ResRect,
    pub rel: RelRect,
}

impl URect {
    #[inline]
    pub fn topleft(&self) -> UPoint {
        let abs = self.abs.v.as_array_ref();
        let rel = self.rel.v.as_array_ref();
        UPoint(f32x4::new([abs[0], abs[1], rel[0], rel[1]]))
    }

    #[inline]
    pub fn bottomright(&self) -> UPoint {
        let abs = self.abs.v.as_array_ref();
        let rel = self.rel.v.as_array_ref();
        UPoint(f32x4::new([abs[2], abs[3], rel[2], rel[3]]))
    }

    #[inline]
    pub fn resolve(&self, rect: ERect) -> ERect {
        let ltrb = rect.v.as_array_ref();
        let topleft = f32x4::new([ltrb[0], ltrb[1], ltrb[0], ltrb[1]]);
        let bottomright = f32x4::new([ltrb[2], ltrb[3], ltrb[2], ltrb[3]]);

        ERect {
            v: topleft + self.abs.v + self.rel.v * (bottomright - topleft),
            _unit: PhantomData,
        }
    }

    #[inline]
    pub fn to_perimeter(&self, rect: ERect) -> Perimeter<Evaluated> {
        Perimeter::<Evaluated> {
            v: self.resolve(rect).v,
            _unit: PhantomData,
        }
    }
}

pub const ZERO_URECT: URect = URect {
    abs: ResRect::zero(),
    rel: RelRect::zero(),
};

pub const FILL_URECT: URect = URect {
    abs: ResRect::zero(),
    rel: RelRect::unit(),
};

pub const AUTO_URECT: URect = URect {
    abs: ResRect::zero(),
    rel: RelRect::new(0.0, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),
};

impl Mul<ERect> for URect {
    type Output = ERect;

    #[inline]
    fn mul(self, rhs: ERect) -> Self::Output {
        self.resolve(rhs)
    }
}

impl Mul<EDim> for URect {
    type Output = ERect;

    #[inline]
    fn mul(self, rhs: EDim) -> Self::Output {
        Self::Output {
            v: self.abs.v + self.rel.v * splat_size(rhs),
            _unit: PhantomData,
        }
    }
}

/// Display Rectangle with both per-pixel and display-independent pixels
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DRect {
    pub px: PxRect,
    pub dp: AbsRect,
    pub rel: RelRect,
}

impl DRect {
    fn resolve(&self, dpi: RelDim) -> URect {
        URect {
            abs: ResRect {
                v: self.px.v + (self.dp.v * splat_size(dpi)),
                _unit: PhantomData,
            },
            rel: self.rel,
        }
    }
    pub fn topleft(&self) -> DPoint {
        DPoint {
            dp: self.dp.topleft(),
            px: self.px.topleft(),
            rel: self.rel.topleft(),
        }
    }
    pub fn bottomright(&self) -> DPoint {
        DPoint {
            dp: self.dp.bottomright(),
            px: self.px.bottomright(),
            rel: self.rel.bottomright(),
        }
    }
}

pub const ZERO_DRECT: DRect = DRect {
    px: PxRect::zero(),
    dp: AbsRect::zero(),
    rel: RelRect::zero(),
};

pub const FILL_DRECT: DRect = DRect {
    px: PxRect::zero(),
    dp: AbsRect::zero(),
    rel: RelRect::new(0.0, 0.0, 1.0, 1.0),
};

pub const AUTO_DRECT: DRect = DRect {
    px: PxRect::zero(),
    dp: AbsRect::zero(),
    rel: RelRect::new(0.0, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),
};

impl From<RelRect> for DRect {
    fn from(value: RelRect) -> Self {
        Self {
            px: PxRect::zero(),
            dp: AbsRect::zero(),
            rel: value,
        }
    }
}

impl From<AbsRect> for DRect {
    fn from(value: AbsRect) -> Self {
        Self {
            px: PxRect::zero(),
            dp: value,
            rel: RelRect::zero(),
        }
    }
}

// We use our own SSE optimized Rect type instead of the euclid ones
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Limits<U> {
    pub v: f32x4,
    #[doc(hidden)]
    pub _unit: PhantomData<U>,
}

pub type PxLimits = Limits<Pixel>;
pub type AbsLimits = Limits<DIP>;
pub type RelLimits = Limits<Relative>;
pub type ResLimits = Limits<Resolved>;
pub type ELimits = Limits<Evaluated>;

// It would be cheaper to avoid using actual infinities here but we currently need them to make the math work
pub const DEFAULT_LIMITS: f32x4 = f32x4::new([
    f32::NEG_INFINITY,
    f32::NEG_INFINITY,
    f32::INFINITY,
    f32::INFINITY,
]);

pub const DEFAULT_ABSLIMITS: AbsLimits = AbsLimits {
    v: DEFAULT_LIMITS,
    _unit: PhantomData,
};

pub const DEFAULT_RLIMITS: RelLimits = RelLimits {
    v: DEFAULT_LIMITS,
    _unit: PhantomData,
};

impl<U> Limits<U> {
    pub const fn new(min: Size2D<f32, U>, max: Size2D<f32, U>) -> Self {
        Self {
            v: f32x4::new([min.width, min.height, max.width, max.height]),
            _unit: PhantomData,
        }
    }
    #[inline]
    pub fn min(&self) -> Size2D<f32, U> {
        let minmax = self.v.as_array_ref();
        Size2D::new(minmax[0], minmax[1])
    }
    #[inline]
    pub fn max(&self) -> Size2D<f32, U> {
        let minmax = self.v.as_array_ref();
        Size2D::new(minmax[2], minmax[3])
    }
}

impl<U> Default for Limits<U> {
    #[inline]
    fn default() -> Self {
        Self {
            v: DEFAULT_LIMITS,
            _unit: PhantomData,
        }
    }
}

impl<U> Add<Limits<U>> for Limits<U> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Limits<U>) -> Self::Output {
        let minmax = self.v.as_array_ref();
        let r = rhs.v.as_array_ref();

        Self {
            v: f32x4::new([
                minmax[0].max(r[0]),
                minmax[1].max(r[1]),
                minmax[2].min(r[2]),
                minmax[3].min(r[3]),
            ]),
            _unit: PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct DLimits {
    dp: AbsLimits,
    px: PxLimits,
}

pub const DEFAULT_DLIMITS: DLimits = DLimits {
    dp: AbsLimits {
        v: DEFAULT_LIMITS,
        _unit: PhantomData,
    },
    px: PxLimits {
        v: DEFAULT_LIMITS,
        _unit: PhantomData,
    },
};

impl DLimits {
    pub fn resolve(&self, dpi: RelDim) -> ELimits {
        ELimits {
            v: self.px.v + self.dp.v * splat_size(dpi),
            _unit: PhantomData,
        }
    }
}

impl From<AbsLimits> for DLimits {
    fn from(value: AbsLimits) -> Self {
        DLimits {
            dp: value,
            px: Default::default(),
        }
    }
}

impl From<PxLimits> for DLimits {
    fn from(value: PxLimits) -> Self {
        DLimits {
            dp: Default::default(),
            px: value,
        }
    }
}

impl Mul<EDim> for RelLimits {
    type Output = ELimits;

    #[inline]
    fn mul(self, rhs: EDim) -> Self::Output {
        let (unsized_x, unsized_y) = crate::layout::check_unsized_dim(rhs);
        let minmax = self.v.as_array_ref();
        Self::Output {
            v: f32x4::new([
                if unsized_x {
                    minmax[0]
                } else {
                    minmax[0] * rhs.width
                },
                if unsized_y {
                    minmax[1]
                } else {
                    minmax[1] * rhs.height
                },
                if unsized_x {
                    minmax[2]
                } else {
                    minmax[2] * rhs.width
                },
                if unsized_y {
                    minmax[3]
                } else {
                    minmax[3] * rhs.height
                },
            ]),
            _unit: PhantomData,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct UValue {
    pub abs: f32,
    pub rel: f32,
}

impl UValue {
    pub const fn resolve(&self, outer_dim: f32) -> f32 {
        if self.rel == UNSIZED_AXIS {
            UNSIZED_AXIS
        } else {
            self.abs + (self.rel * outer_dim)
        }
    }
    pub const fn is_unsized(&self) -> bool {
        self.rel == UNSIZED_AXIS
    }
}

impl From<f32> for UValue {
    fn from(value: f32) -> Self {
        Self {
            abs: value,
            rel: 0.0,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct DValue {
    pub dp: f32,
    pub px: f32,
    pub rel: f32,
}

impl DValue {
    pub const fn resolve(&self, dpi: f32) -> UValue {
        UValue {
            abs: self.px + (self.dp * dpi),
            rel: self.rel,
        }
    }
    pub const fn is_unsized(&self) -> bool {
        self.rel == UNSIZED_AXIS
    }
}

impl From<f32> for DValue {
    fn from(value: f32) -> Self {
        Self {
            dp: value,
            px: 0.0,
            rel: 0.0,
        }
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

// If a component provides a CrossReferenceDomain, it's children can register themselves with it.
// Registered children will write their fully resolved area to the mapping, which can then be
// retrieved during the render step via a source ID.
#[derive(Default)]
pub struct CrossReferenceDomain {
    mappings: RwLock<im::HashMap<Arc<SourceID>, AnyRect>>,
}

impl CrossReferenceDomain {
    pub fn write_area(&self, target: Arc<SourceID>, area: AnyRect) {
        self.mappings.write().insert(target, area);
    }

    pub fn get_area(&self, target: &Arc<SourceID>) -> Option<AnyRect> {
        self.mappings.read().get(target).copied()
    }

    pub fn remove_self(&self, target: &Arc<SourceID>) {
        // TODO: Is this necessary? Does it even make sense? Do you simply need to wipe the mapping for every new layout instead?
        self.mappings.write().remove(target);
    }
}

/// Object-safe version of Hash + PartialEq
pub trait DynHashEq: DynClone + std::fmt::Debug {
    fn dyn_hash(&self, state: &mut dyn Hasher);
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}

dyn_clone::clone_trait_object!(DynHashEq);

impl<H: Hash + PartialEq + std::cmp::Eq + Clone + std::fmt::Debug + Any> DynHashEq for H {
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

#[derive(Clone, Default, Debug)]
pub enum DataID {
    Named(&'static str),
    Owned(String),
    Int(i64),
    Other(Box<dyn DynHashEq>),
    #[default]
    None, // Marks an invalid default ID, crashes if you ever try to actually use it.
}

impl Hash for DataID {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            DataID::Named(s) => s.hash(state),
            DataID::Owned(s) => s.hash(state),
            DataID::Int(i) => i.hash(state),
            DataID::Other(hash_comparable) => hash_comparable.dyn_hash(state),
            DataID::None => {
                panic!("Invalid ID! Did you forget to initialize a component node's ID field?")
            }
        }
    }
}

impl std::cmp::Eq for DataID {}
impl PartialEq for DataID {
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

#[derive(Clone, Default, Debug)]
pub struct SourceID {
    parent: OnceCell<std::sync::Arc<SourceID>>,
    id: DataID,
}

// SAFETY: We cannot explain to the compiler that we only ever set "parent" while SourceID
// only exists in one thread, so we manually implement these unsafe traits.
unsafe impl Send for SourceID {}
unsafe impl Sync for SourceID {}

impl SourceID {
    pub const fn new(id: DataID) -> Self {
        Self {
            parent: OnceCell::new(),
            id,
        }
    }

    /// Creates a new SourceID out of the given DataID, with ourselves as its parent
    pub fn child(self: &Arc<Self>, id: DataID) -> Arc<Self> {
        Self {
            parent: self.clone().into(),
            id,
        }
        .into()
    }

    /// Creates a new, unique duplicate ID using this as the parent and the strong count
    /// of this Rc as the child DataID.
    pub fn duplicate(self: &Arc<Self>) -> Arc<Self> {
        self.child(DataID::Int(Arc::strong_count(self) as i64))
    }

    #[cfg(debug_assertions)]
    #[allow(clippy::mutable_key_type)]
    pub(crate) fn parents(self: &Arc<Self>) -> std::collections::HashSet<&Arc<Self>> {
        let mut set = std::collections::HashSet::new();

        let mut cur = self.parent.get();
        while let Some(id) = cur {
            set.insert(id);
            cur = id.parent.get();
        }
        set
    }
}
impl std::cmp::Eq for SourceID {}
impl PartialEq for SourceID {
    fn eq(&self, other: &Self) -> bool {
        if let Some(parent) = self.parent.get() {
            if let Some(pother) = other.parent.get() {
                parent == pother && self.id == other.id
            } else {
                false
            }
        } else {
            other.parent.get().is_none() && self.id == other.id
        }
    }
}
impl Hash for SourceID {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if let Some(parent) = self.parent.get() {
            parent.id.hash(state);
        }
        self.id.hash(state);
    }
}
impl Display for SourceID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(parent) = self.parent.get() {
            parent.fmt(f)?;
        }

        match (&self.id, self.parent.get().is_some()) {
            (DataID::Named(_), true) | (DataID::Owned(_), true) => f.write_str(" -> "),
            (DataID::Other(_), true) => f.write_str(" "),
            _ => Ok(()),
        }?;

        match &self.id {
            DataID::Named(s) => f.write_str(s),
            DataID::Owned(s) => f.write_str(s),
            DataID::Int(i) => write!(f, "[{i}]"),
            DataID::Other(dyn_hash_eq) => {
                let mut h = std::hash::DefaultHasher::new();
                dyn_hash_eq.dyn_hash(&mut h);
                write!(f, "{{{}}}", h.finish())
            }
            DataID::None => Ok(()),
        }
    }
}

#[derive(Clone)]
pub struct Slot(pub Arc<SourceID>, pub u64);

pub type AppEvent<State> = Box<dyn FnMut(DispatchPair, State) -> Result<State, State>>;

pub trait WrapEventEx<State: 'static + PartialEq, Input: Dispatchable + 'static> {
    fn wrap(self) -> impl FnMut(DispatchPair, State) -> Result<State, State>;
}

impl<AppData: 'static + PartialEq, Input: Dispatchable + 'static, T> WrapEventEx<AppData, Input>
    for T
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

pub trait StateMachineChild {
    #[allow(unused_variables)]
    fn init(
        &self,
        driver: &std::sync::Weak<Driver>,
    ) -> Result<Box<dyn StateMachineWrapper>, crate::Error> {
        Err(crate::Error::Stateless)
    }
    fn apply_children(
        &self,
        _: &mut dyn FnMut(&dyn StateMachineChild) -> eyre::Result<()>,
    ) -> eyre::Result<()> {
        // Default implementation assumes no children
        Ok(())
    }
    fn id(&self) -> Arc<SourceID>;
}

// This was originally supposed to use a pointer, but rust moves things all over the place, so a version that
// doesn't store the ID would have to be pinned (which likely isn't even possible inside an appstate).
pub struct StateCell<T> {
    value: T,
    id: Arc<SourceID>,
}

impl<T> StateCell<T> {
    pub fn new(v: T, id: Arc<SourceID>) -> Self {
        Self { value: v, id }
    }

    pub fn borrow(&self) -> &Self {
        self
    }

    pub fn borrow_mut<'a>(&'a mut self, manager: &mut StateManager) -> &'a mut Self {
        manager.mutate_id(&self.id);
        self
    }
}

impl<T> std::ops::Deref for StateCell<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.value
    }
}

#[derive(Default)]
pub struct StateManager {
    states: HashMap<Arc<SourceID>, Box<dyn StateMachineWrapper>>,
    pointers: HashMap<*const c_void, Arc<SourceID>>,
    changed: bool,
}

impl StateManager {
    pub fn register_pointer<T>(&mut self, p: *const T, id: Arc<SourceID>) -> Option<Arc<SourceID>> {
        let ptr = p as *const c_void;
        self.pointers.insert(ptr, id)
    }

    pub fn invalidate_pointer<T>(&mut self, p: *const T) -> Option<Arc<SourceID>> {
        let ptr = p as *const c_void;
        self.pointers.remove(&ptr)
    }

    pub fn mutate_pointer<T>(&mut self, p: *const T) {
        let ptr = p as *const c_void;
        let id = self
            .pointers
            .get(&ptr)
            .expect("Tried to mutate pointer that wasn't registered!")
            .clone();

        self.mutate_id(&id);
    }

    fn mutate_id(&mut self, id: &Arc<SourceID>) {
        if let Some(state) = self.states.get_mut(id) {
            state.set_changed(true);
            self.propagate_change(id);
        }
    }

    #[allow(dead_code)]
    fn init_default<State: 'static + component::StateMachineWrapper + Default>(
        &mut self,
        id: Arc<SourceID>,
    ) -> eyre::Result<&mut State> {
        if !self.states.contains_key(&id) {
            self.states.insert(id.clone(), Box::new(State::default()));
        }
        let v = &mut *self
            .states
            .get_mut(&id)
            .ok_or_eyre("Failed to insert state!")?
            .as_mut() as &mut dyn Any;
        v.downcast_mut().ok_or_eyre("Runtime type mismatch!")
    }
    pub fn init(&mut self, id: Arc<SourceID>, state: Box<dyn StateMachineWrapper>) {
        if !self.states.contains_key(&id) {
            self.states.insert(id.clone(), state);
        }
    }
    pub fn get<'a, State: 'static + component::StateMachineWrapper>(
        &'a self,
        id: &SourceID,
    ) -> eyre::Result<&'a State> {
        let v = self
            .states
            .get(id)
            .ok_or_eyre("State does not exist")?
            .as_ref() as &dyn Any;
        v.downcast_ref().ok_or_eyre("Runtime type mismatch!")
    }
    pub fn get_mut<'a, State: 'static + component::StateMachineWrapper>(
        &'a mut self,
        id: &SourceID,
    ) -> eyre::Result<&'a mut State> {
        let v = &mut *self
            .states
            .get_mut(id)
            .ok_or_eyre("State does not exist")?
            .as_mut() as &mut dyn Any;
        v.downcast_mut().ok_or_eyre("Runtime type mismatch!")
    }
    #[allow(clippy::borrowed_box)]
    pub fn get_trait<'a>(
        &'a self,
        id: &SourceID,
    ) -> eyre::Result<&'a Box<dyn StateMachineWrapper>> {
        self.states.get(id).ok_or_eyre("State does not exist")
    }

    fn propagate_change(&mut self, mut id: &Arc<SourceID>) {
        while let Some(parent) = id.parent.get() {
            if let Some(state) = self.states.get_mut(parent) {
                if state.changed() {
                    // If this state is marked change, then this change must have already propagated upwards and we have no more work to do.
                    return;
                }
                state.set_changed(true);
            }
            id = parent;
        }
    }

    pub fn process(
        &mut self,
        event: DispatchPair,
        slot: &Slot,
        dpi: RelDim,
        area: AnyRect,
        extent: AnyRect,
        driver: &std::sync::Weak<crate::Driver>,
    ) -> eyre::Result<()> {
        type IterTuple = (Box<dyn Any>, u64, Option<Slot>);

        // We use smallvec here so we can satisfy the borrow checker without making yet another heap allocation in most cases
        let iter: SmallVec<[IterTuple; 2]> = {
            let state = self.states.get_mut(&slot.0).ok_or_eyre("Invalid slot")?;
            let v = state.process(event, slot.1, dpi, area, extent, driver)?;
            if state.changed() {
                self.changed = true;
                self.propagate_change(&slot.0);
            }
            let state = self.states.get(&slot.0).unwrap();

            v.into_iter()
                .map(|(i, e)| (e, i, state.output_slot(i.ilog2() as usize).unwrap().clone()))
        }
        .collect();

        for (e, index, slot) in iter {
            if let Some(s) = slot.as_ref() {
                self.process((index, e), s, dpi, area, extent, driver)?;
            }
        }
        Ok(())
    }

    fn init_child(
        &mut self,
        target: &dyn StateMachineChild,
        driver: &std::sync::Weak<Driver>,
    ) -> eyre::Result<()> {
        if !self.states.contains_key(&target.id()) {
            match target.init(driver) {
                Ok(v) => self.init(target.id().clone(), v),
                Err(Error::Stateless) => (),
                Err(e) => return Err(e.into()),
            };
        }

        target.apply_children(&mut |child| self.init_child(child, driver))
    }
}

#[allow(clippy::declare_interior_mutable_const)]
pub const APP_SOURCE_ID: SourceID = SourceID {
    parent: OnceCell::new(),
    id: DataID::Named("__fg_AppData_ID__"),
};

pub struct App<AppData, O: FnPersist<AppData, im::HashMap<Arc<SourceID>, Option<Window>>>> {
    pub instance: wgpu::Instance,
    pub driver: std::sync::Weak<graphics::Driver>,
    state: StateManager,
    store: Option<O::Store>,
    outline: O,
    _parents: BTreeMap<DataID, DataID>,
    root: component::Root, // Root component node containing all windows
    driver_init: Option<Box<dyn FnOnce(std::sync::Weak<Driver>) + 'static>>,
}

struct AppDataMachine<AppData> {
    pub state: Option<AppData>,
    input: Vec<AppEvent<AppData>>,
    changed: bool,
}

impl<AppData: 'static + PartialEq> StateMachineWrapper for AppDataMachine<AppData> {
    fn output_slot(&self, _: usize) -> eyre::Result<&Option<Slot>> {
        Ok(&None)
    }

    fn input_mask(&self) -> u64 {
        0
    }

    fn changed(&self) -> bool {
        self.changed
    }

    fn set_changed(&mut self, changed: bool) {
        self.changed = changed;
    }

    fn process(
        &mut self,
        input: DispatchPair,
        index: u64,
        _: RelDim,
        _: AnyRect,
        _: AnyRect,
        _: &std::sync::Weak<crate::Driver>,
    ) -> eyre::Result<SmallVec<[DispatchPair; 1]>> {
        let f = self
            .input
            .get_mut(index as usize)
            .ok_or_eyre("index out of bounds")?;
        let processed = match f(input, self.state.take().unwrap()) {
            Ok(s) | Err(s) => Some(s),
        };
        // If it actually changed, set the change marker
        self.changed |= processed != self.state;
        self.state = processed;
        Ok(SmallVec::new())
    }
}
#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;

//  This logic is the same for both X11 and Wayland because the any_thread variable is the same on both
#[cfg(target_os = "linux")]
use winit::platform::x11::EventLoopBuilderExtX11;

impl<
    AppData: PartialEq + 'static,
    O: FnPersist<AppData, im::HashMap<Arc<SourceID>, Option<Window>>>,
> App<AppData, O>
{
    pub fn new<T: 'static>(
        app_state: AppData,
        inputs: Vec<AppEvent<AppData>>,
        outline: O,
        driver_init: impl FnOnce(std::sync::Weak<Driver>) + 'static,
    ) -> eyre::Result<(Self, EventLoop<T>)> {
        #[cfg(test)]
        let any_thread = true;
        #[cfg(not(test))]
        let any_thread = false;

        Self::new_any_thread(app_state, inputs, outline, any_thread, driver_init)
    }

    pub fn new_any_thread<T: 'static>(
        app_state: AppData,
        inputs: Vec<AppEvent<AppData>>,
        outline: O,
        any_thread: bool,
        driver_init: impl FnOnce(std::sync::Weak<Driver>) + 'static,
    ) -> eyre::Result<(Self, EventLoop<T>)> {
        let mut manager: StateManager = Default::default();
        manager.init(
            Arc::new(APP_SOURCE_ID),
            Box::new(AppDataMachine {
                input: inputs,
                state: Some(app_state),
                changed: true,
            }),
        );

        #[cfg(target_os = "windows")]
        let event_loop = EventLoop::with_user_event()
            .with_any_thread(any_thread)
            .with_dpi_aware(true)
            .build()?;
        #[cfg(not(target_os = "windows"))]
        let event_loop = EventLoop::with_user_event()
            .with_any_thread(any_thread)
            .build()
            .map_err(|e| {
                if e.to_string()
                    .eq_ignore_ascii_case("Could not find wayland compositor")
                {
                    eyre::eyre!(
                        "Wayland initialization failed! winit cannot automatically fall back to X11 (). Try running the program with `WAYLAND_DISPLAY=\"\"`"
                    )
                } else {
                    e.into()
                }
            })?;

        #[cfg(debug_assertions)]
        let desc = InstanceDescriptor {
            flags: InstanceFlags::debugging(),
            ..Default::default()
        };
        #[cfg(not(debug_assertions))]
        let desc = InstanceDescriptor {
            flags: InstanceFlags::DISCARD_HAL_LABELS,
            ..Default::default()
        };

        Ok((
            Self {
                instance: wgpu::Instance::new(&desc),
                driver: std::sync::Weak::<graphics::Driver>::new(),
                store: None,
                outline,
                state: manager,
                _parents: Default::default(),
                root: component::Root::new(),
                driver_init: Some(Box::new(driver_init)),
            },
            event_loop,
        ))
    }

    #[allow(clippy::borrow_interior_mutable_const)]
    fn update_outline(&mut self, event_loop: &ActiveEventLoop, store: O::Store) {
        let app_state: &AppDataMachine<AppData> = self.state.get(&APP_SOURCE_ID).unwrap();
        let (store, windows) = self.outline.call(store, app_state.state.as_ref().unwrap());
        debug_assert!(
            APP_SOURCE_ID.parent.get().is_none(),
            "Something set the APP_SOURCE parent! This should never happen!"
        );

        self.store.replace(store);
        self.root.children = windows;
        #[cfg(debug_assertions)]
        self.root.validate_ids().unwrap();

        let (root, manager, graphics, instance, driver_init) = (
            &mut self.root,
            &mut self.state,
            &mut self.driver,
            &mut self.instance,
            &mut self.driver_init,
        );

        root.layout_all(manager, graphics, driver_init, instance, event_loop)
            .unwrap();

        root.stage_all(manager).unwrap();
    }
}

impl<
    AppData: PartialEq + 'static,
    T: 'static,
    O: FnPersist<AppData, im::HashMap<Arc<SourceID>, Option<Window>>>,
> winit::application::ApplicationHandler<T> for App<AppData, O>
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // If this is our first resume, call the start function that can create the necessary graphics context
        let store = self.store.take();
        self.update_outline(event_loop, store.unwrap_or_else(|| O::init(&self.outline)));
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
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
                    WindowEvent::CloseRequested => Window::on_window_event(
                        window.id(),
                        rtree,
                        event,
                        &mut self.state,
                        self.driver.clone(),
                    )
                    .inspect_err(|_| {
                        delete = Some(window.id());
                    }),
                    WindowEvent::RedrawRequested => {
                        if let Ok(state) = self.state.get_mut::<WindowStateMachine>(&window.id())
                            && let Some(driver) = self.driver.upgrade()
                        {
                            if let Some(staging) = root.staging.as_ref() {
                                let inner = state.state.as_mut().unwrap();
                                let surface_dim = inner.surface_dim().to_f32();

                                loop {
                                    // Construct a default compositor view with no offset.
                                    let mut viewer = CompositorView {
                                        index: 0,
                                        window: &mut inner.compositor,
                                        layer0: &mut driver.layer_composite[0].write(),
                                        layer1: &mut driver.layer_composite[1].write(),
                                        clipstack: &mut inner.clipstack,
                                        offset: AnyVector::zero(),
                                        surface_dim,
                                        pass: 0,
                                        slice: 0,
                                    };

                                    // Reset our layer tracker before beginning a render
                                    inner.layers.clear();
                                    viewer.clipstack.clear();
                                    if let Err(e) = staging.render(
                                        EPoint::zero(),
                                        &driver,
                                        &mut viewer,
                                        &mut inner.layers,
                                    ) {
                                        match e {
                                            Error::ResizeTextureAtlas(layers, kind) => {
                                                // Resize the texture atlas with the requested number of layers (the extent has already been changed)
                                                match kind {
                                                    AtlasKind::Primary => driver.atlas.write(),
                                                    AtlasKind::Layer0 => {
                                                        driver.layer_atlas[0].write()
                                                    }
                                                    AtlasKind::Layer1 => {
                                                        driver.layer_atlas[1].write()
                                                    }
                                                }
                                                .resize(&driver.device, &driver.queue, layers);
                                                viewer.window.cleanup();
                                                viewer.layer0.cleanup();
                                                viewer.layer1.cleanup();
                                                continue; // Retry frame
                                            }
                                            e => panic!("Fatal draw error: {e}"),
                                        }
                                    }
                                    break;
                                }
                            }

                            let mut encoder = driver.device.create_command_encoder(
                                &wgpu::CommandEncoderDescriptor {
                                    label: Some("Root Encoder"),
                                },
                            );

                            driver.atlas.write().process_mipmaps(&driver, &mut encoder);
                            driver.atlas.read().draw(&driver, &mut encoder);

                            let max_depth = driver.layer_composite[0]
                                .read()
                                .segments
                                .len()
                                .max(driver.layer_composite[1].read().segments.len());

                            for i in 0..2 {
                                let surface_dim = driver.layer_atlas[i].read().texture.size();
                                driver.layer_composite[i].write().prepare(
                                    &driver,
                                    &mut encoder,
                                    Size2D::<u32, Pixel>::new(
                                        surface_dim.width,
                                        surface_dim.height,
                                    )
                                    .to_f32(),
                                );
                            }

                            // A depth of "zero" means the window compositor, so we only go down to 1.
                            for i in (1..max_depth).rev() {
                                // Odd is layer0, even is layer1, so we add one before modulo to reverse the result
                                let idx: usize = (i + 1) % 2;
                                let mut compositor = driver.layer_composite[idx].write();
                                let atlas = driver.layer_atlas[idx].read();

                                // We create one render pass for each slice of the layer atlas
                                for slice in 0..atlas.texture.depth_or_array_layers() {
                                    let name = format!(
                                        "Layer {idx} (depth {i}) Atlas (slice {slice}) Pass"
                                    );
                                    let mut pass =
                                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                            label: Some(&name),
                                            color_attachments: &[Some(
                                                wgpu::RenderPassColorAttachment {
                                                    view: &atlas.targets[slice as usize],
                                                    resolve_target: None,
                                                    ops: wgpu::Operations {
                                                        load: if true {
                                                            wgpu::LoadOp::Clear(
                                                                wgpu::Color::TRANSPARENT,
                                                            )
                                                        } else {
                                                            wgpu::LoadOp::Load
                                                        },
                                                        store: wgpu::StoreOp::Store,
                                                    },
                                                },
                                            )],
                                            depth_stencil_attachment: None,
                                            timestamp_writes: None,
                                            occlusion_query_set: None,
                                        });

                                    pass.set_viewport(
                                        0.0,
                                        0.0,
                                        atlas.texture.width() as f32,
                                        atlas.texture.height() as f32,
                                        0.0,
                                        1.0,
                                    );

                                    compositor.draw(&driver, &mut pass, i as u8, slice as u8);
                                }
                            }

                            state.state.as_mut().unwrap().draw(encoder);
                            driver.layer_composite[0].write().cleanup();
                            driver.layer_composite[1].write().cleanup();
                        }

                        Ok(())
                    }
                    WindowEvent::Resized(_) => {
                        resized = true;
                        Window::on_window_event(
                            window.id(),
                            rtree,
                            event,
                            &mut self.state,
                            self.driver.clone(),
                        )
                    }
                    _ => Window::on_window_event(
                        window.id(),
                        rtree,
                        event,
                        &mut self.state,
                        self.driver.clone(),
                    ),
                };

                if self.state.changed || resized {
                    self.state.changed = false;
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
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let _ = (event_loop, device_id, event);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, _: T) {
        event_loop.exit();
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        let _ = event_loop;
    }
}

#[cfg(test)]
struct TestApp {}

#[cfg(test)]
impl FnPersist<u8, im::HashMap<Arc<SourceID>, Option<Window>>> for TestApp {
    type Store = (u8, im::HashMap<Arc<SourceID>, Option<Window>>);

    fn init(&self) -> Self::Store {
        use crate::color::sRGB;
        use crate::component::shape::Shape;
        let rect = Shape::<DRect, { component::shape::ShapeKind::RoundRect as u8 }>::new(
            gen_id!(),
            crate::FILL_DRECT.into(),
            0.0,
            0.0,
            [0.0, 0.0, 0.0, 0.0],
            sRGB::new(1.0, 0.0, 0.0, 1.0),
            sRGB::transparent(),
        );
        let window = Window::new(
            gen_id!(),
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
        &mut self,
        store: Self::Store,
        _: &u8,
    ) -> (Self::Store, im::HashMap<Arc<SourceID>, Option<Window>>) {
        let windows = store.1.clone();
        (store, windows)
    }
}

#[test]
fn test_basic() {
    let (mut app, event_loop): (App<u8, TestApp>, EventLoop<()>) =
        App::new(0u8, vec![], TestApp {}, |_| ()).unwrap();

    let proxy = event_loop.create_proxy();
    proxy.send_event(()).unwrap();
    event_loop.run_app(&mut app).unwrap();
}

#[test]
fn test_absrect_contain() {
    let target = AbsRect::new(0.0, 0.0, 2.0, 2.0);

    for x in 0..=2 {
        for y in 0..=2 {
            if x == 2 || y == 2 {
                assert!(!target.contains(AbsPoint::new(x as f32, y as f32)));
            } else {
                assert!(
                    target.contains(AbsPoint::new(x as f32, y as f32)),
                    "{x} {y} not inside {target}"
                );
            }
        }
    }

    assert!(target.contains(AbsPoint::new(1.999, 1.999)));

    for y in -1..=3 {
        assert!(!target.contains(AbsPoint::new(-1.0, y as f32)));
        assert!(!target.contains(AbsPoint::new(3.0, y as f32)));
        assert!(!target.contains(AbsPoint::new(3000000.0, y as f32)));
    }

    for x in -1..=3 {
        assert!(!target.contains(AbsPoint::new(x as f32, -1.0)));
        assert!(!target.contains(AbsPoint::new(x as f32, 3.0)));
        assert!(!target.contains(AbsPoint::new(x as f32, -3000000.0)));
    }
}

#[test]
fn test_absrect_collide() {
    let target = AbsRect::new(0.0, 0.0, 4.0, 4.0);

    for l in 0..=3 {
        for t in 0..=3 {
            for r in 1..=4 {
                for b in 1..=4 {
                    let rhs = AbsRect::new(l as f32, t as f32, r as f32, b as f32);
                    assert!(
                        target.collides(&rhs),
                        "{target} not detected as touching {rhs}"
                    );
                }
            }
        }
    }

    for l in -2..=3 {
        for t in -2..=3 {
            for r in 1..=4 {
                for b in 1..=4 {
                    assert!(target.collides(&AbsRect::new(l as f32, t as f32, r as f32, b as f32)));
                }
            }
        }
    }

    for l in 1..=3 {
        for t in 1..=3 {
            for r in 3..=6 {
                if r > t {
                    for b in 3..=6 {
                        if b > t {
                            let rhs = AbsRect::new(l as f32, t as f32, r as f32, b as f32);
                            assert!(
                                target.collides(&rhs),
                                "{target} not detected as touching {rhs}"
                            );
                        }
                    }
                }
            }
        }
    }

    assert!(!target.collides(&AbsRect::new(1.0, 4.0, 5.0, 5.0)));

    // Because our rectangles are technically supposed to be inclusive-exclusive, they should not collide if the bottomright is coincident with the topleft.
    assert!(!target.collides(&AbsRect::new(4.0, 4.0, 5.0, 5.0)));
    assert!(!target.collides(&AbsRect::new(4.0, 0.0, 5.0, 4.0)));
    assert!(!target.collides(&AbsRect::new(0.0, 4.0, 4.0, 5.0)));

    assert!(!target.collides(&AbsRect::new(-1.0, -1.0, 0.0, 0.0)));
    assert!(!target.collides(&AbsRect::new(-1.0, 0.0, 0.0, 4.0)));
    assert!(!target.collides(&AbsRect::new(0.0, -1.0, 4.0, 0.0)));
}

#[test]
fn test_absrect_intersect() {
    let target = AbsRect::new(0.0, 0.0, 4.0, 4.0);

    assert!(target.intersect(AbsRect::new(2.0, 2.0, 6.0, 6.0)) == AbsRect::new(2.0, 2.0, 4.0, 4.0));
    assert!(
        target.intersect(AbsRect::new(-2.0, -2.0, 2.0, 2.0)) == AbsRect::new(0.0, 0.0, 2.0, 2.0)
    );

    assert!(
        target.intersect(AbsRect::new(-2.0, -2.0, -1.0, -1.0)) == AbsRect::new(0.0, 0.0, 0.0, 0.0)
    );

    assert!(target.intersect(AbsRect::new(6.0, 6.0, 8.0, 8.0)) == AbsRect::new(6.0, 6.0, 6.0, 6.0));
}

#[test]
fn test_absrect_extend() {
    let target = AbsRect::new(0.0, 0.0, 4.0, 4.0);

    assert!(target.extend(AbsRect::new(2.0, 2.0, 6.0, 6.0)) == AbsRect::new(0.0, 0.0, 6.0, 6.0));
    assert!(
        target.extend(AbsRect::new(-2.0, -2.0, 2.0, 2.0)) == AbsRect::new(-2.0, -2.0, 4.0, 4.0)
    );
}
