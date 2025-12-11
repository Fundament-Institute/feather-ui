// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

//! # Reactive Data-Driven UI
//!
//! Feather is a reactive data-driven UI framework that only mutates application
//! state in response to user inputs or events, using event streams and reactive
//! properties, and represents application state using persistent data
//! structures, which then efficiently render only the parts of the UI that
//! changed using either a standard GPU compositor or custom shaders.
//!
//! Examples can be found in [feather-ui/examples](feather-ui/examples), and can
//! be run via `cargo run --example <example_name>`.

//#![warn(unreachable_pub, missing_docs)]
#![allow(dead_code)]

extern crate alloc;

pub mod color;
pub mod component;
mod editor;
pub mod event;
pub mod graphics;
pub mod input;
pub mod layout;
#[cfg(feature = "lua")]
pub mod lua;
pub mod persist;
//mod propbag;
mod pool;
pub mod reactive;
pub mod render;
pub mod resource;
mod rtree;
mod shaders;
pub mod text;
pub mod util;

use crate::component::ComponentMarker;
use crate::component::window::{Window, WindowState};
use crate::graphics::Driver;
use crate::reactive::{
    DynSignal, Identity, MutableSignalProvider, Sampler, const_signal, empty_signal, map_vec,
    sample,
};
use crate::render::atlas::AtlasKind;
use crate::render::compositor::CompositorView;
use bytemuck::NoUninit;
use core::f32;
use dyn_clone::DynClone;
use eyre::OptionExt;
pub use guillotiere::euclid;
use guillotiere::euclid::{Point2D, Size2D, Vector2D};
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::any::Any;
use std::cell::RefCell;
use std::cmp::PartialEq;
use std::collections::HashMap;
use std::convert::Infallible;
use std::ffi::c_void;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use wgpu::{InstanceDescriptor, InstanceFlags};
use wide::{CmpGe, CmpGt, f32x4};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{WindowAttributes, WindowId};
pub use {cosmic_text, eyre, imbl, notify, wgpu, wide, winit};

type Mat4x4 = euclid::default::Transform3D<f32>;

#[cfg(feature = "lua")]
pub use mlua;

const MAX_ALLOCA: usize = 1 << 20;

/// While the ID scope provides a .next() method to generate a new, unique ID,
/// this macro allows you to generate a unique ID with the file name and line
/// embedded in it. This helps when debugging, because it'll tell you exactly
/// what line of code generated the element causing the problem.
///
/// # Examples:
///
/// ```
/// use feather_ui::{gen_id, ScopeID};
///
/// fn app(id: &mut ScopeID) {
///   let unique_id = gen_id!(id); // unique_id now contains the source location where gen_id! was invoked.
/// }
/// ```
#[macro_export]
macro_rules! gen_id {
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
    #[error("An index was out of range: {0}")]
    OutOfRange(usize),
    #[error("Type mismatch occurred when attempting a downcast that should never fail!")]
    RuntimeTypeMismatch,
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::FileError(value)
    }
}

/// Represents an axis that is "unsized", which is roughly equivalent to CSS
/// `auto`. It will set the size of the axis either to the size of the children,
/// if the layout has any, or to the intrinsic size of the element, if one
/// exists. Otherwise it will evaluate to 0.
pub const UNSIZED_AXIS: f32 = f32::MAX;

/// The standard base DPI, by convention, is 96, which corresponds to a scale
/// factor of 1.0 - all other DPI values are divided by this to get the
/// appropriate scale factor.
pub const BASE_DPI: RelDim = RelDim::new(96.0, 96.0);

const MINUS_BOTTOMRIGHT: f32x4 = f32x4::new([1.0, 1.0, -1.0, -1.0]);

/// This macro automates away some boilerplate necessary to make a vector of
/// children that can be passed into a component. The first argument is the
/// required layout of the parent, followed by a list of children to include (by
/// value).
///
/// # Examples
///
/// ```
/// use feather_ui::{DRect, FILL_DRECT, gen_id, color::sRGB, wide, UNSIZED_AXIS, DAbsPoint, AbsRect, APP_SOURCE_ID};
/// use feather_ui::layout::fixed;
/// use feather_ui::component::{region::Region, shape};
/// use std::sync::Arc;
///
/// let rect = shape::round_rect::<DRect>(
///     gen_id!(Arc::new(APP_SOURCE_ID)),
///     FILL_DRECT,
///     0.0,
///     0.0,
///     wide::f32x4::splat(10.0),
///     sRGB::new(0.2, 0.7, 0.4, 1.0),
///     sRGB::transparent(),
///     DAbsPoint::zero(),
/// );
/// let region = Region::<DRect>::new(
///     gen_id!(Arc::new(APP_SOURCE_ID)),
///      AbsRect::new(45.0, 45.0, 0.0, 0.0).into(),
///     feather_ui::children![fixed::Prop, rect],
/// );
/// ```
#[macro_export]
macro_rules! children {
    () => { [] };
    ($prop:path, $($param:expr),+ $(,)?) => { $crate::imbl::Vector::from_iter([$(Box::new($param) as Box<$crate::component::ChildOf<dyn $prop>>),+]) };
}

#[macro_export]
macro_rules! handlers {
    () => { [] };
    ($app:path, $($param:ident),+ $(,)?) => { Vec::from_iter([$((stringify!($param).to_string(), Box::new($param) as $crate::AppEvent<$app>)),+]) };
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
/// Represents display-independent pixels, or logical units
pub struct Logical {}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
/// Represents relative values
pub struct Relative {}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
/// Represents an actual pixel
pub struct Pixel {}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
/// Represents a combination of DIP and Pixels that have been resolved for the
/// current DPI
pub struct Resolved {}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
/// Represents potentially unsized pixels, only used in intermediate layout calculations.
pub struct Unsized {}

/// A 2D point in logical units (display-independent pixels)
pub type AbsPoint = Point2D<f32, Logical>;
/// A 2D point in physical device pixels
pub type PxPoint = Point2D<f32, Pixel>;
/// A 2D point in relative coordinates
pub type RelPoint = Point2D<f32, Relative>;
/// A 2D point in resolved physical pixels that hasn't yet been combined with
/// it's paired relative coordinates.
pub type ResPoint = Point2D<f32, Resolved>;

/// A 2D vector in logical units (display-independent pixels)
pub type AbsVector = Vector2D<f32, Logical>;
/// A 2D vector in physical device pixels
pub type PxVector = Vector2D<f32, Pixel>;
/// A 2D vector in relative coordinates
pub type RelVector = Vector2D<f32, Relative>;
/// A 2D vector in resolved physical pixels that hasn't yet been combined with
/// it's paired relative coordinates.
pub type ResVector = Vector2D<f32, Resolved>;

/// A 2D dimension (or size) in logical units (display-independent pixels)
pub type AbsDim = Size2D<f32, Logical>;
/// A 2D dimension (or size) in physical device pixels
pub type PxDim = Size2D<f32, Pixel>;
/// A 2D dimension (or size) in relative coordinates
pub type RelDim = Size2D<f32, Relative>;
/// A 2D dimension (or size) in physical pixels that could potentially be unsized.
pub type UnsizedDim = Size2D<f32, Unsized>;

/// Internal trait for "unresolving" a set of physical pixels into logical
/// units.
trait UnResolve<U> {
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

/// Internal trait for allowing conversions between foreign types that we're not
/// allowed to implement From or Into on.
trait Convert<U> {
    fn to(self) -> U;
}

impl Convert<Size2D<u32, Pixel>> for winit::dpi::PhysicalSize<u32> {
    fn to(self) -> Size2D<u32, Pixel> {
        Size2D::<u32, Pixel>::new(self.width, self.height)
    }
}

impl Convert<Size2D<u32, Logical>> for winit::dpi::LogicalSize<u32> {
    fn to(self) -> Size2D<u32, Logical> {
        Size2D::<u32, Logical>::new(self.width, self.height)
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

/// Represents a 2D Rectangle, similar to the Euclid rectangle, but SSE
/// optimized and uses a LTRB absolute representation, instead of a position and
/// a size.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Rect<U> {
    pub v: f32x4,
    #[doc(hidden)]
    pub _unit: PhantomData<U>,
}

/// This trait is used to canonicalize floats into forms suitable for hashing.
/// Because NaNs should never get this far in the pipeline, we only care about
/// the +0.0 and -0.0 problem, which can be solved because IEEE defines -0.0
/// plus +0.0 to equal +0.0.
trait Canonicalize {
    type Bits;

    fn canonical_bits(self) -> Self::Bits;
}

impl Canonicalize for f32 {
    type Bits = u32;

    fn canonical_bits(self) -> Self::Bits {
        debug_assert!(!self.is_nan()); // In debug mode, ensure this is not a NaN
        (self + 0.0).to_bits()
    }
}

impl Canonicalize for f64 {
    type Bits = u64;

    fn canonical_bits(self) -> Self::Bits {
        debug_assert!(!self.is_nan()); // In debug mode, ensure this is not a NaN
        (self + 0.0).to_bits()
    }
}

/// A 2D rectangle in logical units (display-independent pixels)
pub type AbsRect = Rect<Logical>;
/// A 2D rectangle in physical pixels
pub type PxRect = Rect<Pixel>;
/// A 2D rectangle in relative values
pub type RelRect = Rect<Relative>;
/// A 2D rectangle in resolved pixels that haven't been merged with their paired
/// relative component
pub type ResRect = Rect<Resolved>;

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

    #[inline]
    pub const fn splat(x: f32) -> Self {
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
    pub const fn offsetdim(offset: Point2D<f32, U>, dim: Size2D<f32, U>) -> Self {
        Self {
            v: f32x4::new([
                offset.x,
                offset.y,
                offset.x + dim.width,
                offset.y + dim.height,
            ]),
            _unit: PhantomData,
        }
    }

    #[inline]
    pub fn contains(&self, p: Point2D<f32, U>) -> bool {
        //let test: u32x4 = bytemuck::cast(f32x4::new([p.x, p.y, p.x,
        // p.y]).cmp_ge(self.0));

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

        // This rect is potentially degenerate, where topleft > bottomright, so we have
        // to guard against this.
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
    pub fn left(&self) -> f32 {
        self.v.as_array_ref()[0]
    }

    #[inline]
    pub fn top(&self) -> f32 {
        self.v.as_array_ref()[1]
    }

    #[inline]
    pub fn right(&self) -> f32 {
        self.v.as_array_ref()[2]
    }

    #[inline]
    pub fn bottom(&self) -> f32 {
        self.v.as_array_ref()[3]
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
    pub fn to_untyped(self) -> PxRect {
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

impl<U> AddAssign<Point2D<f32, U>> for Rect<U> {
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

impl<U> AddAssign<Vector2D<f32, U>> for Rect<U> {
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

impl<U> SubAssign<Point2D<f32, U>> for Rect<U> {
    #[inline]
    fn sub_assign(&mut self, rhs: Point2D<f32, U>) {
        self.v -= splat_point(rhs)
    }
}

impl<U> Neg for Rect<U> {
    type Output = Rect<U>;

    fn neg(self) -> Self::Output {
        Self::Output {
            v: -self.v,
            _unit: PhantomData,
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

/// A perimeter has the same top/left/right/bottom elements as a rectangle, but
/// when used in calculations, the bottom and right elements are subtracted, not
/// added.
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
    pub fn to_untyped(self) -> Perimeter<euclid::UnknownUnit> {
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

pub type PxPerimeter = Perimeter<Pixel>;

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
/// A rectangle with both pixel and display independent units, but no relative
/// component.
pub struct DAbsRect {
    dp: AbsRect,
    px: PxRect,
}

pub const ZERO_DABSRECT: DAbsRect = DAbsRect {
    dp: AbsRect::zero(),
    px: PxRect::zero(),
};

impl DAbsRect {
    fn resolve(&self, dpi: RelDim) -> PxRect {
        PxRect {
            v: self.px.v + (self.dp.v * splat_size(dpi)),
            _unit: PhantomData,
        }
    }

    fn as_perimeter(&self, dpi: RelDim) -> PxPerimeter {
        PxPerimeter {
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

impl From<PxRect> for DAbsRect {
    fn from(value: PxRect) -> Self {
        DAbsRect {
            dp: AbsRect::zero(),
            px: value,
        }
    }
}

impl Neg for DAbsRect {
    type Output = DAbsRect;

    fn neg(self) -> Self::Output {
        Self::Output {
            dp: -self.dp,
            px: -self.px,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// A point with both pixel and display independent units, but no relative
/// component. Must be constructed manually or from a [`PxPoint`] or
/// [`AbsPoint`]. This is commonly used in DPI sensitive values that could
/// theoretically have pixels, or logical units, or both, but where a relative
/// value doesn't make any sense (such as the intrinsic size of a shape).
///
/// # Examples
/// ```
/// use feather_ui::{DAbsPoint, AbsPoint, PxPoint, RelPoint};
/// let foo: DAbsPoint = AbsPoint::new(1.0,2.0).into();
///
/// let bar: DAbsPoint = PxPoint::new(2.0,3.0).into();
///
/// let foobar = foo + bar;
///
/// let test = DAbsPoint{
///     px: PxPoint::new(1.0,2.0),
///     dp: AbsPoint::new(1.0,4.0),
/// };
///
/// let bartest = bar + test;
/// ```
pub struct DAbsPoint {
    pub dp: AbsPoint,
    pub px: PxPoint,
}

impl DAbsPoint {
    pub const fn zero() -> Self {
        Self {
            dp: AbsPoint::new(0.0, 0.0),
            px: PxPoint::new(0.0, 0.0),
        }
    }

    pub const fn unit() -> Self {
        Self {
            dp: AbsPoint::new(1.0, 1.0),
            px: PxPoint::new(1.0, 1.0),
        }
    }

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

impl From<PxPoint> for DAbsPoint {
    fn from(value: PxPoint) -> Self {
        DAbsPoint {
            dp: AbsPoint::zero(),
            px: value,
        }
    }
}

impl Add<DAbsPoint> for DAbsPoint {
    type Output = DAbsPoint;

    fn add(self, rhs: DAbsPoint) -> Self::Output {
        Self::Output {
            dp: self.dp + rhs.dp.to_vector(),
            px: self.px + rhs.px.to_vector(),
        }
    }
}

impl AddAssign<DAbsPoint> for DAbsPoint {
    fn add_assign(&mut self, rhs: DAbsPoint) {
        self.dp += rhs.dp.to_vector();
        self.px += rhs.px.to_vector();
    }
}

impl Neg for DAbsPoint {
    type Output = DAbsPoint;

    fn neg(self) -> Self::Output {
        Self::Output {
            dp: -self.dp,
            px: -self.px,
        }
    }
}

#[inline]
pub fn build_aabb<U>(a: Point2D<f32, U>, b: Point2D<f32, U>) -> Rect<U> {
    Rect::<U>::corners(a.min(b), a.max(b))
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Partially resolved unified coordinate
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

impl Mul<PxDim> for UPoint {
    type Output = PxPoint;

    #[inline]
    fn mul(self, rhs: PxDim) -> Self::Output {
        let rel = self.rel();
        self.abs()
            .add_size(&Size2D::<f32, Resolved>::new(
                rel.x * rhs.width,
                rel.y * rhs.height,
            ))
            .cast_unit()
    }
}

impl Neg for UPoint {
    type Output = UPoint;

    fn neg(self) -> Self::Output {
        UPoint(-self.0)
    }
}

/// Unified Display Point with both per-pixel and display-independent pixels.
/// Unlike a Rect, must be constructed manually or from a [`PxPoint`],
/// [`AbsPoint`] or [`RelPoint`].
///
/// # Examples
/// ```
/// use feather_ui::{DPoint, AbsPoint, PxPoint, RelPoint};
/// let foo: DPoint = AbsPoint::new(1.0,2.0).into();
///
/// let bar: DPoint = PxPoint::new(2.0,3.0).into();
///
/// let foobar = foo + bar;
///
/// let test = DPoint{
///     px: PxPoint::new(1.0,2.0),
///     dp: AbsPoint::new(1.0,4.0),
///     rel: RelPoint::new(3.0,4.0),
/// };
///
/// let bartest = bar + test;
/// ```
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DPoint {
    pub dp: AbsPoint,
    pub px: PxPoint,
    pub rel: RelPoint,
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

impl From<AbsPoint> for DPoint {
    fn from(value: AbsPoint) -> Self {
        Self {
            dp: value,
            px: PxPoint::zero(),
            rel: RelPoint::zero(),
        }
    }
}

impl From<PxPoint> for DPoint {
    fn from(value: PxPoint) -> Self {
        Self {
            dp: AbsPoint::zero(),
            px: value,
            rel: RelPoint::zero(),
        }
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

impl Add<DPoint> for DPoint {
    type Output = Self;

    #[inline]
    fn add(self, rhs: DPoint) -> Self::Output {
        Self::Output {
            dp: self.dp + rhs.dp.to_vector(),
            px: self.px + rhs.px.to_vector(),
            rel: self.rel + rhs.rel.to_vector(),
        }
    }
}

impl Sub<DPoint> for DPoint {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: DPoint) -> Self::Output {
        self + (-rhs)
    }
}

impl Neg for DPoint {
    type Output = DPoint;

    fn neg(self) -> Self::Output {
        Self::Output {
            dp: -self.dp,
            px: -self.px,
            rel: -self.rel,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Partially resolved unified coordinate rectangle
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
    pub fn resolve(&self, rect: PxRect) -> PxRect {
        let ltrb = rect.v.as_array_ref();
        let topleft = f32x4::new([ltrb[0], ltrb[1], ltrb[0], ltrb[1]]);
        let bottomright = f32x4::new([ltrb[2], ltrb[3], ltrb[2], ltrb[3]]);

        PxRect {
            v: topleft + self.abs.v + self.rel.v * (bottomright - topleft),
            _unit: PhantomData,
        }
    }

    #[inline]
    pub fn to_perimeter(&self, rect: PxRect) -> PxPerimeter {
        PxPerimeter {
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

impl Mul<PxRect> for URect {
    type Output = PxRect;

    #[inline]
    fn mul(self, rhs: PxRect) -> Self::Output {
        self.resolve(rhs)
    }
}

impl Mul<PxDim> for URect {
    type Output = PxRect;

    #[inline]
    fn mul(self, rhs: PxDim) -> Self::Output {
        Self::Output {
            v: self.abs.v + self.rel.v * splat_size(rhs),
            _unit: PhantomData,
        }
    }
}

impl Neg for URect {
    type Output = URect;

    fn neg(self) -> Self::Output {
        URect {
            abs: -self.abs,
            rel: -self.rel,
        }
    }
}

/// Unified Display Rectangle with both per-pixel and display-independent
/// pixels. Can be constructed by adding together any combination of [`PxRect`],
/// [`AbsRect`] or [`RelRect`].
///
/// # Examples
/// ```
/// use feather_ui::{DRect, AbsRect, PxRect, RelRect};
/// let foo: DRect = AbsRect::new(1.0,2.0,3.0,4.0).into();
///
/// let bar = AbsRect::new(1.0,2.0,3.0,4.0) + PxRect::new(1.0,2.0,3.0,4.0);
///
/// let baz = RelRect::new(1.0,2.0,3.0,4.0) + PxRect::new(1.0,2.0,3.0,4.0);
///
/// // These can be added together because bar turned into a `DRect` from adding `PxRect`
/// // and `AbsRect` together
/// let foobar = foo + bar;
///
/// let test = DRect{
///     px: PxRect::new(1.0,2.0,3.0,4.0),
///     dp: AbsRect::new(1.0,2.0,3.0,4.0),
///     rel: RelRect::new(1.0,2.0,3.0,4.0),
/// };
///
/// let baztest = baz + test;
/// ```
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

    /// Returns the top-left corner of the unified display rectangle as a
    /// unified display point.
    pub fn topleft(&self) -> DPoint {
        DPoint {
            dp: self.dp.topleft(),
            px: self.px.topleft(),
            rel: self.rel.topleft(),
        }
    }

    /// Returns the bottom-right corner of the unified display rectangle. This
    /// is ***not*** the size of the rectangle! To get the actual size of the
    /// rectangle, you must subtract the top-left corner from the bottom-right
    /// corner, or call [`DRect::size`] which does this for you.
    pub fn bottomright(&self) -> DPoint {
        DPoint {
            dp: self.dp.bottomright(),
            px: self.px.bottomright(),
            rel: self.rel.bottomright(),
        }
    }

    /// Returns the size of the rectangle as a unified display point.
    pub fn size(&self) -> DPoint {
        self.bottomright() - self.topleft()
    }

    /// Returns a degenerate zero-sized rectangle.
    pub const fn zero() -> Self {
        Self {
            px: PxRect::zero(),
            dp: AbsRect::zero(),
            rel: RelRect::zero(),
        }
    }

    /// Returns a DRect with a relative component mapped to the entire available
    /// area. This is often used for any element that should be the same
    /// size as it's parent container.
    pub const fn fill() -> Self {
        DRect {
            px: PxRect::zero(),
            dp: AbsRect::zero(),
            rel: RelRect::unit(),
        }
    }

    /// Returns a DRect with two [`UNSIZED_AXIS`], meaning they will be set to
    /// the size of the children of the element, or the element's intrinsic
    /// size (or zero if it doesn't have any).
    pub const fn auto() -> Self {
        DRect {
            px: PxRect::zero(),
            dp: AbsRect::zero(),
            rel: RelRect::new(0.0, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),
        }
    }
}

pub const ZERO_DRECT: DRect = DRect::zero();
pub const FILL_DRECT: DRect = DRect::fill();
pub const AUTO_DRECT: DRect = DRect::auto();

impl Add<DRect> for DRect {
    type Output = Self;

    #[inline]
    fn add(self, rhs: DRect) -> Self::Output {
        Self::Output {
            dp: AbsRect {
                v: self.dp.v + rhs.dp.v,
                _unit: PhantomData,
            },
            px: PxRect {
                v: self.px.v + rhs.px.v,
                _unit: PhantomData,
            },
            rel: RelRect {
                v: self.rel.v + rhs.rel.v,
                _unit: PhantomData,
            },
        }
    }
}

impl Sub<DRect> for DRect {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: DRect) -> Self::Output {
        self + (-rhs)
    }
}

impl Neg for DRect {
    type Output = DRect;

    fn neg(self) -> Self::Output {
        Self::Output {
            dp: -self.dp,
            px: -self.px,
            rel: -self.rel,
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

impl From<PxRect> for DRect {
    fn from(value: PxRect) -> Self {
        Self {
            px: value,
            dp: AbsRect::zero(),
            rel: RelRect::zero(),
        }
    }
}

impl From<RelRect> for DRect {
    fn from(value: RelRect) -> Self {
        Self {
            px: PxRect::zero(),
            dp: AbsRect::zero(),
            rel: value,
        }
    }
}

impl<T, U: Into<DRect>> Add<U> for Rect<T>
where
    Self: Into<DRect>,
{
    type Output = DRect;

    fn add(self, rhs: U) -> Self::Output {
        self.into() + rhs.into()
    }
}

/// The Limits type represents both a minimum size and a maximum size for a
/// given unit. Adding limits together actually merges them, by taking the
/// largest minimum size and the smallest maximum size of either. You aren't
/// expected to construct this type manually, however - the Limits constructor
/// takes a range parameter to make it easier to represent the minimum and
/// maximum range of sizes you want.
///
/// # Examples
/// ```
/// use feather_ui::euclid::Size2D;
/// use feather_ui::{AbsLimits, RelLimits, Logical, Relative};
///
/// // Results in a minimum size of [-inf, 10.0] and a maximum size of [inf, 200.0]
/// let limits = AbsLimits::new(.., 10.0..200.0);
/// assert_eq!(limits.min(), Size2D::<f32, Logical>::new(f32::NEG_INFINITY, 10.0));
/// assert_eq!(limits.max(), Size2D::<f32, Logical>::new(f32::INFINITY, 200.0));
///
/// // Results in a minimum size of [-inf, -inf] and a maximum size of [1.0, inf]
/// let rlimits = RelLimits::new(..1.0, ..);
/// assert_eq!(rlimits.min(), Size2D::<f32, Relative>::new(f32::NEG_INFINITY, f32::NEG_INFINITY));
/// assert_eq!(rlimits.max(), Size2D::<f32, Relative>::new(1.0, f32::INFINITY));
///
/// // Result in a minimum size of [0.0, 10.0] and a maximum size of [inf, 100.0]
/// let merged = AbsLimits::new(0.0.., 5.0..100.0) + limits;
/// assert_eq!(merged.min(), Size2D::<f32, Logical>::new(0.0, 10.0));
/// assert_eq!(merged.max(), Size2D::<f32, Logical>::new(f32::INFINITY, 100.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Limits<U> {
    v: f32x4,
    #[doc(hidden)]
    pub _unit: PhantomData<U>,
}

pub type PxLimits = Limits<Pixel>;
pub type AbsLimits = Limits<Logical>;
pub type RelLimits = Limits<Relative>;
pub type ResLimits = Limits<Resolved>;

//pub const Unbounded: std::ops::Range<f32> = std::ops::Range
// It would be cheaper to avoid using actual infinities here but we currently
// need them to make the math work

/// Represents the default limit values for any limit type. This simply
/// represents a minimum size of [`f32::NEG_INFINITY`] and a maximum size of
/// [`f32::INFINITY`]
pub const DEFAULT_LIMITS: f32x4 = f32x4::new([
    f32::NEG_INFINITY,
    f32::NEG_INFINITY,
    f32::INFINITY,
    f32::INFINITY,
]);

/// Represents the default absolute value limit with a minimum size of
/// [`f32::NEG_INFINITY`] and a maximum size of [`f32::INFINITY`]
pub const DEFAULT_ABSLIMITS: AbsLimits = AbsLimits {
    v: DEFAULT_LIMITS,
    _unit: PhantomData,
};

/// Represents the default relative value limit with a minimum size of
/// [`f32::NEG_INFINITY`] and a maximum size of [`f32::INFINITY`]
pub const DEFAULT_RLIMITS: RelLimits = RelLimits {
    v: DEFAULT_LIMITS,
    _unit: PhantomData,
};

impl<U> Limits<U> {
    #[inline]
    const fn from_bound(bound: std::ops::Bound<&f32>, inf: f32) -> f32 {
        match bound {
            std::ops::Bound::Included(v) | std::ops::Bound::Excluded(v) => *v,
            std::ops::Bound::Unbounded => inf,
        }
    }
    pub fn new(x: impl std::ops::RangeBounds<f32>, y: impl std::ops::RangeBounds<f32>) -> Self {
        Self {
            v: f32x4::new([
                Self::from_bound(x.start_bound(), f32::NEG_INFINITY),
                Self::from_bound(y.start_bound(), f32::NEG_INFINITY),
                Self::from_bound(x.end_bound(), f32::INFINITY),
                Self::from_bound(y.end_bound(), f32::INFINITY),
            ]),
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

    #[inline]
    pub fn set_min(&mut self, bound: Size2D<f32, U>) {
        let minmax = self.v.as_array_mut();
        minmax[0] = bound.width;
        minmax[1] = bound.height;
    }

    #[inline]
    pub fn set_max(&mut self, bound: Size2D<f32, U>) {
        let minmax = self.v.as_array_mut();
        minmax[2] = bound.width;
        minmax[3] = bound.height;
    }

    /// Discard the units
    #[inline]
    pub fn to_untyped(self) -> Limits<euclid::UnknownUnit> {
        self.cast_unit()
    }

    /// Cast the unit
    #[inline]
    pub fn cast_unit<V>(self) -> Limits<V> {
        Limits::<V> {
            v: self.v,
            _unit: PhantomData,
        }
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
    pub fn resolve(&self, dpi: RelDim) -> PxLimits {
        self.px.cast_unit()
            + PxLimits {
                v: self.dp.v * splat_size(dpi),
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

impl Mul<UnsizedDim> for RelLimits {
    type Output = PxLimits;

    #[inline]
    fn mul(self, rhs: UnsizedDim) -> Self::Output {
        let (unsized_x, unsized_y) = crate::layout::check_unsized_dim(rhs);
        let minmax = self.v.as_array_ref();
        let v = f32x4::new([
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
        ]);

        Self::Output {
            v: self.v.is_finite().blend(v, self.v),
            _unit: PhantomData,
        }
    }
}

impl Mul<PxDim> for RelLimits {
    type Output = PxLimits;

    #[inline]
    fn mul(self, rhs: PxDim) -> Self::Output {
        let minmax = self.v.as_array_ref();
        let v = f32x4::new([
            minmax[0] * rhs.width,
            minmax[1] * rhs.height,
            minmax[2] * rhs.width,
            minmax[3] * rhs.height,
        ]);

        Self::Output {
            v: self.v.is_finite().blend(v, self.v),
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

/// Represents a particular layout direction, which is used in several layout
/// operations. Note that this is also used for a Grid's layout directions, but
/// [`RowDirection::TopToBottom`] doesn't make sense for a grid and instead
/// means a combination of [`RowDirection::RightToLeft`] and
/// [`RowDirection::BottomToTop`]. While this is confusing, Rust does not allow
/// us to create two variants with the same discriminator, so we can't make
/// a `RowDirection::RightToLeftAndBottomToTop` option without duplicating the
/// entire enum for Grid. [Tracking issue](https://github.com/Fundament-Institute/feather-ui/issues/159).
#[derive(
    Debug, Copy, Clone, PartialEq, Eq, Default, derive_more::TryFrom, derive_more::Display,
)]
#[try_from(repr)]
#[repr(u8)]
pub enum RowDirection {
    #[default]
    LeftToRight = 0,
    RightToLeft = 1,
    BottomToTop = 2,
    TopToBottom = 3,
}

// If a component provides a CrossReferenceDomain, it's children can register
// themselves with it. Registered children will write their fully resolved area
// to the mapping, which can then be retrieved during the render step via a
// source ID.
#[derive(Default)]
pub struct CrossReferenceDomain {
    mappings: RwLock<imbl::HashMap<Identity<Arc<dyn ComponentMarker + Send + Sync>>, PxRect>>,
}

impl CrossReferenceDomain {
    pub fn write_area(&self, target: Arc<dyn ComponentMarker + Send + Sync>, area: PxRect) {
        self.mappings.write().insert(Identity(target), area);
    }

    pub fn get_area(&self, target: Arc<dyn ComponentMarker + Send + Sync>) -> Option<PxRect> {
        self.mappings.read().get(&Identity(target)).copied()
    }

    pub fn remove_self(&self, target: Arc<dyn ComponentMarker + Send + Sync>) {
        // TODO: Is this necessary? Does it even make sense? Do you simply need to wipe
        // the mapping for every new layout instead?
        self.mappings.write().remove(&Identity(target));
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

/// Used internally for event routing.
pub type DispatchPair = (u64, Box<dyn Any>);

pub trait Dispatchable
where
    Self: Sized,
{
    const SIZE: usize;
    fn extract(self) -> DispatchPair;
    fn restore(pair: DispatchPair) -> Result<Self, Error>;
}

impl Dispatchable for Infallible {
    const SIZE: usize = 0;

    fn extract(self) -> DispatchPair {
        (0, Box::new(self))
    }

    fn restore(_: DispatchPair) -> Result<Self, Error> {
        Err(Error::Stateless)
    }
}

// This was originally supposed to use a pointer, but rust moves things all over
// the place, so a version that doesn't store the ID would have to be pinned
// (which likely isn't even possible inside an appstate).
/*pub struct StateCell<T> {
    value: T,
    id: Arc<SourceID>,
}

impl<T> StateCell<T> {
    pub fn new(v: T, id: Arc<SourceID>) -> Self {
        Self { value: v, id }
    }

    pub fn borrow_mut<'a>(&'a mut self, manager: &mut StateManager) -> &'a mut T {
        manager.mutate_id(&self.id);
        &mut self.value
    }
}

impl<T> std::borrow::Borrow<T> for StateCell<T> {
    fn borrow(&self) -> &T {
        &self.value
    }
}

impl<T> std::ops::Deref for StateCell<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.value
    }
}*/

/// `AccessCell` allows feather to track when a value passed into a function has
/// actually been changed, by tracking if a mutable borrow has been requested.
/// Like [`std::cell::RefCell`], it implements [`std::borrow::Borrow`] and
/// [`std::borrow::BorrowMut`], but also implements the [`std::ops::Deref`] and
/// [`std::ops::DerefMut`] operators so it can be used more like a smart
/// pointer.
///
/// Generally speaking, **this type should never be constructed** - it is used
/// at Feather's API boundaries where appropriate.
///
/// # Examples
///
/// ```
/// use feather_ui::AccessCell;
///
/// struct FooBar {
///   i: i32,
/// }
///
/// fn change(change: bool, mut v: AccessCell<FooBar>) {
///     if change {
///         // FooBar only marked as changed once this mutable access happens
///         v.i = 4;
///     }
/// }
/// ```
///
/// # Future-proofing
/// Currently, `AccessCell` does not attempt to determine if the new value is
/// actually *different* than what was previously stored, because this would be
/// a very expensive comparison. However, in the future, a specialization of
/// AccessCell for Persistent data structures that only marks the value as
/// changed if it is actually different when the AccessCell is dropped might be
/// implemented. As a result, you should assume that AccessCell implements
/// [`Drop`] even if it technically doesn't right now.
pub struct AccessCell<'a, 'b, T> {
    value: &'a mut T,
    changed: &'b mut bool,
}

impl<'a, 'b, T> std::borrow::BorrowMut<T> for AccessCell<'a, 'b, T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        // TODO: Later, this can be optimized for persistent data structures by cloning
        // the state here, then comparing the resulting value with the original
        // value when this cell is dropped and only setting changed to true if
        // it was actually modified.
        *self.changed = true;
        self.value
    }
}

impl<'a, 'b, T> std::borrow::Borrow<T> for AccessCell<'a, 'b, T> {
    #[inline]
    fn borrow(&self) -> &T {
        self.value
    }
}

impl<'a, 'b, T> std::ops::Deref for AccessCell<'a, 'b, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<'a, 'b, T> std::ops::DerefMut for AccessCell<'a, 'b, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        *self.changed = true;
        self.value
    }
}

#[test]
fn test_access_cell() {
    struct FooBar {
        i: i32,
    }

    fn change(change: bool, mut v: AccessCell<FooBar>) {
        if change {
            v.i = 4;
        }
    }
    let mut foobar = FooBar { i: 1 };
    let mut tracker = false;

    let accessor = AccessCell {
        value: &mut foobar,
        changed: &mut tracker,
    };
    change(false, accessor);
    assert_eq!(foobar.i, 1);
    assert_eq!(tracker, false);

    let accessor = AccessCell {
        value: &mut foobar,
        changed: &mut tracker,
    };
    change(true, accessor);
    assert_eq!(foobar.i, 4);
    assert_eq!(tracker, true);
}

/// Represents a feather application with a given `AppData` and persistent
/// outline function `O`. The outline function must always be a persistent
/// function that takes two parameters, a copy of the AppData, and a ScopeID. It
/// must always return [`OutlineReturn`].
///
/// An App creates all the top level structures needed for Feather to function.
/// It stores all wgpu, winit, and any other global state needed. See
/// [`App::new`] for examples.
pub struct App<AppData: Clone, T: 'static> {
    pub instance: wgpu::Instance,
    pub driver: std::sync::Weak<graphics::Driver>,
    pub ready: AtomicBool,
    trees: DynSignal<HashMap<Identity<Rc<Window>>, Rc<rtree::Node>>>,
    driver_init: Option<Box<dyn FnOnce(std::sync::Weak<Driver>)>>,
    #[allow(clippy::type_complexity)]
    user_events: Option<Box<dyn FnMut(&mut Self, &ActiveEventLoop, T)>>,
    window_map: HashMap<WindowId, Rc<Window>>, // We can't use WindowId everywhere because windows don't exist until we create them.
    windows: Rc<
        RefCell<
            HashMap<
                Identity<Rc<Window>>,
                (
                    WindowState,
                    reactive::Sampler<MutableSignalProvider<WindowAttributes>>,
                ),
            >,
        >,
    >,
    sampler: reactive::Sampler<dyn reactive::SignalProvider<Item = component::UI>>,
    proxy: winit::event_loop::EventLoopProxy<FeatherEvent<T>>,
}

pub struct AppDataMachine<AppData> {
    pub state: AppData,
    changed: bool,
}

#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;

//  This logic is the same for both X11 and Wayland because the any_thread
// variable is the same on both
#[cfg(target_os = "linux")]
use winit::platform::x11::EventLoopBuilderExtX11;

enum FeatherEvent<T> {
    ChangeUI,
    ChangeWindow(Rc<Window>),
    UserEvent(T),
}

impl<T> std::fmt::Debug for FeatherEvent<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChangeUI => f.debug_tuple("ChangeUI").finish(),
            Self::ChangeWindow(arg0) => f.debug_tuple("SpawnWindow").field(arg0).finish(),
            Self::UserEvent(_) => f.debug_tuple("UserEvent").finish(),
        }
    }
}

type WindowRef = Identity<Rc<Window>>;

impl<AppData: 'static + Clone, T> App<AppData, T> {
    /// Creates a new feather application. `app_state` represents the initial
    /// state of the application, and will override any value returned by
    /// `<O as FnPersist2>::init()`. `inputs` must be an array of
    /// [`AppEvent`], which can be acquired by boxing and wrapping lambdas using
    /// [`WrapEventEx`]. The `outline` must by a persistent function that
    /// takes two arguments (and implements [`FnPersist2`]): a copy
    /// of the AppData, and a ScopeID. It must always return [`OutlineReturn`].
    ///
    /// `driver_init` is an *optional* hook used to enable hotloading of
    /// resources. `user_event` is also an optional handler for any user
    /// events you generate via an event_loop proxy, which can be created with
    /// [`EventLoop::create_proxy`]. This is often used to inject appdata
    /// changes outside of the input handler.
    ///
    /// This function returns 4 values - the [`App`] object itself, the
    /// [`EventLoop`] that you must call [`EventLoop::run_app`] on to
    /// actually start the application, a channel for sending dynamic `AppEvent`
    /// handlers, and an atomic integer representing the current dynamic slot
    /// for any additional events. If your handlers are not going to change
    /// after the app has been created, you can ignore the last 2 returns.
    ///
    /// # Examples
    /// ```
    /// use feather_ui::component::window::Window;
    /// use feather_ui::persist::{ FnPersist2, FnPersistStore };
    /// use feather_ui::{ SourceID, ScopeID, App };
    /// use std::sync::Arc;
    ///
    /// struct MyState {
    ///   count: i32
    /// }
    ///
    /// struct MyApp {}
    ///
    /// impl FnPersistStore for MyApp { type Store = (); }
    ///
    /// impl FnPersist2<&MyState, ScopeID<'_>, imbl::HashMap<Arc<SourceID>, Option<Window>>> for MyApp {
    ///     fn init(&self) -> Self::Store { () }
    ///
    ///     fn call(&mut self,  _: Self::Store,  _: &MyState, _: ScopeID<'_>) -> (Self::Store, imbl::HashMap<Arc<SourceID>, Option<Window>>) {
    ///         ((), imbl::HashMap::new())
    ///     }
    /// }
    ///
    /// let (mut app, event_loop, _, _) = App::<MyState, MyApp, ()>::new(MyState { count: 0 }, Vec::new(), MyApp {}, None, None).unwrap();
    ///
    /// // You would then run the app like so (commented out because docs can't test UIs)
    /// // event_loop.run_app(&mut app).unwrap();
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn new(
        appstate: DynSignal<AppData>,
        outline: impl (Fn(&AppData) -> component::UI) + 'static,
        user_event: Option<Box<dyn FnMut(&mut Self, &ActiveEventLoop, T)>>,
        driver_init: Option<Box<dyn FnOnce(std::sync::Weak<Driver>)>>,
    ) -> eyre::Result<(Self, EventLoop<FeatherEvent<T>>)> {
        #[cfg(test)]
        let any_thread = true;
        #[cfg(not(test))]
        let any_thread = false;

        Self::new_any_thread(appstate, outline, any_thread, user_event, driver_init)
    }

    fn empty_layout_root()
    -> layout::Node<DynSignal<Size2D<u32, crate::Pixel>>, dyn layout::root::Prop> {
        let empty_child: Rc<dyn crate::layout::DynLayout<dyn layout::base::Empty>> =
            Rc::new(layout::Node::<(), dyn layout::base::Empty> {
                props: Rc::new(()),
                children: empty_signal().into(),
                renderable: None,
            });

        layout::Node {
            props: Rc::new(const_signal(Size2D::zero()).into_dyn_signal()),
            children: const_signal(empty_child).into_dyn_signal(),
            renderable: None,
        }
    }

    /// This is the same as [`App::new`], but it allows overriding the main
    /// thread detection that winit uses. This is necessary for running
    /// tests, which don't run on the main thread.
    #[allow(clippy::type_complexity)]
    pub fn new_any_thread(
        appstate: DynSignal<AppData>,
        outline: impl (Fn(&AppData) -> component::UI) + 'static,
        any_thread: bool,
        user_event: Option<Box<dyn FnMut(&mut Self, &ActiveEventLoop, T)>>,
        driver_init: Option<Box<dyn FnOnce(std::sync::Weak<Driver>)>>,
    ) -> eyre::Result<(Self, EventLoop<FeatherEvent<T>>)> {
        use reactive::SignalMap;
        // let count = AtomicU64::new(inputs.len() as u64);

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

        let outline_tree = appstate.map(outline);
        let instance = wgpu::Instance::new(&desc);
        //let on_driver = Rc::new(driver_init);
        let driver = std::sync::Weak::<graphics::Driver>::new();

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
                        "Wayland initialization failed! winit cannot automatically fall back to X11 (https://github.com/rust-windowing/winit/issues/4267). Try running the program with `WAYLAND_DISPLAY=\"\"`"
                    )
                } else {
                    e.into()
                }
            })?;

        let proxy = event_loop.create_proxy();

        let mut sampler = reactive::Sampler::new(outline_tree.clone().into());
        sampler.notify(move || {
            proxy
                .send_event(FeatherEvent::ChangeUI)
                .expect("Failed to send internal changeUI message!");
        });

        let windows = Rc::new(RefCell::new(HashMap::new()));
        let windows2 = windows.clone();

        let layouts = outline_tree.map(move |outline| {
            let windows2 = windows2.clone();
            map_vec(
                move |w| {
                    (
                        Rc::from(
                            if let Some((state, _)) = windows2.borrow().get(&Identity(w.clone())) {
                                w.as_ref().layout(state)
                            } else {
                                Box::new(Self::empty_layout_root())
                            },
                        ),
                        w.clone(),
                    )
                },
                |w| Identity(w.clone()),
                outline.children.clone(),
            )
        });

        let windows2 = windows.clone();
        let nodes = map_vec(
            move |tuple: &(
                Rc<dyn layout::Layout<Props = DynSignal<Size2D<u32, crate::Pixel>>>>,
                Rc<Window>,
            )| {
                let (v, w) = tuple;
                if let Some((state, _)) = windows2.borrow().get(&Identity((*w).clone())) {
                    let (_, mut f) = v.stage(
                        state
                            .surface_dim
                            .clone()
                            .map(|x| x.to_f32().cast_unit())
                            .into_dyn_signal(),
                        const_signal(PxLimits::default()).into_dyn_signal(),
                        state.dpi.clone(),
                    );

                    (
                        Rc::new(f(
                            const_signal(PxPoint::zero()).into_dyn_signal(),
                            state
                                .surface_dim
                                .clone()
                                .map(|x| x.to_f32())
                                .into_dyn_signal(),
                            const_signal(PxLimits::default()).into_dyn_signal(),
                        )),
                        (*w).clone(),
                    )
                } else {
                    todo!()
                }
            },
            |(_, w)| Identity(w.clone()),
            reactive::join(layouts),
        );

        let trees = nodes.map_mut::<HashMap<Identity<Rc<Window>>, Rc<rtree::Node>>>(|v, old| {
            let mut old = old.unwrap_or_default();

            // TODO: Replace with a smolset. Can be made even more efficient if made aware of the underlying persistent vector, because
            // then it can compare each chunk and use petitset because the maximum number of elements is known
            let mut exist = std::collections::HashSet::new();

            for (n, w) in v.iter() {
                old.entry(Identity(w.clone())).or_insert(n.clone());
                exist.insert(Identity(w.clone()));
            }

            old.retain(|k, _| exist.contains(k));

            old
        });

        let app = Self {
            instance,
            driver,
            trees: trees.into(),
            driver_init,
            user_events: user_event,
            ready: AtomicBool::new(false),
            window_map: HashMap::new(),
            windows,
            sampler,
            proxy: event_loop.create_proxy(),
        };

        Ok((app, event_loop))
    }

    fn update_window(&mut self, window: Rc<Window>, event_loop: &ActiveEventLoop) {
        if !self.ready.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }

        let binding = window.attributes.clone();
        let attributes = sample(&binding);
        let mut windows = self.windows.borrow_mut();
        let (w, sampler) = windows.entry(Identity(window)).or_insert_with_key(|r| {
            let state = WindowState::new(
                &attributes,
                &mut self.driver,
                &self.instance,
                event_loop,
                &mut self.driver_init,
            )
            .expect("failed to create window");
            self.window_map.insert(state.window.id(), r.0.clone());
            let mut sampler = Sampler::new(r.0.attributes.clone());
            let proxy = self.proxy.clone();
            let w = r.0.clone();
            sampler.notify(move || {
                proxy
                    .send_event(FeatherEvent::ChangeWindow(w.clone()))
                    .expect("Failed to send change window message!")
            });
            (state, sampler)
        });

        if let Some(attributes) = sampler.sample() {
            w.window.set_blur(attributes.blur);
            w.window.set_content_protected(attributes.content_protected);
            w.window.set_cursor(attributes.cursor.clone());
            w.window.set_decorations(attributes.decorations);
            w.window.set_enabled_buttons(attributes.enabled_buttons);
            w.window.set_fullscreen(attributes.fullscreen.clone());
            w.window.set_max_inner_size(attributes.max_inner_size);
            w.window.set_maximized(attributes.maximized);
            w.window.set_min_inner_size(attributes.min_inner_size);
            //w.window.set_minimized(attributes.minimized);
            w.window.set_resizable(attributes.resizable);
            w.window.set_resize_increments(attributes.resize_increments);
            w.window.set_title(&attributes.title);
            w.window.set_transparent(attributes.transparent);
            w.window.set_visible(attributes.visible);
            w.window.set_window_level(attributes.window_level);
        }
    }
}

impl<AppData: 'static + Clone, T: 'static> winit::application::ApplicationHandler<FeatherEvent<T>>
    for App<AppData, T>
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // When we resume, we treat this as a ChangeUI event
        self.ready.store(true, std::sync::atomic::Ordering::Release);
        self.user_event(event_loop, FeatherEvent::ChangeUI);
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let trees = sample(&self.trees);
        if let Some(id) = self.window_map.get(&window_id)
            && let Some(root) = trees.get(&Identity(id.clone()))
            && let Some((state, _)) = self.windows.borrow_mut().get_mut(&Identity(id.clone()))
        {
            let mut resized = false;
            let _ = match event {
                WindowEvent::CloseRequested => {
                    Window::on_window_event(state, root.clone(), event, self.driver.clone())
                }
                WindowEvent::RedrawRequested => {
                    // TODO: this doesn't properly update all window aspects
                    if let Some(driver) = self.driver.upgrade() {
                        let surface_dim = sample(&state.surface_dim).to_f32();

                        loop {
                            // Construct a default compositor view with no offset.
                            let mut viewer = CompositorView {
                                index: 0,
                                window: &mut state.compositor,
                                layer0: &mut driver.layer_composite[0].write(),
                                layer1: &mut driver.layer_composite[1].write(),
                                clipstack: &mut state.clipstack,
                                offset: PxVector::zero(),
                                surface_dim,
                                pass: 0,
                                slice: 0,
                            };

                            // Reset our layer tracker before beginning a render
                            state.layers.clear();
                            viewer.clipstack.clear();
                            if let Err(e) = root.render(
                                PxPoint::zero(),
                                &driver,
                                &mut viewer,
                                &mut state.layers,
                            ) {
                                match e {
                                    Error::ResizeTextureAtlas(layers, kind) => {
                                        // Resize the texture atlas with the requested
                                        // number of layers (the extent has already been
                                        // changed)
                                        match kind {
                                            AtlasKind::Primary => driver.atlas.write(),
                                            AtlasKind::Layer0 => driver.layer_atlas[0].write(),
                                            AtlasKind::Layer1 => driver.layer_atlas[1].write(),
                                        }
                                        .resize(
                                            &driver.device,
                                            &driver.queue,
                                            layers,
                                        );
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

                        let mut encoder =
                            driver
                                .device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("Root Encoder"),
                                });

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
                                Size2D::<u32, Pixel>::new(surface_dim.width, surface_dim.height)
                                    .to_f32(),
                            );
                        }

                        // A depth of "zero" means the window compositor, so we only go down to
                        // 1.
                        for i in (1..max_depth).rev() {
                            // Odd is layer0, even is layer1, so we add one before modulo to
                            // reverse the result
                            let idx: usize = (i + 1) % 2;
                            let mut compositor = driver.layer_composite[idx].write();
                            let atlas = driver.layer_atlas[idx].read();

                            // We create one render pass for each slice of the layer atlas
                            for slice in 0..atlas.texture.depth_or_array_layers() {
                                let name =
                                    format!("Layer {idx} (depth {i}) Atlas (slice {slice}) Pass");
                                let mut pass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: Some(&name),
                                        color_attachments: &[Some(
                                            wgpu::RenderPassColorAttachment {
                                                view: &atlas.targets[slice as usize],
                                                resolve_target: None,
                                                depth_slice: None,
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

                        state.draw(encoder);
                        driver.layer_composite[0].write().cleanup();
                        driver.layer_composite[1].write().cleanup();
                    }

                    true
                }
                WindowEvent::Resized(_) => {
                    resized = true;
                    Window::on_window_event(state, root.clone(), event, self.driver.clone())
                }
                _ => Window::on_window_event(state, root.clone(), event, self.driver.clone()),
            };
        }

        if sample(&self.sampler.inspect().children).is_empty() {
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

    fn user_event(&mut self, event_loop: &ActiveEventLoop, evt: FeatherEvent<T>) {
        match evt {
            FeatherEvent::ChangeUI => {
                // TODO: There are much more efficient ways to perform this check that can be done using purely stack-allocated storage, but we don't bother right now
                let mut exist = std::collections::HashSet::new();

                let children = reactive::sample_val(self.sampler.inspect().children.clone());
                for w in children.iter() {
                    if !self.windows.borrow().contains_key(&Identity(w.clone())) {
                        self.update_window(w.clone(), event_loop);
                    }
                    exist.insert(Identity(w.clone()));
                }

                self.windows.borrow_mut().retain(|k, _| exist.contains(k));
                self.window_map
                    .retain(|_, v| exist.contains(&Identity(v.clone())));
            }
            FeatherEvent::ChangeWindow(window) => {
                self.update_window(window, event_loop);
            }
            FeatherEvent::UserEvent(evt) => {
                if let Some(mut f) = self.user_events.take() {
                    (f)(self, event_loop, evt);
                    self.user_events.replace(f);
                }
            }
        }
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        let _ = event_loop;
    }
}

#[cfg(test)]
struct TestApp {}
/*
#[cfg(test)]
impl crate::persist::FnPersistStore for TestApp {
    type Store = (u8, OutlineReturn);
}

#[cfg(test)]
impl FnPersist2<&u8, ScopeID<'static>, OutlineReturn> for TestApp {
    fn init(&self) -> Self::Store {
        (0, imbl::HashMap::new())
    }

    fn call(
        &mut self,
        store: Self::Store,
        _: &u8,
        mut id: ScopeID<'static>,
    ) -> (Self::Store, OutlineReturn) {
        use crate::color::sRGB;
        use crate::component::shape::Shape;

        let rect = Shape::<DRect, { component::shape::ShapeKind::RoundRect as u8 }>::new(
            gen_id!(id),
            crate::FILL_DRECT.into(),
            0.0,
            0.0,
            f32x4::splat(0.0),
            sRGB::new(1.0, 0.0, 0.0, 1.0),
            sRGB::transparent(),
            DAbsPoint::zero(),
        );
        let window = Window::new(
            winit::window::Window::default_attributes()
                .with_title("test_blank")
                .with_resizable(true),
            Box::new(rect),
        );

        let mut windows = imbl::HashMap::new();
        windows.insert(window.id().clone(), Some(window));

        (store, windows)
    }
}

#[test]
fn test_basic() {
    let (mut app, event_loop, _, _) = App::<u8, TestApp, ()>::new(
        0u8,
        vec![],
        TestApp {},
        Some(Box::new(|_, evt: &ActiveEventLoop, _| evt.exit())),
        None,
    )
    .unwrap();

    let proxy = event_loop.create_proxy();
    proxy.send_event(()).unwrap();
    event_loop.run_app(&mut app).unwrap();
}
*/

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

    // Because our rectangles are technically supposed to be inclusive-exclusive,
    // they should not collide if the bottomright is coincident with the topleft.
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

#[test]
fn test_limits_add() {
    let limits = AbsLimits::new(.., 10.0..200.0);
    assert_eq!(
        limits.min(),
        Size2D::<f32, Logical>::new(f32::NEG_INFINITY, 10.0)
    );
    assert_eq!(
        limits.max(),
        Size2D::<f32, Logical>::new(f32::INFINITY, 200.0)
    );

    let rlimits = RelLimits::new(..1.0, ..);
    assert_eq!(
        rlimits.min(),
        Size2D::<f32, Relative>::new(f32::NEG_INFINITY, f32::NEG_INFINITY)
    );
    assert_eq!(
        rlimits.max(),
        Size2D::<f32, Relative>::new(1.0, f32::INFINITY)
    );

    let merged = AbsLimits::new(0.0.., 5.0..100.0) + limits;
    assert_eq!(merged.min(), Size2D::<f32, Logical>::new(0.0, 10.0));
    assert_eq!(
        merged.max(),
        Size2D::<f32, Logical>::new(f32::INFINITY, 100.0)
    );
}
