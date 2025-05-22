// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

extern crate alloc;

pub mod component;
pub mod input;
pub mod layout;
pub mod lua;
pub mod persist;
mod propbag;
mod render;
mod rtree;
mod shaders;

use crate::component::window::Window;
use component::window::WindowStateMachine;
use component::{Component, StateMachineWrapper};
use core::cell::Cell;
use core::f32;
use dyn_clone::DynClone;
use eyre::OptionExt;
pub use glyphon::Wrap;
use once_cell::unsync::OnceCell;
use persist::FnPersist;
use shaders::ShaderCache;
use smallvec::SmallVec;
use std::any::Any;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hasher;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use ultraviolet::f32x4;
use ultraviolet::vec::Vec2;
use wide::CmpLe;
use winit::window::WindowId;

/// Allocates `&[T]` on stack space.
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
    #[error("Not an error, this component simply has no layout state.")]
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
const MINUS_BOTTOMRIGHT: f32x4 = f32x4::new([1.0, 1.0, -1.0, -1.0]);

#[derive(Copy, Clone, Debug, Default)]
pub struct AbsDim(Vec2);

impl From<AbsDim> for Vec2 {
    fn from(val: AbsDim) -> Self {
        val.0
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Absolutely positioned rectangle
pub struct AbsRect(f32x4);

pub const ZERO_RECT: AbsRect = AbsRect(f32x4::ZERO);

impl AbsRect {
    #[inline]
    pub const fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self(f32x4::new([left, top, right, bottom]))
    }
    #[inline]
    pub const fn new_corners(topleft: Vec2, bottomright: Vec2) -> Self {
        Self(f32x4::new([
            topleft.x,
            topleft.y,
            bottomright.x,
            bottomright.y,
        ]))
    }

    #[inline]
    pub fn contains(&self, p: Vec2) -> bool {
        (self.0 * MINUS_BOTTOMRIGHT)
            .cmp_le(f32x4::new([p.x, p.y, -p.x, -p.y]))
            .all()

        /*p.x >= self.0[0]
        && p.y >= self.0[1]
        && p.x <= self.0[2]
        && p.y <= self.0[3]*/
    }

    #[inline]
    pub fn extend(&self, rhs: AbsRect) -> AbsRect {
        /*AbsRect {
            topleft: self.topleft().min_by_component(rhs.topleft()),
            bottomright: self.bottomright().max_by_component(rhs.bottomright()),
        }*/
        AbsRect(
            (self.0 * MINUS_BOTTOMRIGHT).fast_min(rhs.0 * MINUS_BOTTOMRIGHT) * MINUS_BOTTOMRIGHT,
        )
    }

    #[inline]
    pub fn topleft(&self) -> Vec2 {
        let ltrb = self.0.as_array_ref();
        Vec2 {
            x: ltrb[0],
            y: ltrb[1],
        }
    }

    #[inline]
    pub fn set_topleft(&mut self, v: Vec2) {
        let ltrb = self.0.as_array_mut();
        ltrb[0] = v.x;
        ltrb[1] = v.y;
    }

    #[inline]
    pub fn bottomright(&self) -> Vec2 {
        let ltrb = self.0.as_array_ref();
        Vec2 {
            x: ltrb[2],
            y: ltrb[3],
        }
    }

    #[inline]
    pub fn set_bottomright(&mut self, v: Vec2) {
        let ltrb = self.0.as_array_mut();
        ltrb[2] = v.x;
        ltrb[3] = v.y;
    }

    #[inline]
    pub fn dim(&self) -> AbsDim {
        let ltrb = self.0.as_array_ref();
        AbsDim(Vec2 {
            x: ltrb[2] - ltrb[0],
            y: ltrb[3] - ltrb[1],
        })
    }
}

impl From<[f32; 4]> for AbsRect {
    #[inline]
    fn from(value: [f32; 4]) -> Self {
        Self(f32x4::new(value))
    }
}

#[inline]
fn splat_vec2(v: Vec2) -> f32x4 {
    f32x4::new([v.x, v.y, v.x, v.y])
}

impl Add<Vec2> for AbsRect {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Vec2) -> Self::Output {
        Self(self.0 + splat_vec2(rhs))
    }
}

impl Add<RelRect> for AbsRect {
    type Output = URect;

    #[inline]
    fn add(self, rhs: RelRect) -> Self::Output {
        URect {
            abs: self,
            rel: rhs,
        }
    }
}

impl AddAssign<Vec2> for AbsRect {
    #[inline]
    fn add_assign(&mut self, rhs: Vec2) {
        self.0 += splat_vec2(rhs)
    }
}

impl Sub<Vec2> for AbsRect {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Vec2) -> Self::Output {
        Self(self.0 - splat_vec2(rhs))
    }
}

impl SubAssign<Vec2> for AbsRect {
    #[inline]
    fn sub_assign(&mut self, rhs: Vec2) {
        self.0 -= splat_vec2(rhs)
    }
}

impl From<AbsDim> for AbsRect {
    fn from(value: AbsDim) -> Self {
        Self(f32x4::new([0.0, 0.0, value.0.x, value.0.y]))
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// A rectangle with both pixel and display independent units, but no relative component.
pub struct DAbsRect {
    dp: AbsRect,
    px: AbsRect,
}

pub const ZERO_DABSRECT: DAbsRect = DAbsRect {
    dp: ZERO_RECT,
    px: ZERO_RECT,
};

impl DAbsRect {
    fn resolve(&self, dpi: Vec2) -> AbsRect {
        AbsRect(self.px.0 + (self.dp.0 * splat_vec2(dpi)))
    }
}

impl From<AbsRect> for DAbsRect {
    fn from(value: AbsRect) -> Self {
        DAbsRect {
            dp: value,
            px: ZERO_RECT,
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
pub struct RelRect(f32x4);

pub const ZERO_RELRECT: RelRect = RelRect(f32x4::ZERO);

impl RelRect {
    #[inline]
    pub const fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self(f32x4::new([left, top, right, bottom]))
    }

    #[inline]
    pub fn topleft(&self) -> Vec2 {
        let ltrb = self.0.as_array_ref();
        Vec2 {
            x: ltrb[0],
            y: ltrb[1],
        }
    }
    #[inline]
    pub fn bottomright(&self) -> Vec2 {
        let ltrb = self.0.as_array_ref();
        Vec2 {
            x: ltrb[2],
            y: ltrb[3],
        }
    }
}

impl Add<AbsRect> for RelRect {
    type Output = URect;

    #[inline]
    fn add(self, rhs: AbsRect) -> Self::Output {
        rhs + self
    }
}

#[inline]
pub fn build_aabb(a: Vec2, b: Vec2) -> AbsRect {
    AbsRect::new_corners(a.min_by_component(b), a.max_by_component(b))
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Unified coordinate
pub struct UPoint(f32x4);

pub const ZERO_UPOINT: UPoint = UPoint(f32x4::ZERO);

impl UPoint {
    #[inline]
    pub const fn new(abs: Vec2, rel: RelPoint) -> Self {
        Self(f32x4::new([abs.x, abs.y, rel.0.x, rel.0.y]))
    }
    #[inline]
    pub fn abs(&self) -> Vec2 {
        let ltrb = self.0.as_array_ref();
        Vec2 {
            x: ltrb[0],
            y: ltrb[1],
        }
    }
    #[inline]
    pub fn rel(&self) -> RelPoint {
        let ltrb = self.0.as_array_ref();
        RelPoint(Vec2 {
            x: ltrb[2],
            y: ltrb[3],
        })
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

impl Mul<AbsDim> for UPoint {
    type Output = Vec2;

    #[inline]
    fn mul(self, rhs: AbsDim) -> Self::Output {
        self.abs() + (self.rel().0 * rhs.0)
    }
}

impl Mul<AbsRect> for UPoint {
    type Output = Vec2;

    #[inline]
    fn mul(self, rhs: AbsRect) -> Self::Output {
        self * rhs.dim()
    }
}

impl From<RelPoint> for UPoint {
    fn from(value: RelPoint) -> Self {
        Self(f32x4::new([0.0, 0.0, value.0.x, value.0.y]))
    }
}

impl From<Vec2> for UPoint {
    fn from(value: Vec2) -> Self {
        Self(f32x4::new([value.x, value.y, 0.0, 0.0]))
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DPoint {
    dp: Vec2,
    px: Vec2,
    rel: RelPoint,
}

pub const ZERO_DPOINT: DPoint = DPoint {
    px: ZERO_POINT,
    dp: ZERO_POINT,
    rel: RelPoint(ZERO_POINT),
};

impl DPoint {
    fn resolve(&self, dpi: Vec2) -> UPoint {
        UPoint(f32x4::new([
            self.px.x + (self.dp.x * dpi.x),
            self.px.y + (self.dp.y * dpi.y),
            self.rel.0.x,
            self.rel.0.y,
        ]))
    }
}

impl From<RelPoint> for DPoint {
    fn from(value: RelPoint) -> Self {
        Self {
            dp: ZERO_POINT,
            px: ZERO_POINT,
            rel: value,
        }
    }
}

impl From<Vec2> for DPoint {
    fn from(value: Vec2) -> Self {
        Self {
            dp: value,
            px: ZERO_POINT,
            rel: RelPoint(ZERO_POINT),
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
/// Unified coordinate rectangle
pub struct URect {
    pub abs: AbsRect,
    pub rel: RelRect,
}

impl URect {
    pub fn topleft(&self) -> UPoint {
        let abs = self.abs.0.as_array_ref();
        let rel = self.rel.0.as_array_ref();
        UPoint(f32x4::new([abs[0], abs[1], rel[0], rel[1]]))
    }
    pub fn bottomright(&self) -> UPoint {
        let abs = self.abs.0.as_array_ref();
        let rel = self.rel.0.as_array_ref();
        UPoint(f32x4::new([abs[2], abs[3], rel[2], rel[3]]))
    }
}

pub const ZERO_URECT: URect = URect {
    abs: ZERO_RECT,
    rel: ZERO_RELRECT,
};

pub const FILL_URECT: URect = URect {
    abs: ZERO_RECT,
    rel: RelRect::new(0.0, 0.0, 1.0, 1.0),
};

pub const AUTO_URECT: URect = URect {
    abs: ZERO_RECT,
    rel: RelRect::new(0.0, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),
};

impl Mul<AbsRect> for URect {
    type Output = AbsRect;

    #[inline]
    fn mul(self, rhs: AbsRect) -> Self::Output {
        let ltrb = rhs.0.as_array_ref();
        let topleft = f32x4::new([ltrb[0], ltrb[1], ltrb[0], ltrb[1]]);
        let bottomright = f32x4::new([ltrb[2], ltrb[3], ltrb[2], ltrb[3]]);

        AbsRect(topleft + self.abs.0 + self.rel.0 * (bottomright - topleft))
    }
}

impl Mul<AbsDim> for URect {
    type Output = AbsRect;

    #[inline]
    fn mul(self, rhs: AbsDim) -> Self::Output {
        AbsRect(self.abs.0 + self.rel.0 * splat_vec2(rhs.0))
    }
}

impl From<AbsRect> for URect {
    #[inline]
    fn from(value: AbsRect) -> Self {
        Self {
            abs: value,
            rel: ZERO_RELRECT,
        }
    }
}

impl From<RelRect> for URect {
    #[inline]
    fn from(value: RelRect) -> Self {
        Self {
            abs: ZERO_RECT,
            rel: value,
        }
    }
}

/// Display Rectangle with both per-pixel and display-independent pixels
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DRect {
    pub px: AbsRect,
    pub dp: AbsRect,
    pub rel: RelRect,
}

impl DRect {
    fn resolve(&self, dpi: Vec2) -> URect {
        URect {
            abs: AbsRect(self.px.0 + (self.dp.0 * splat_vec2(dpi))),
            rel: self.rel,
        }
    }
    pub fn topleft(&self) -> DPoint {
        DPoint {
            dp: self.dp.topleft(),
            px: self.px.topleft(),
            rel: RelPoint(self.rel.topleft()),
        }
    }
    pub fn bottomright(&self) -> DPoint {
        DPoint {
            dp: self.dp.bottomright(),
            px: self.px.bottomright(),
            rel: RelPoint(self.rel.bottomright()),
        }
    }
}

pub const ZERO_DRECT: DRect = DRect {
    px: ZERO_RECT,
    dp: ZERO_RECT,
    rel: ZERO_RELRECT,
};

pub const FILL_DRECT: DRect = DRect {
    px: ZERO_RECT,
    dp: ZERO_RECT,
    rel: RelRect::new(0.0, 0.0, 1.0, 1.0),
};

pub const AUTO_DRECT: DRect = DRect {
    px: ZERO_RECT,
    dp: ZERO_RECT,
    rel: RelRect::new(0.0, 0.0, UNSIZED_AXIS, UNSIZED_AXIS),
};

impl From<URect> for DRect {
    /// By default we assume everything is in device independent pixels - if you wanted pixels, you should be using DRect explicitly
    fn from(value: URect) -> Self {
        Self {
            px: ZERO_RECT,
            dp: value.abs,
            rel: value.rel,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AbsLimits(f32x4);

// It would be cheaper to avoid using actual infinities here but we currently need them to make the math work
pub const DEFAULT_LIMITS: AbsLimits = AbsLimits(f32x4::new([
    f32::NEG_INFINITY,
    f32::NEG_INFINITY,
    f32::INFINITY,
    f32::INFINITY,
]));

impl AbsLimits {
    pub const fn new(min: Vec2, max: Vec2) -> Self {
        Self(f32x4::new([min.x, min.y, max.x, max.y]))
    }
    #[inline]
    pub fn min(&self) -> Vec2 {
        let minmax = self.0.as_array_ref();
        Vec2 {
            x: minmax[0],
            y: minmax[1],
        }
    }
    #[inline]
    pub fn max(&self) -> Vec2 {
        let minmax = self.0.as_array_ref();
        Vec2 {
            x: minmax[2],
            y: minmax[3],
        }
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
        let minmax = self.0.as_array_ref();
        let r = rhs.0.as_array_ref();

        Self(f32x4::new([
            minmax[0].max(r[0]),
            minmax[1].max(r[1]),
            minmax[2].min(r[2]),
            minmax[3].min(r[3]),
        ]))
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct DLimits {
    dp: AbsLimits,
    px: AbsLimits,
}

pub const DEFAULT_DLIMITS: DLimits = DLimits {
    dp: DEFAULT_LIMITS,
    px: AbsLimits(f32x4::ZERO),
};

impl DLimits {
    pub fn resolve(&self, dpi: Vec2) -> AbsLimits {
        AbsLimits(self.px.0 + self.dp.0 * splat_vec2(dpi))
    }
}

impl From<AbsLimits> for DLimits {
    fn from(value: AbsLimits) -> Self {
        DLimits {
            dp: value,
            px: AbsLimits(f32x4::ZERO),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RelLimits(f32x4);

pub const DEFAULT_RLIMITS: RelLimits = RelLimits(f32x4::new([
    f32::NEG_INFINITY,
    f32::NEG_INFINITY,
    f32::INFINITY,
    f32::INFINITY,
]));

impl Default for RelLimits {
    fn default() -> Self {
        DEFAULT_RLIMITS
    }
}

impl RelLimits {
    #[inline]
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self(f32x4::new([min.x, min.y, max.x, max.y]))
    }
    #[inline]
    pub fn min(&self) -> RelPoint {
        let minmax = self.0.as_array_ref();
        RelPoint(Vec2 {
            x: minmax[0],
            y: minmax[1],
        })
    }
    #[inline]
    pub fn max(&self) -> RelPoint {
        let minmax = self.0.as_array_ref();
        RelPoint(Vec2 {
            x: minmax[2],
            y: minmax[3],
        })
    }
}

impl Mul<AbsDim> for RelLimits {
    type Output = AbsLimits;

    #[inline]
    fn mul(self, rhs: AbsDim) -> Self::Output {
        let (unsized_x, unsized_y) = crate::layout::check_unsized_dim(rhs);
        let minmax = self.0.as_array_ref();
        AbsLimits(f32x4::new([
            if unsized_x {
                minmax[0]
            } else {
                minmax[0] * rhs.0.x
            },
            if unsized_y {
                minmax[1]
            } else {
                minmax[1] * rhs.0.y
            },
            if unsized_x {
                minmax[2]
            } else {
                minmax[2] * rhs.0.x
            },
            if unsized_y {
                minmax[3]
            } else {
                minmax[3] * rhs.0.y
            },
        ]))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct UValue {
    pub abs: f32,
    pub rel: f32,
}

impl UValue {
    pub fn resolve(&self, outer_dim: f32) -> f32 {
        if self.rel == UNSIZED_AXIS {
            UNSIZED_AXIS
        } else {
            self.abs + (self.rel * outer_dim)
        }
    }
    pub fn is_unsized(&self) -> bool {
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
    pub fn resolve(&self, dpi: f32) -> UValue {
        UValue {
            abs: self.px + (self.dp * dpi),
            rel: self.rel,
        }
    }
    pub fn is_unsized(&self) -> bool {
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

pub struct TextEditer {
    pub(crate) text: RefCell<String>,
    count: AtomicUsize,
    cursor: AtomicUsize,
    select: AtomicUsize, // If there's a selection, this is different from cursor and points at the end. Can be less than cursor.
}

impl TextEditer {
    fn get_content(&self) -> std::cell::Ref<'_, String> {
        self.text.borrow()
    }
    fn set_content(&self, content: &str) {
        *self.text.borrow_mut() = content.into();
        self.count.fetch_add(1, Ordering::Release);
    }
    fn edit(&self, multisplice: &[(std::ops::Range<usize>, String)]) {
        // TODO there's probably a more efficient way to do this
        for (range, replace) in multisplice {
            self.text.borrow_mut().replace_range(range.clone(), replace);
        }
        self.count.fetch_add(1, Ordering::Release);
    }
    fn get_cursor(&self) -> (usize, usize) {
        (
            self.cursor.load(Ordering::Relaxed),
            self.select.load(Ordering::Relaxed),
        )
    }
    fn set_cursor(&self, cursor: usize) {
        self.cursor.store(cursor, Ordering::Release);
        self.select.store(cursor, Ordering::Release);
        self.count.fetch_add(1, Ordering::Release);
    }
    fn set_selection(&self, cursor: (usize, usize)) {
        self.cursor.store(cursor.0, Ordering::Release);
        self.select.store(cursor.1, Ordering::Release);
        self.count.fetch_add(1, Ordering::Release);
    }
}

// If a component provides a CrossReferenceDomain, it's children can register themselves with it.
// Registered children will write their fully resolved area to the mapping, which can then be
// retrieved during the render step via a source ID.
#[derive(Default)]
pub struct CrossReferenceDomain {
    mappings: crate::RefCell<im::HashMap<Rc<SourceID>, AbsRect>>,
}

impl CrossReferenceDomain {
    pub fn write_area(&self, target: Rc<SourceID>, area: AbsRect) {
        self.mappings.borrow_mut().insert(target, area);
    }

    pub fn get_area(&self, target: &Rc<SourceID>) -> Option<AbsRect> {
        self.mappings.borrow().get(target).copied()
    }

    pub fn remove_self(&self, target: &Rc<SourceID>) {
        // TODO: Is this necessary? Does it even make sense? Do you simply need to wipe the mapping for every new layout instead?
        self.mappings.borrow_mut().remove(target);
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

// Points are specified as 72 per inch, and a scale factor of 1.0 corresponds to 96 DPI, so we multiply by the
// ratio times the scaling factor.
#[inline]
fn point_to_pixel(pt: f32, scale_factor: f32) -> f32 {
    pt * (72.0 / 96.0) * scale_factor
}

#[inline]
fn pixel_to_vec(p: winit::dpi::PhysicalPosition<f32>) -> Vec2 {
    Vec2::new(p.x, p.y)
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
                panic!("Invalid ID! Did you forget to initialize a component node's ID field?")
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

pub type EventHandler<Input, Output, State> = Box<
    dyn FnMut(Input, AbsRect, Vec2, State) -> Result<(State, Vec<Output>), (State, Vec<Output>)>,
>;

pub type EventWrapper<Output, State> = Box<
    dyn FnMut(
        DispatchPair,
        AbsRect,
        Vec2,
        State,
    ) -> Result<(State, Vec<Output>), (State, Vec<Output>)>,
>;

pub type AppEvent<State> = Box<dyn FnMut(DispatchPair, State) -> Result<State, State>>;

#[inline]
#[allow(clippy::type_complexity)]
fn wrap_event<Input: Dispatchable + 'static, Output: Dispatchable, State>(
    mut typed: impl FnMut(
        Input,
        AbsRect,
        Vec2,
        State,
    ) -> Result<(State, Vec<Output>), (State, Vec<Output>)>,
) -> impl FnMut(DispatchPair, AbsRect, Vec2, State) -> Result<(State, Vec<Output>), (State, Vec<Output>)>
{
    move |pair, rect, dpi, state| typed(Input::restore(pair).unwrap(), rect, dpi, state)
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
    fn init_default<State: 'static + component::StateMachineWrapper + Default>(
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
    pub fn get<'a, State: 'static + component::StateMachineWrapper>(
        &'a self,
        id: &SourceID,
    ) -> eyre::Result<&'a State> {
        let v = self.states.get(id).ok_or_eyre("State does not exist")?;
        v.as_any()
            .downcast_ref()
            .ok_or_eyre("Runtime type mismatch!")
    }
    pub fn get_mut<'a, State: 'static + component::StateMachineWrapper>(
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

    pub fn process(
        &mut self,
        event: DispatchPair,
        slot: &Slot,
        dpi: Vec2,
        area: AbsRect,
    ) -> eyre::Result<()> {
        type IterTuple = (Box<dyn Any>, u64, Option<Slot>);

        // We use smallvec here so we can satisfy the borrow checker without making yet another heap allocation in most cases
        let iter: SmallVec<[IterTuple; 2]> = {
            let state = self.states.get_mut(&slot.0).ok_or_eyre("Invalid slot")?;
            let v = state.process(event, slot.1, dpi, area)?;
            v.into_iter()
                .map(|(i, e)| (e, i, state.output_slot(i.ilog2() as usize).unwrap().clone()))
        }
        .collect();

        for (e, index, slot) in iter {
            if let Some(s) = slot.as_ref() {
                self.process((index, e), s, dpi, area)?;
            }
        }
        Ok(())
    }

    fn init_component<Parent: ?Sized>(
        &mut self,
        target: &dyn crate::component::ComponentWrap<Parent>,
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
    root: component::Root, // Root component node containing all windows
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
        _: crate::Vec2,
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
                root: component::Root::new(),
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
        use crate::component::shape::Shape;
        use ultraviolet::Vec4;
        let rect = Shape::<DRect>::round_rect(
            gen_id!().into(),
            crate::FILL_DRECT.into(),
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
