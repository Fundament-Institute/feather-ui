pub mod component;
pub mod input;
pub mod layout;
pub mod persist;
mod rtree;
pub mod window;

use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

use crate::input::Event;
use component::Component;
use std::rc::{Rc, Weak};
use ultraviolet::vec::Vec2;
use wgpu::Device;
use wgpu::RenderPass;

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

pub fn build_aabb(a: Vec2, b: Vec2) -> AbsRect {
    AbsRect {
        topleft: a.min_by_component(b),
        bottomright: a.max_by_component(b),
    }
}

#[derive(Copy, Clone, Debug, Default)]
/// Unified coordinate rectangle
struct URect {
    pub topleft: UPoint,
    pub bottomright: UPoint,
}

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

pub trait RenderLambda: Fn(&wgpu::RenderPass, &wgpu::Device) + dyn_clone::DynClone {}
impl<T: Fn(&wgpu::RenderPass, &wgpu::Device) + ?Sized + dyn_clone::DynClone> RenderLambda for T {}
dyn_clone::clone_trait_object!(RenderLambda);

type EventHandler<AppData> = Box<dyn Fn(Event, AppData) -> AppData>;
type RenderInstruction = Box<dyn RenderLambda>;

pub fn draw_pass(
    surface: &RenderPass,
    device: &Device,
    commands: im::OrdMap<i64, RenderInstruction>,
) {
    for (_, f) in commands.iter() {
        f(surface, device);
    }
}

pub struct App<AppData: 'static> {
    app_state: AppData,
    component_tree: Rc<dyn Component<AppData, ()>>,
    layout_tree: Rc<layout::Node<AppData, layout::root::Root, ()>>,
    //resolved_tree: Rc<dyn layout::Resolved<AppData>>,
    rtree: Weak<rtree::Node<AppData>>,
}
