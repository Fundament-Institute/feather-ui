// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

pub mod basic;
pub mod domain_write;
//pub mod flex;
pub mod leaf;
pub mod prop;
pub mod root;
pub mod simple;

use dyn_clone::DynClone;
use ultraviolet::Vec2;

use crate::outline::Renderable;
use crate::persist::FnPersist2;
use crate::persist::VectorFold;
use crate::rtree;
use crate::AbsRect;
use crate::DriverState;
use crate::RenderInstruction;
use crate::SourceID;
use derive_where::derive_where;
use std::rc::{Rc, Weak};

pub trait Layout<Props>: DynClone {
    fn get_props(&self) -> &Props;
    fn inner_stage<'a>(
        &self,
        area: AbsRect,
        parent_pos: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a>;
}

dyn_clone::clone_trait_object!(<Imposed> Layout<Imposed> where Imposed:Sized);

pub trait LayoutWrap<Imposed: ?Sized>: DynClone {
    fn get_imposed(&self) -> &Imposed;
    fn stage<'a>(
        &self,
        area: AbsRect,
        parent_pos: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a>;
}

dyn_clone::clone_trait_object!(<Imposed> LayoutWrap<Imposed> where Imposed:?Sized);

impl<U: ?Sized, T> LayoutWrap<U> for Box<dyn Layout<T>>
where
    for<'a> &'a T: Into<&'a U>,
{
    fn get_imposed(&self) -> &U {
        self.get_props().into()
    }

    fn stage<'a>(
        &self,
        area: AbsRect,
        parent_pos: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a> {
        self.inner_stage(area, parent_pos, driver)
    }
}

impl<U: ?Sized, T> LayoutWrap<U> for &dyn Layout<T>
where
    for<'a> &'a T: Into<&'a U>,
{
    fn get_imposed(&self) -> &U {
        self.get_props().into()
    }

    fn stage<'a>(
        &self,
        area: AbsRect,
        parent_pos: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a> {
        self.inner_stage(area, parent_pos, driver)
    }
}

impl<T: 'static> From<Box<dyn Layout<T>>> for Box<dyn LayoutWrap<T>> {
    fn from(value: Box<dyn Layout<T>>) -> Self {
        Box::new(value)
    }
}

pub trait Desc {
    type Props: ?Sized;
    type Child: ?Sized;
    type Children: Clone;

    /// Resolves a pending layout into a resolved node, which contains a pointer to the R-tree
    fn stage<'a>(
        props: &Self::Props,
        true_area: AbsRect,
        parent_pos: Vec2,
        children: &Self::Children,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a>;
}

#[derive_where(Clone)]
pub struct Node<T, D: Desc + ?Sized> {
    pub props: Rc<T>,
    pub id: std::rc::Weak<SourceID>,
    pub children: D::Children,
    pub renderable: Option<Rc<dyn Renderable>>,
}

impl<T, D: Desc + ?Sized> Layout<T> for Node<T, D>
where
    for<'a> &'a T: Into<&'a D::Props>,
{
    fn get_props(&self) -> &T {
        self.props.as_ref()
    }
    fn inner_stage<'a>(
        &self,
        area: AbsRect,
        parent_pos: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a> {
        D::stage(
            self.props.as_ref().into(),
            area,
            parent_pos,
            &self.children,
            self.id.clone(),
            self.renderable.as_ref().map(|x| x.clone()),
            driver,
        )
    }
}

pub trait Staged: DynClone {
    fn render(&self) -> im::Vector<RenderInstruction>;
    fn get_rtree(&self) -> Weak<rtree::Node>;
    fn get_area(&self) -> AbsRect;
}

dyn_clone::clone_trait_object!(Staged);

#[derive(Clone)]
pub(crate) struct Concrete {
    pub render: im::Vector<RenderInstruction>,
    pub area: AbsRect,
    pub rtree: Rc<rtree::Node>,
    pub children: im::Vector<Option<Box<dyn Staged>>>,
}

impl Staged for Concrete {
    fn render(&self) -> im::Vector<RenderInstruction> {
        let fold = VectorFold::new(
            |list: &im::Vector<RenderInstruction>,
             n: &Option<Box<dyn Staged>>|
             -> im::Vector<RenderInstruction> {
                let mut a = n.as_ref().unwrap().render();
                a.append(list.clone());
                a
            },
        );

        let (_, result) = fold.call(fold.init(), &self.render, &self.children);
        result
    }

    fn get_rtree(&self) -> Weak<rtree::Node> {
        Rc::downgrade(&self.rtree)
    }

    fn get_area(&self) -> AbsRect {
        self.area
    }
}

pub(crate) fn zero_infinity(mut v: Vec2) -> Vec2 {
    if v.x.is_infinite() {
        v.x = 0.0
    }
    if v.y.is_infinite() {
        v.y = 0.0
    }
    v
}
