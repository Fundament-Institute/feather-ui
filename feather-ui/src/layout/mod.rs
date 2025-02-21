// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

pub mod basic;
pub mod domain_write;
pub mod empty;
pub mod flex;
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

pub trait Layout<Imposed: Clone>: DynClone {
    fn get_imposed(&self) -> &Imposed;
    fn stage<'a>(
        &self,
        area: AbsRect,
        parent_pos: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a>;
}

dyn_clone::clone_trait_object!(<Imposed> Layout<Imposed> where Imposed:Clone);

pub trait Desc {
    type Props: Clone;
    type Impose: Clone;
    type Children<A: DynClone + ?Sized>: Clone;

    /// Resolves a pending layout into a resolved node, which contains a pointer to the R-tree
    fn stage<'a>(
        props: &Self::Props,
        true_area: AbsRect,
        parent_pos: Vec2,
        children: &Self::Children<dyn Layout<Self::Impose> + '_>,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable>>,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a>;
}

#[derive_where(Clone)]
pub struct Node<D: Desc, Imposed: Clone> {
    pub props: D::Props,
    pub imposed: Imposed,
    pub id: std::rc::Weak<SourceID>,
    pub children: D::Children<dyn Layout<D::Impose>>,
    pub renderable: Option<Rc<dyn Renderable>>,
}

impl<D: Desc, Imposed: Clone> Layout<Imposed> for Node<D, Imposed> {
    fn get_imposed(&self) -> &Imposed {
        &self.imposed
    }
    fn stage<'a>(
        &self,
        area: AbsRect,
        parent_pos: Vec2,
        driver: &DriverState,
    ) -> Box<dyn Staged + 'a> {
        D::stage(
            &self.props,
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
