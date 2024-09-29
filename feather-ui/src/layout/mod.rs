pub mod basic;
pub mod root;

use dyn_clone::DynClone;

use crate::persist::Concat;
use crate::persist::FnPersist;
use crate::persist::VectorFold;
use crate::rtree;
use crate::AbsRect;
use crate::RenderInstruction;
use derive_where::derive_where;
use std::rc::{Rc, Weak};

pub trait Layout<Imposed, AppData>: DynClone {
    fn get_imposed(&self) -> &Imposed;
    fn stage(&self, area: AbsRect) -> Box<dyn Staged<AppData>>;
}

dyn_clone::clone_trait_object!(<Imposed, AppData> Layout<Imposed, AppData>);

pub trait Desc<AppData: 'static> {
    type Props: Clone;
    type Impose: Clone;
    type Children<A: DynClone + ?Sized>: Clone + 'static;

    /// Resolves a pending layout into a resolved node, which contains a pointer to the R-tree
    fn stage(
        props: &Self::Props,
        area: AbsRect,
        children: &Self::Children<dyn Layout<Self::Impose, AppData>>,
    ) -> Box<dyn Staged<AppData>>;

    fn fake_stage(
        props: &Self::Props,
        area: AbsRect,
        children: &Self::Children<dyn Layout<Self::Impose, AppData>>,
    ) -> () {
    }
}

#[derive_where(Clone)]
pub struct Node<AppData: 'static, D: Desc<AppData>, Imposed: Clone> {
    props: D::Props,
    imposed: Imposed,
    children: D::Children<dyn Layout<D::Impose, AppData>>,
}

impl<AppData: 'static, D: Desc<AppData>, Imposed: Clone> Layout<Imposed, AppData>
    for Node<AppData, D, Imposed>
{
    fn get_imposed(&self) -> &Imposed {
        &self.imposed
    }
    fn stage(&self, area: AbsRect) -> Box<dyn Staged<AppData>> {
        //D::stage(&self.props, area, &self.children)
        D::fake_stage(&self.props, area, &self.children);
        todo!();
    }
}

pub trait Staged<AppData>: DynClone {
    fn render(&self) -> im::Vector<RenderInstruction>;
    fn get_rtree(&self) -> Weak<rtree::Node<AppData>>;
}

dyn_clone::clone_trait_object!(<AppData> Staged<AppData>);

#[derive_where(Clone)]
struct Concrete<AppData: 'static> {
    render: im::Vector<RenderInstruction>,
    area: AbsRect,
    rtree: Rc<rtree::Node<AppData>>,
    children: im::Vector<Box<dyn Staged<AppData>>>,
}

impl<AppData> Staged<AppData> for Concrete<AppData> {
    fn render(&self) -> im::Vector<RenderInstruction> {
        let f: VectorFold<RenderInstruction, im::Vector<RenderInstruction>, Concat>;
        //(Concat {}).into();

        //f.call()
        todo!()
    }

    fn get_rtree(&self) -> Weak<rtree::Node<AppData>> {
        Rc::downgrade(&self.rtree.clone())
    }
}

/*pub fn fixed_point(mut node: Weak<dyn Layout<AppData>>) {
    while let Some(n) = node.upgrade() {
        if !n.evaluate() {
            return;
        }
        // It is critically important we do this walk outside of the evaluate call, because the parent
        // could modify it's children, so we MUST drop that reference after calling evaluate.
        node = n.get_parent();
    }
}*/
