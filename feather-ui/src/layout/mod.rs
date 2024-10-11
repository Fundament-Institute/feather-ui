pub mod basic;
pub mod center;
pub mod empty;
pub mod root;

use dyn_clone::DynClone;

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

pub trait Layout<Imposed: Clone, AppData>: DynClone {
    fn get_imposed(&self) -> &Imposed;
    fn stage<'a>(&self, area: AbsRect, driver: &DriverState) -> Box<dyn Staged<AppData> + 'a>
    where
        AppData: 'a;
}

dyn_clone::clone_trait_object!(<Imposed, AppData> Layout<Imposed, AppData> where Imposed:Clone);

pub trait Desc<AppData> {
    type Props: Clone;
    type Impose: Clone;
    type Children<A: DynClone + ?Sized>: Clone;

    /// Resolves a pending layout into a resolved node, which contains a pointer to the R-tree
    fn stage<'a>(
        props: &Self::Props,
        area: AbsRect,
        children: &Self::Children<dyn Layout<Self::Impose, AppData> + '_>,
        id: std::rc::Weak<SourceID>,
        renderable: Option<Rc<dyn Renderable<AppData>>>,
        driver: &DriverState,
    ) -> Box<dyn Staged<AppData> + 'a>
    where
        AppData: 'a;
}

#[derive_where(Clone)]
pub struct Node<AppData, D: Desc<AppData>, Imposed: Clone> {
    pub props: D::Props,
    pub imposed: Imposed,
    pub id: std::rc::Weak<SourceID>,
    pub children: D::Children<dyn Layout<D::Impose, AppData>>,
    pub renderable: Option<Rc<dyn Renderable<AppData>>>,
}

impl<AppData, D: Desc<AppData>, Imposed: Clone> Layout<Imposed, AppData>
    for Node<AppData, D, Imposed>
{
    fn get_imposed(&self) -> &Imposed {
        &self.imposed
    }
    fn stage<'a>(&self, area: AbsRect, driver: &DriverState) -> Box<dyn Staged<AppData> + 'a>
    where
        AppData: 'a,
    {
        D::stage(
            &self.props,
            area,
            &self.children,
            self.id.clone(),
            self.renderable.as_ref().map(|x| x.clone()),
            driver,
        )
    }
}

pub trait Staged<AppData>: DynClone {
    fn render(&self) -> im::Vector<RenderInstruction>;
    fn get_rtree(&self) -> Weak<rtree::Node<AppData>>;
    fn get_area(&self) -> AbsRect;
}

dyn_clone::clone_trait_object!(<AppData> Staged<AppData>);

#[derive_where(Clone)]
struct Concrete<AppData> {
    render: im::Vector<RenderInstruction>,
    area: AbsRect,
    rtree: Rc<rtree::Node<AppData>>,
    children: im::Vector<Option<Box<dyn Staged<AppData>>>>,
}

impl<AppData> Staged<AppData> for Concrete<AppData> {
    fn render(&self) -> im::Vector<RenderInstruction> {
        let fold = VectorFold::new(
            |list: &im::Vector<RenderInstruction>,
             n: &Option<Box<dyn Staged<AppData>>>|
             -> im::Vector<RenderInstruction> {
                let mut a = n.as_ref().unwrap().render();
                a.append(list.clone());
                a
            },
        );

        let (_, result) = fold.call(fold.init(), &self.render, &self.children);
        result
    }

    fn get_rtree(&self) -> Weak<rtree::Node<AppData>> {
        Rc::downgrade(&self.rtree)
    }

    fn get_area(&self) -> AbsRect {
        self.area
    }
}
