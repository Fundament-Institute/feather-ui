use dyn_clone::DynClone;

use crate::layout::Layout;

pub mod arc;
pub mod button;
pub mod mouse_area;
pub mod region;
pub mod round_rect;
pub mod text;
pub mod window;

use crate::layout::Desc;
use crate::AbsRect;
use crate::DriverState;
use crate::RenderInstruction;

pub trait Component<AppData, Parent: Clone>: DynClone {
    fn layout(
        &self,
        data: &AppData,
        driver: &DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Parent, AppData>>;
}

dyn_clone::clone_trait_object!(<AppData, Parent> Component<AppData, Parent> where Parent: Clone);

pub type ComponentFrom<AppData, D> = dyn Component<AppData, <D as Desc<AppData>>::Impose>;

pub trait Renderable<AppData> {
    fn render(&self, area: AbsRect, driver: &DriverState) -> im::Vector<RenderInstruction>;
}
