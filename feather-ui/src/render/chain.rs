use std::rc::Rc;

pub struct Pipeline<const N: usize>(pub [Rc<dyn super::Renderable>; N]);

impl<const N: usize> super::Renderable for Pipeline<N> {
    fn render(
        &self,
        area: crate::AbsRect,
        driver: &crate::DriverState,
    ) -> im::Vector<crate::RenderInstruction> {
        self.0.iter().flat_map(|x| x.render(area, driver)).collect()
    }
}
