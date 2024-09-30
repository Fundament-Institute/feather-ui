use color_eyre::owo_colors::colors::Default;

use crate::input::Event;
use crate::layout::LayoutParent;
use crate::persist::Concat;
use crate::persist::FnPersist;
use crate::Component;
use crate::EventHandler;

// A button component that contains a mousearea and N arbitrary inner components
pub struct Button<AppData, Parent: Clone> {
    props: Parent,
    basic: Basic,
    marea: MouseArea<AppData>,
    inner: im::Vector<Box<dyn Component<AppData>>>,
    //render_store: im::Vector<Concat<crate::RenderInstruction>>,
}

impl<AppData, const N: usize> ClickHandler<AppData> for Button<AppData, N> {
    fn on_click(&self, data: AppData) -> AppData {
        (self.onclick)(data)
    }
}
impl<AppData, const N: usize> Component<AppData> for Button<AppData, N> {
    fn on_event(&self, event: Event, data: AppData) -> AppData {
        todo!()
    }

    fn render(&self) -> im::Vector<crate::RenderInstruction> {
        // Fold over our inner components with a persistent concat
        self.inner
            .iter()
            .fold(im::Vector::<crate::RenderInstruction>::new(), |seed, v| {
                Concat::call(Default::default(), (seed, v.render()))
            })
    }

    fn layout(
        &self,
        layout: &mut Vec<std::rc::Weak<dyn crate::layout::LayoutNode>>,
        data: AppData,
    ) {
        todo!()
    }
}
