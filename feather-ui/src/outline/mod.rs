use dyn_clone::DynClone;
use eyre::OptionExt;
use glyphon::fontdb::Source;
use smallvec::SmallVec;
use window::WindowStateMachine;

use crate::layout::Layout;

pub mod arc;
pub mod button;
pub mod mouse_area;
pub mod region;
pub mod round_rect;
pub mod text;
pub mod window;

use crate::layout::Desc;
use crate::layout::Staged;
use crate::outline::window::Window;
use crate::outline::window::WindowState;
use crate::rtree;
use crate::AbsRect;
use crate::DispatchPair;
use crate::Dispatchable;
use crate::DriverState;
use crate::EventWrapper;
use crate::RenderInstruction;
use crate::Slot;
use crate::SourceID;
use crate::StateManager;
use eyre::Result;
use std::any::Any;
use std::collections::HashMap;
use std::rc::Rc;
use std::rc::Weak;

pub trait StateMachineWrapper {
    fn process(&mut self, input: DispatchPair, index: u64) -> Result<Vec<(u64, Box<dyn Any>)>>;
    fn as_any(&mut self) -> &mut dyn Any;
    fn output_slot(&self, i: usize) -> Result<&Option<Slot>>;
    fn input_masks(&self) -> SmallVec<[u64; 4]>;
}

pub struct StateMachine<
    Output: Dispatchable + 'static,
    State: 'static,
    const INPUT_SIZE: usize,
    const OUTPUT_SIZE: usize,
> {
    pub state: Option<State>,
    pub input: [(u64, EventWrapper<Output, State>); INPUT_SIZE],
    pub output: [Option<Slot>; OUTPUT_SIZE],
}

impl<
        Output: Dispatchable + 'static,
        State: 'static,
        const INPUT_SIZE: usize,
        const OUTPUT_SIZE: usize,
    > StateMachineWrapper for StateMachine<Output, State, INPUT_SIZE, OUTPUT_SIZE>
{
    fn process(&mut self, input: DispatchPair, index: u64) -> Result<Vec<(u64, Box<dyn Any>)>> {
        let (mask, f) = self
            .input
            .get_mut(index as usize)
            .ok_or_eyre("index out of bounds")?;
        if input.0 & *mask == 0 {
            return Err(eyre::eyre!("Event handler doesn't handle this method"));
        }
        let result = match f(input, Default::default(), self.state.take().unwrap()) {
            Ok((s, r)) => {
                self.state = Some(s);
                r
            }
            Err((s, r)) => {
                self.state = Some(s);
                r
            }
        };
        Ok(result.into_iter().map(|x| x.extract()).collect())
    }
    fn output_slot(&self, i: usize) -> Result<&Option<Slot>> {
        self.output.get(i).ok_or_eyre("index out of bounds")
    }
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
    fn input_masks(&self) -> SmallVec<[u64; 4]> {
        self.input.iter().map(|x| x.0).collect()
    }
}

pub trait Outline<AppData, Parent: Clone>: DynClone {
    fn layout(
        &self,
        state: &mut StateManager,
        driver: &DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Parent, AppData>>;

    fn id(&self) -> std::rc::Rc<SourceID>;
}

dyn_clone::clone_trait_object!(<AppData, Parent> Outline<AppData, Parent> where Parent: Clone);

pub type OutlineFrom<AppData, D> = dyn Outline<AppData, <D as Desc<AppData>>::Impose>;

pub trait Renderable<AppData> {
    fn render(&self, area: AbsRect, driver: &DriverState) -> im::Vector<RenderInstruction>;
}

// Stores the root node for the various trees.

pub struct RootState<AppData> {
    layout_tree: Option<Box<dyn crate::layout::Layout<(), AppData>>>,
    staging: Option<Box<dyn Staged<AppData>>>,
    rtree: Weak<rtree::Node<AppData>>,
}

impl<AppData> Default for RootState<AppData> {
    fn default() -> Self {
        Self {
            layout_tree: None,
            staging: None,
            rtree: std::rc::Weak::<rtree::Node<AppData>>::new(),
        }
    }
}

pub struct Root<AppData: 'static + std::cmp::PartialEq> {
    pub(crate) state: HashMap<SourceID, RootState<AppData>>,
    // We currently rely on window-specific functions, so there's no point trying to make this more general right now.
    pub(crate) children: im::HashMap<winit::window::WindowId, Option<Window<AppData>>>,
}

impl<AppData: 'static + std::cmp::PartialEq> Root<AppData> {
    pub fn new() -> Self {
        Self {
            state: HashMap::new(),
            children: im::HashMap::new(),
        }
    }

    #[inline]
    fn with_window(
        &mut self,
        mut f: impl FnMut(&Window<AppData>, &mut RootState<AppData>) -> eyre::Result<()>,
    ) -> eyre::Result<()> {
        for (id, window) in self.children.iter() {
            let window = window.as_ref().unwrap();
            let root = self
                .state
                .get_mut(&window.id())
                .ok_or_eyre("Couldn't find window state")?;
            f(window, root)?;
        }
        Ok(())
    }

    pub fn layout_all<
        O: crate::FnPersist<AppData, im::HashMap<winit::window::WindowId, Option<Window<AppData>>>>,
    >(
        &mut self,
        manager: &mut StateManager,
        driver: &mut std::sync::Weak<DriverState>,
        instance: &wgpu::Instance,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> eyre::Result<()> {
        self.with_window(|window, root| {
            root.layout_tree =
                Some(window.custom_layout::<O>(manager, driver, instance, event_loop)?);
            Ok(())
        })
    }

    pub fn stage_all(&mut self, states: &mut StateManager) -> eyre::Result<()> {
        self.with_window(|window, root| {
            if let Some(layout) = root.layout_tree.as_ref() {
                let state: &mut WindowStateMachine = states.get(&window.id())?;
                let staging =
                    layout.stage(Default::default(), &state.state.as_ref().unwrap().driver);
                root.rtree = staging.get_rtree();
                root.staging = Some(staging);
            }

            Ok(())
        })
    }

    /*
    pub fn render_all(&mut self, states: &mut StateManager) -> eyre::Result<()> {
        self.with_window(|window, root| {
            if let Some(staging) = root.staging.as_ref() {
                let state: &mut WindowStateMachine = states.get(&window.id())?;
                state.state.draw = staging.render();
            }

            Ok(())
        })
    }*/
}
