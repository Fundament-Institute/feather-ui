use dyn_clone::DynClone;
use eyre::OptionExt;
use smallvec::SmallVec;
use window::WindowStateMachine;

use crate::layout::Layout;

pub mod arc;
pub mod button;
pub mod flexbox;
pub mod mouse_area;
pub mod region;
pub mod round_rect;
pub mod text;
pub mod window;

use crate::layout::Desc;
use crate::layout::Staged;
use crate::outline::window::Window;
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
    fn process(
        &mut self,
        input: DispatchPair,
        index: u64,
        area: AbsRect,
    ) -> Result<Vec<DispatchPair>>;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn as_any(&self) -> &dyn Any;
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
    fn process(
        &mut self,
        input: DispatchPair,
        index: u64,
        area: AbsRect,
    ) -> Result<Vec<DispatchPair>> {
        let (mask, f) = self
            .input
            .get_mut(index as usize)
            .ok_or_eyre("index out of bounds")?;
        if input.0 & *mask == 0 {
            return Err(eyre::eyre!("Event handler doesn't handle this method"));
        }
        let result = match f(input, area, self.state.take().unwrap()) {
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
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn input_masks(&self) -> SmallVec<[u64; 4]> {
        self.input.iter().map(|x| x.0).collect()
    }
}

pub trait Outline<Parent: Clone>: DynClone {
    fn layout(
        &self,
        state: &StateManager,
        driver: &DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<Parent>>;

    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        Err(crate::Error::Stateless.into())
    }
    fn init_all(&self, _: &mut StateManager) -> eyre::Result<()>;
    fn id(&self) -> Rc<SourceID>;
}

dyn_clone::clone_trait_object!(<Parent> Outline<Parent> where Parent: Clone);

pub type OutlineFrom<D> = dyn Outline<<D as Desc>::Impose>;

pub trait Renderable {
    fn render(&self, area: AbsRect, driver: &DriverState) -> im::Vector<RenderInstruction>;
}

// Stores the root node for the various trees.

pub struct RootState {
    pub(crate) id: Rc<SourceID>,
    layout_tree: Option<Box<dyn crate::layout::Layout<()>>>,
    pub(crate) staging: Option<Box<dyn Staged>>,
    rtree: Weak<rtree::Node>,
}

impl RootState {
    fn new(id: Rc<SourceID>) -> Self {
        Self {
            id,
            layout_tree: None,
            staging: None,
            rtree: std::rc::Weak::<rtree::Node>::new(),
        }
    }
}

pub struct Root {
    pub(crate) states: HashMap<winit::window::WindowId, RootState>,
    // We currently rely on window-specific functions, so there's no point trying to make this more general right now.
    pub(crate) children: im::HashMap<Rc<SourceID>, Option<Window>>,
}

impl Root {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            children: im::HashMap::new(),
        }
    }

    pub fn layout_all<
        AppData: 'static + std::cmp::PartialEq,
        O: crate::FnPersist<AppData, im::HashMap<Rc<SourceID>, Option<Window>>>,
    >(
        &mut self,
        manager: &mut StateManager,
        driver: &mut std::sync::Weak<DriverState>,
        instance: &wgpu::Instance,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> eyre::Result<()> {
        // Initialize any states that need to be initialized before calling the layout function
        // TODO: make this actually efficient by perfoming the initialization when a new outline is initialized
        for (_, window) in self.children.iter() {
            let window = window.as_ref().unwrap();
            window.init_custom::<AppData, O>(manager, driver, instance, event_loop)?;
            let state: &WindowStateMachine = manager.get(&window.id())?;
            let id = state.state.as_ref().unwrap().window.id();
            if !self.states.contains_key(&id) {
                self.states
                    .insert(id.clone(), RootState::new(window.id().clone()));
            }
            let root = self
                .states
                .get_mut(&id)
                .ok_or_eyre("Couldn't find window state")?;
            root.layout_tree = Some(window.layout(
                manager,
                &state.state.as_ref().unwrap().driver,
                &state.state.as_ref().unwrap().config,
            ));
        }
        Ok(())
    }

    pub fn stage_all(&mut self, states: &mut StateManager) -> eyre::Result<()> {
        for (_, window) in self.children.iter() {
            let window = window.as_ref().unwrap();
            let state: &WindowStateMachine = states.get(&window.id())?;
            let id = state.state.as_ref().unwrap().window.id();
            let root = self
                .states
                .get_mut(&id)
                .ok_or_eyre("Couldn't find window state")?;
            if let Some(layout) = root.layout_tree.as_ref() {
                let staging = layout.stage(
                    Default::default(),
                    Default::default(),
                    &state.state.as_ref().unwrap().driver,
                );
                root.rtree = staging.get_rtree();
                root.staging = Some(staging);
                state.state.as_ref().unwrap().window.request_redraw();
            }
        }
        Ok(())
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
