// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use dyn_clone::DynClone;
use eyre::OptionExt;
use smallvec::SmallVec;
use window::WindowStateMachine;

use crate::layout::Layout;

pub mod button;
pub mod domain_line;
pub mod domain_point;
//pub mod draggable;
//pub mod flexbox;
pub mod mouse_area;
//pub mod paragraph;
pub mod region;
pub mod shape;
pub mod text;
pub mod window;

use crate::layout::root;
use crate::layout::Desc;
use crate::layout::LayoutWrap;
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

pub trait Outline<T>: DynClone {
    fn layout(
        &self,
        state: &StateManager,
        driver: &DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T> + 'static>;

    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        Err(crate::Error::Stateless)
    }
    fn init_all(&self, _: &mut StateManager) -> eyre::Result<()>;
    fn id(&self) -> Rc<SourceID>;
}

dyn_clone::clone_trait_object!(<Parent> Outline<Parent>);

pub type OutlineFrom<D> = dyn OutlineWrap<<D as Desc>::Child>;

pub trait OutlineWrap<T: ?Sized>: DynClone {
    fn layout(
        &self,
        state: &StateManager,
        driver: &DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn LayoutWrap<T> + 'static>;
    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error>;
    fn init_all(&self, _: &mut StateManager) -> eyre::Result<()>;
    fn id(&self) -> Rc<SourceID>;
}

dyn_clone::clone_trait_object!(<T> OutlineWrap<T> where T:?Sized);

impl<U: ?Sized, T: 'static> OutlineWrap<U> for Box<dyn Outline<T>>
where
    for<'a> &'a T: Into<&'a U>,
{
    fn layout(
        &self,
        state: &StateManager,
        driver: &DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn LayoutWrap<U> + 'static> {
        Box::new(Outline::<T>::layout(self.as_ref(), state, driver, config))
    }

    fn init(&self) -> Result<Box<dyn crate::StateMachineWrapper>, crate::Error> {
        Outline::<T>::init(self.as_ref())
    }

    fn init_all(&self, manager: &mut StateManager) -> eyre::Result<()> {
        Outline::<T>::init_all(self.as_ref(), manager)
    }

    fn id(&self) -> Rc<SourceID> {
        Outline::<T>::id(self.as_ref())
    }
}

impl<U: ?Sized, T: 'static> OutlineWrap<U> for &dyn Outline<T>
where
    for<'a> &'a T: Into<&'a U>,
{
    fn layout(
        &self,
        state: &StateManager,
        driver: &DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn LayoutWrap<U> + 'static> {
        Box::new(Outline::<T>::layout(*self, state, driver, config))
    }

    fn init(&self) -> Result<Box<dyn crate::StateMachineWrapper>, crate::Error> {
        Outline::<T>::init(*self)
    }

    fn init_all(&self, manager: &mut StateManager) -> eyre::Result<()> {
        Outline::<T>::init_all(*self, manager)
    }

    fn id(&self) -> Rc<SourceID> {
        Outline::<T>::id(*self)
    }
}

pub trait Renderable {
    fn render(&self, area: AbsRect, driver: &DriverState) -> im::Vector<RenderInstruction>;
}

// Stores the root node for the various trees.

pub struct RootState {
    pub(crate) id: Rc<SourceID>,
    layout_tree: Option<Box<dyn crate::layout::Layout<crate::AbsDim>>>,
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

impl Default for Root {
    fn default() -> Self {
        Self::new()
    }
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
        driver: &mut std::rc::Weak<DriverState>,
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
            self.states
                .entry(id)
                .or_insert_with(|| RootState::new(window.id().clone()));

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
                let layout: &dyn LayoutWrap<dyn root::Prop> = &layout.as_ref();
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

// If an outline provides a CrossReferenceDomain, it's children can register themselves with it.
// Registered children will write their fully resolved area to the mapping, which can then be
// retrieved during the render step via a source ID.
#[derive(Default)]
pub struct CrossReferenceDomain {
    mappings: crate::RefCell<im::HashMap<Rc<SourceID>, AbsRect>>,
}

impl CrossReferenceDomain {
    pub fn write_area(&self, target: Rc<SourceID>, area: AbsRect) {
        self.mappings.borrow_mut().insert(target, area);
    }

    pub fn get_area(&self, target: &Rc<SourceID>) -> Option<AbsRect> {
        self.mappings.borrow().get(target).copied()
    }

    pub fn remove_self(&self, target: &Rc<SourceID>) {
        // TODO: Is this necessary? Does it even make sense? Do you simply need to wipe the mapping for every new layout instead?
        self.mappings.borrow_mut().remove(target);
    }
}

#[macro_export]
macro_rules! gen_outline_wrap_inner {
    () => {
        fn layout(
            &self,
            state: &crate::StateManager,
            driver: &crate::DriverState,
            config: &wgpu::SurfaceConfiguration,
        ) -> Box<dyn crate::outline::LayoutWrap<U> + 'static> {
            Box::new(crate::outline::Outline::<T>::layout(
                self, state, driver, config,
            ))
        }

        fn init(&self) -> Result<Box<dyn crate::StateMachineWrapper>, crate::Error> {
            crate::outline::Outline::<T>::init(self)
        }

        fn init_all(&self, manager: &mut crate::StateManager) -> eyre::Result<()> {
            crate::outline::Outline::<T>::init_all(self, manager)
        }

        fn id(&self) -> Rc<SourceID> {
            crate::outline::Outline::<T>::id(self)
        }
    };
}

#[macro_export]
macro_rules! gen_outline_wrap {
    ($name:ident, $prop:path) => {
        impl<U: ?Sized, T: $prop + 'static> crate::outline::OutlineWrap<U> for $name<T>
        where
            $name<T>: crate::outline::Outline<T>,
            for<'a> &'a T: Into<&'a U>,
        {
            crate::gen_outline_wrap_inner!();
        }
    };
    ($name:ident, $prop:path, $aux:path) => {
        impl<U: ?Sized, T: $prop + $aux + 'static> crate::outline::OutlineWrap<U> for $name<T>
        where
            $name<T>: crate::outline::Outline<T>,
            for<'a> &'a T: Into<&'a U>,
        {
            crate::gen_outline_wrap_inner!();
        }
    };
    ($a:lifetime, $name:ident, $prop:path) => {
        impl<$a, U: ?Sized, T: $prop + 'static> crate::outline::OutlineWrap<U> for $name<$a, T>
        where
            $name<$a, T>: crate::outline::Outline<T>,
            for<'abc> &'abc T: Into<&'abc U>,
        {
            crate::gen_outline_wrap_inner!();
        }
    };
}
