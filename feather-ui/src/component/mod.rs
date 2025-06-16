// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

pub mod button;
pub mod domain_line;
pub mod domain_point;
pub mod flexbox;
pub mod gridbox;
pub mod line;
pub mod listbox;
pub mod mouse_area;
pub mod paragraph;
pub mod region;
pub mod shape;
pub mod text;
pub mod textbox;
pub mod window;

use crate::component::window::Window;
use crate::layout::{Desc, Layout, Staged, root};
use crate::{
    AbsRect, DEFAULT_LIMITS, DispatchPair, Dispatchable, EventWrapper, Slot, SourceID,
    StateMachineChild, StateManager, graphics, rtree,
};
use dyn_clone::DynClone;
use eyre::{OptionExt, Result};
use smallvec::SmallVec;
use std::any::Any;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use window::WindowStateMachine;

pub trait StateMachineWrapper: Any {
    fn process(
        &mut self,
        input: DispatchPair,
        index: u64,
        dpi: crate::Vec2,
        area: AbsRect,
    ) -> Result<(Vec<DispatchPair>, bool)>;
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
    State: PartialEq + 'static,
    const INPUT_SIZE: usize,
    const OUTPUT_SIZE: usize,
> StateMachineWrapper for StateMachine<Output, State, INPUT_SIZE, OUTPUT_SIZE>
{
    fn process(
        &mut self,
        input: DispatchPair,
        index: u64,
        dpi: crate::Vec2,
        area: AbsRect,
    ) -> Result<(Vec<DispatchPair>, bool)> {
        let (mask, f) = self
            .input
            .get_mut(index as usize)
            .ok_or_eyre("index out of bounds")?;
        if input.0 & *mask == 0 {
            return Err(eyre::eyre!("Event handler doesn't handle this method"));
        }
        let changed;
        let result = match f(input, area, dpi, self.state.take().unwrap()) {
            Ok((s, r)) | Err((s, r)) => {
                let s = Some(s);
                changed = self.state != s;
                self.state = s;
                r
            }
        };
        Ok((result.into_iter().map(|x| x.extract()).collect(), changed))
    }
    fn output_slot(&self, i: usize) -> Result<&Option<Slot>> {
        self.output.get(i).ok_or_eyre("index out of bounds")
    }
    fn input_masks(&self) -> SmallVec<[u64; 4]> {
        self.input.iter().map(|x| x.0).collect()
    }
}

/*pub struct EventRouter<const N: usize> {
    pub input: (u64, EventWrapper<Output, State>),
    pub output: [Option<Slot>; N],
}

impl<const N: usize> StateMachineWrapper for EventRouter<N> {
    fn process(
        &mut self,
        input: DispatchPair,
        index: u64,
        dpi: crate::Vec2,
        area: AbsRect,
    ) -> Result<(Vec<DispatchPair>, bool)> {
        todo!()
    }

    fn output_slot(&self, i: usize) -> Result<&Option<Slot>> {
        self.output.get(i).ok_or_eyre("index out of bounds")
    }

    fn input_masks(&self) -> SmallVec<[u64; 4]> {
        SmallVec::from_elem(self.input.0, 1)
    }
}*/

pub trait Component<T: ?Sized>: crate::StateMachineChild + DynClone {
    fn layout(
        &self,
        state: &StateManager,
        graphics: &graphics::Driver,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T> + 'static>;
}

dyn_clone::clone_trait_object!(<Parent> Component<Parent> where Parent:?Sized);

pub type ComponentFrom<D> = dyn ComponentWrap<<D as Desc>::Child>;

pub trait ComponentWrap<T: ?Sized>: crate::StateMachineChild + DynClone {
    fn layout(
        &self,
        state: &StateManager,
        graphics: &graphics::Driver,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T> + 'static>;
}

dyn_clone::clone_trait_object!(<T> ComponentWrap<T> where T:?Sized);

impl<U: ?Sized, T: 'static> ComponentWrap<U> for Box<dyn Component<T>>
where
    for<'a> &'a T: Into<&'a U>,
{
    fn layout(
        &self,
        state: &StateManager,
        graphics: &graphics::Driver,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<U> + 'static> {
        Box::new(Component::<T>::layout(
            self.as_ref(),
            state,
            graphics,
            window,
            config,
        ))
    }
}

impl<T: 'static> StateMachineChild for Box<dyn Component<T>> {
    fn init(&self) -> Result<Box<dyn crate::StateMachineWrapper>, crate::Error> {
        StateMachineChild::init(self.as_ref())
    }

    fn apply_children(
        &self,
        f: &mut dyn FnMut(&dyn StateMachineChild) -> eyre::Result<()>,
    ) -> eyre::Result<()> {
        StateMachineChild::apply_children(self.as_ref(), f)
    }

    fn id(&self) -> Rc<SourceID> {
        StateMachineChild::id(self.as_ref())
    }
}

impl<U: ?Sized, T: 'static> ComponentWrap<U> for &dyn Component<T>
where
    for<'a> &'a T: Into<&'a U>,
{
    fn layout(
        &self,
        state: &StateManager,
        graphics: &graphics::Driver,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<U> + 'static> {
        Box::new(Component::<T>::layout(
            *self, state, graphics, window, config,
        ))
    }
}

impl<T: 'static> StateMachineChild for &dyn Component<T> {
    fn init(&self) -> Result<Box<dyn crate::StateMachineWrapper>, crate::Error> {
        StateMachineChild::init(*self)
    }

    fn apply_children(
        &self,
        f: &mut dyn FnMut(&dyn StateMachineChild) -> eyre::Result<()>,
    ) -> eyre::Result<()> {
        StateMachineChild::apply_children(*self, f)
    }

    fn id(&self) -> Rc<SourceID> {
        StateMachineChild::id(*self)
    }
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
        graphics: &mut std::sync::Weak<graphics::Driver>,
        on_driver: &mut Option<Box<dyn FnOnce(std::sync::Weak<graphics::Driver>) -> () + 'static>>,
        instance: &wgpu::Instance,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> eyre::Result<()> {
        // Initialize any states that need to be initialized before calling the layout function
        // TODO: make this actually efficient by performing the initialization when a new component is initialized
        for (_, window) in self.children.iter() {
            let window = window.as_ref().unwrap();
            window.init_custom::<AppData, O>(manager, graphics, instance, event_loop, on_driver)?;
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
                &state.state.as_ref().unwrap().graphics,
                &window.id(),
                &state.state.as_ref().unwrap().config,
            ));
        }
        Ok(())
    }

    pub fn stage_all(&mut self, states: &mut StateManager) -> eyre::Result<()> {
        for (_, window) in self.children.iter() {
            let window = window.as_ref().unwrap();
            let state: &mut WindowStateMachine = states.get_mut(&window.id())?;
            let id = state.state.as_ref().unwrap().window.id();
            let root = self
                .states
                .get_mut(&id)
                .ok_or_eyre("Couldn't find window state")?;
            if let Some(layout) = root.layout_tree.as_ref() {
                let layout: &dyn Layout<dyn root::Prop> = &layout.as_ref();
                let staging = layout.stage(
                    Default::default(),
                    DEFAULT_LIMITS,
                    state.state.as_mut().unwrap(),
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

#[macro_export]
macro_rules! gen_component_wrap_inner {
    () => {
        fn layout(
            &self,
            state: &$crate::StateManager,
            graphics: &$crate::graphics::Driver,
            window: &Rc<SourceID>,
            config: &wgpu::SurfaceConfiguration,
        ) -> Box<dyn $crate::component::Layout<U> + 'static> {
            Box::new($crate::component::Component::<T>::layout(
                self, state, graphics, window, config,
            ))
        }
    };
}

#[macro_export]
macro_rules! gen_component_wrap {
    ($name:ident, $prop:path) => {
        impl<U: ?Sized, T: $prop + 'static> $crate::component::ComponentWrap<U> for $name<T>
        where
            $name<T>: $crate::component::Component<T>,
            for<'a> &'a T: Into<&'a U>,
        {
            $crate::gen_component_wrap_inner!();
        }
    };
    ($name:ident, $prop:path, $aux:path) => {
        impl<U: ?Sized, T: $prop + $aux + 'static> $crate::component::ComponentWrap<U> for $name<T>
        where
            $name<T>: $crate::component::Component<T>,
            for<'a> &'a T: Into<&'a U>,
        {
            $crate::gen_component_wrap_inner!();
        }
    };
    ($a:lifetime, $name:ident, $prop:path) => {
        impl<$a, U: ?Sized, T: $prop + 'static> $crate::component::ComponentWrap<U> for $name<$a, T>
        where
            $name<$a, T>: $crate::component::Component<T>,
            for<'abc> &'abc T: Into<&'abc U>,
        {
            $crate::gen_component_wrap_inner!();
        }
    };
}
