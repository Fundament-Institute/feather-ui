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
//pub mod scroll_area;
pub mod shape;
pub mod text;
pub mod textbox;
pub mod window;

use crate::component::window::Window;
use crate::layout::{Desc, Layout, Staged, root};
use crate::{
    AbsRect, DEFAULT_LIMITS, DispatchPair, Dispatchable, Slot, SourceID, StateMachineChild,
    StateManager, graphics, rtree,
};
use dyn_clone::DynClone;
use eyre::{OptionExt, Result};
use smallvec::SmallVec;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use window::WindowStateMachine;

#[cfg(debug_assertions)]
#[allow(clippy::mutable_key_type)]
fn cycle_check(id: &Arc<SourceID>, set: &std::collections::HashSet<&Arc<SourceID>>) {
    debug_assert!(!set.contains(id), "Cycle detected!");
}

fn set_child_parent(mut child: &Arc<SourceID>, id: Arc<SourceID>) -> Result<()> {
    while let Some(parent) = child.parent.get() {
        // If we're already in this child's inheritance stack, bail out
        if Arc::ptr_eq(&id, parent) {
            return Ok(());
        }
        child = parent;
    }
    child
        .parent
        .set(id.clone())
        .map_err(|e| eyre::Report::msg(e.to_string()))
}

#[inline]
#[allow(clippy::mutable_key_type)]
fn set_children<T: StateMachineChild>(this: T) -> T {
    let id = this.id();
    #[cfg(debug_assertions)]
    let parents = id.parents();
    this.apply_children(&mut |x| {
        #[cfg(debug_assertions)]
        cycle_check(&x.id(), &parents);
        set_child_parent(&x.id(), id.clone())
    })
    .expect("Parent set failed when it should never fail!");
    this
}

pub trait StateMachineWrapper: Any {
    fn process(
        &mut self,
        input: DispatchPair,
        index: u64,
        dpi: crate::Vec2,
        area: AbsRect,
        extent: AbsRect,
        driver: &std::sync::Weak<crate::Driver>,
    ) -> Result<SmallVec<[DispatchPair; 1]>>;
    fn output_slot(&self, i: usize) -> Result<&Option<Slot>>;
    fn input_mask(&self) -> u64;
    fn changed(&self) -> bool;
    fn set_changed(&mut self, changed: bool);
}

// : zerocopy::Immutable
pub trait EventStream
where
    Self: std::marker::Sized,
{
    type Input: Dispatchable + 'static;
    type Output: Dispatchable + 'static;

    #[allow(unused_variables)]
    #[allow(clippy::type_complexity)]
    fn process(
        self,
        input: Self::Input,
        area: AbsRect,
        extent: AbsRect,
        dpi: crate::Vec2,
        driver: &std::sync::Weak<crate::Driver>,
    ) -> Result<(Self, SmallVec<[Self::Output; 1]>), (Self, SmallVec<[Self::Output; 1]>)> {
        Err((self, SmallVec::new()))
    }
}

pub struct StateMachine<State: EventStream + 'static, const OUTPUT_SIZE: usize> {
    pub state: Option<State>,
    pub output: [Option<Slot>; OUTPUT_SIZE],
    pub input_mask: u64,
    pub(crate) changed: bool,
}

impl<State: EventStream + PartialEq + 'static, const OUTPUT_SIZE: usize> StateMachineWrapper
    for StateMachine<State, OUTPUT_SIZE>
{
    fn process(
        &mut self,
        input: DispatchPair,
        _index: u64,
        dpi: crate::Vec2,
        area: AbsRect,
        extent: AbsRect,
        driver: &std::sync::Weak<crate::Driver>,
    ) -> Result<SmallVec<[DispatchPair; 1]>> {
        if input.0 & self.input_mask == 0 {
            return Err(eyre::eyre!("Event handler doesn't handle this method"));
        }
        let result = match self.state.take().unwrap().process(
            State::Input::restore(input).unwrap(),
            area,
            extent,
            dpi,
            driver,
        ) {
            Ok((s, r)) | Err((s, r)) => {
                let s = Some(s);
                self.changed |= self.state != s;
                self.state = s;
                r
            }
        };
        Ok(result.into_iter().map(|x| x.extract()).collect())
    }
    fn output_slot(&self, i: usize) -> Result<&Option<Slot>> {
        self.output.get(i).ok_or_eyre("index out of bounds")
    }
    fn input_mask(&self) -> u64 {
        self.input_mask
    }
    fn changed(&self) -> bool {
        self.changed
    }
    fn set_changed(&mut self, changed: bool) {
        self.changed = changed
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

pub trait Component: crate::StateMachineChild + DynClone {
    type Props: 'static;

    fn layout(
        &self,
        state: &mut StateManager,
        driver: &graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<Self::Props> + 'static>;
}

dyn_clone::clone_trait_object!(<Parent> Component<Props = Parent> where Parent:?Sized);

pub type ChildOf<D> = dyn ComponentWrap<<D as Desc>::Child>;

pub trait ComponentWrap<T: ?Sized>: crate::StateMachineChild + DynClone {
    fn layout(
        &self,
        state: &mut StateManager,
        driver: &graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T> + 'static>;
}

dyn_clone::clone_trait_object!(<T> ComponentWrap<T> where T:?Sized);

impl<U: ?Sized, C: Component> ComponentWrap<U> for C
where
    for<'a> &'a U: From<&'a <C as Component>::Props>,
    <C as Component>::Props: Sized,
{
    fn layout(
        &self,
        state: &mut StateManager,
        driver: &graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<U> + 'static> {
        Box::new(Component::layout(self, state, driver, window))
    }
}

impl<T: Component + 'static, U> From<Box<T>> for Box<dyn ComponentWrap<U>>
where
    for<'a> &'a U: std::convert::From<&'a <T as Component>::Props>,
    <T as Component>::Props: std::marker::Sized,
{
    fn from(value: Box<T>) -> Self {
        value
    }
}

// Stores the root node for the various trees.

pub struct RootState {
    pub(crate) id: Arc<SourceID>,
    layout_tree: Option<Box<dyn crate::layout::Layout<crate::AbsDim>>>,
    pub(crate) staging: Option<Box<dyn Staged>>,
    rtree: std::rc::Weak<rtree::Node>,
}

impl RootState {
    fn new(id: Arc<SourceID>) -> Self {
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
    pub(crate) children: im::HashMap<Arc<SourceID>, Option<Window>>,
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

    pub fn layout_all(
        &mut self,
        manager: &mut StateManager,
        driver: &mut std::sync::Weak<graphics::Driver>,
        on_driver: &mut Option<Box<dyn FnOnce(std::sync::Weak<graphics::Driver>) + 'static>>,
        instance: &wgpu::Instance,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> eyre::Result<()> {
        // Initialize any states that need to be initialized before calling the layout function
        // TODO: make this actually efficient by performing the initialization when a new component is initialized
        for (_, window) in self.children.iter() {
            let window = window.as_ref().unwrap();
            window.init_custom(manager, driver, instance, event_loop, on_driver)?;
            let state: &WindowStateMachine = manager.get(&window.id())?;
            let id = state.state.as_ref().unwrap().window.id();
            self.states
                .entry(id)
                .or_insert_with(|| RootState::new(window.id().clone()));

            let root = self
                .states
                .get_mut(&id)
                .ok_or_eyre("Couldn't find window state")?;
            let driver = state.state.as_ref().unwrap().driver.clone();
            root.layout_tree = Some(crate::component::Component::layout(
                window,
                manager,
                &driver,
                &window.id(),
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

    pub fn validate_ids(&self) -> eyre::Result<()> {
        struct Validator(std::collections::HashSet<Arc<SourceID>>);
        impl Validator {
            fn f(
                &mut self,
                x: &dyn StateMachineChild,
                parent: Option<&Arc<SourceID>>,
            ) -> eyre::Result<()> {
                let id = x.id();
                let mut cur = Some(&id);
                if let Some(parent_id) = parent {
                    while let Some(cur_id) = cur {
                        if cur_id == parent_id {
                            break;
                        }
                        let c = cur_id.parent.get();
                        cur = c;
                    }
                    if cur.is_none() {
                        return Err(eyre::eyre!(
                            "Invalid Parent ID found! All components must have an ID that is a direct or indirect child of their parent! {}",
                            x.id()
                        ));
                    }
                }
                if !self.0.insert(id.clone()) {
                    return Err(eyre::eyre!(
                        "Duplicate ID found! Did you forget to add a child index to an ID? {}",
                        x.id()
                    ));
                }

                x.apply_children(&mut |x| self.f(x, Some(&id)))
            }
        }
        let mut v = Validator(std::collections::HashSet::new());
        for (_, child) in &self.children {
            if let Some(window) = child {
                v.f(window, None)?;
            }
        }

        Ok(())
    }
}
