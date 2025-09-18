// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

pub mod button;
pub mod domain_line;
pub mod domain_point;
pub mod flexbox;
pub mod gridbox;
pub mod image;
pub mod line;
pub mod listbox;
pub mod mouse_area;
pub mod paragraph;
pub mod region;
pub mod scroll_area;
pub mod shape;
pub mod text;
pub mod textbox;
pub mod window;

use crate::component::window::Window;
use crate::event::EventRouter;
use crate::layout::{Desc, Layout, Staged, root};
use crate::{
    DispatchPair, Dispatchable, InputResult, PxRect, Slot, SourceID, StateMachineChild,
    StateManager, graphics, rtree,
};
use dyn_clone::DynClone;
use eyre::{OptionExt, Result};
use smallvec::SmallVec;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use window::WindowStateMachine;

pub trait StateMachineWrapper: Any {
    fn process(
        &mut self,
        input: DispatchPair,
        index: u64,
        dpi: crate::RelDim,
        area: PxRect,
        extent: PxRect,
        driver: &std::sync::Weak<crate::Driver>,
    ) -> InputResult<SmallVec<[DispatchPair; 1]>>;
    fn output_slot(&self, i: usize) -> Result<&Option<Slot>>;
    fn input_mask(&self) -> u64;
    fn changed(&self) -> bool;
    fn set_changed(&mut self, changed: bool);
}

pub struct StateMachine<State, const OUTPUT_SIZE: usize> {
    pub state: State,
    pub output: [Option<Slot>; OUTPUT_SIZE],
    pub input_mask: u64,
    pub(crate) changed: bool,
}

impl<State: EventRouter + PartialEq + 'static, const OUTPUT_SIZE: usize> StateMachineWrapper
    for StateMachine<State, OUTPUT_SIZE>
{
    fn process(
        &mut self,
        input: DispatchPair,
        _index: u64,
        dpi: crate::RelDim,
        area: PxRect,
        extent: PxRect,
        driver: &std::sync::Weak<crate::Driver>,
    ) -> InputResult<SmallVec<[DispatchPair; 1]>> {
        if input.0 & self.input_mask == 0 {
            return InputResult::Error(crate::Error::UnhandledEvent.into());
        }

        let s = match State::Input::restore(input) {
            Ok(s) => s,
            Err(e) => return InputResult::Error(e.into()),
        };

        State::process(
            crate::AccessCell {
                value: &mut self.state,
                changed: &mut self.changed,
            },
            s,
            area,
            extent,
            dpi,
            driver,
        )
        .map(|x| x.into_iter().map(|x| x.extract()).collect())
    }
    fn output_slot(&self, i: usize) -> Result<&Option<Slot>> {
        self.output.get(i).ok_or(crate::Error::OutOfRange(i).into())
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
        dpi: crate::RelDim,
        area: AbsRect,
    ) -> InputResult<SmallVec<[DispatchPair; 1]>>{
        todo!()
    }

    fn output_slot(&self, i: usize) -> Result<&Option<Slot>> {
        self.output.get(i).ok_or(crate::Error::OutOfRange(i).into())
    }

    fn input_masks(&self) -> SmallVec<[u64; 4]> {
        SmallVec::from_buf([self.input.0])
    }
}*/

/// The trait representing an arbitrary UI component. The Props associated type
/// must be used to expose the concrete property type that was used to
/// instantiate the component. This is expect to be different, so it is assumed
/// that almost all components are generic over the property type, so long as
/// the property type satisfies the requirements of the chosen layout.
///
/// All components must implement [`StateMachineChild`] even if they are
/// stateless, a derive macro is provided to implement an empty version of the
/// trait for you. In addition, the component must enforce some rather specific
/// constraints due to limitations of the rust type system to properly capture
/// them. See the example for what the simplest possible component looks like.
///
/// # Examples
/// ```
/// use feather_ui::component::{Component};
/// use feather_ui::layout::base;
/// use feather_ui::{ StateMachineChild, SourceID, layout, graphics, StateManager };
/// use std::sync::Arc;
/// use std::rc::Rc;
///
/// // #[derive(feather_macro::StateMachineChild)]
/// // This derive macro simply implements the following. The implementation would be
/// // more complex if our component had children, which the derive macro also handles.
/// impl<T> StateMachineChild for MyComponent<T> {
///     fn id(&self) -> Arc<SourceID> {
///         self.id.clone()
///     }
/// }
///
/// #[derive_where::derive_where(Clone)]
/// pub struct MyComponent<T> {
///     pub id: Arc<SourceID>,
///     pub props: Rc<T>,
/// }
/// impl<T: base::Empty + 'static> Component for MyComponent<T>
/// where
///     for<'a> &'a T: Into<&'a (dyn base::Empty + 'static)>,
/// {
///     type Props = T;
///
///     fn layout(
///         &self,
///         _: &mut StateManager,
///         _: &graphics::Driver,
///         _: &Arc<SourceID>,
///     ) -> Box<dyn layout::Layout<T>> {
///         Box::new(layout::Node::<T, dyn base::Empty> {
///             props: self.props.clone(),
///             children: Default::default(),
///             id: Arc::downgrade(&self.id),
///             renderable: None,
///             layer: None,
///         })
///     }
/// }
/// ```
pub trait Component: crate::StateMachineChild + DynClone {
    type Props;

    fn layout(
        &self,
        state: &mut StateManager,
        driver: &graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<Self::Props>>;
}

dyn_clone::clone_trait_object!(<Parent> Component<Props = Parent> where Parent:?Sized);

pub type ChildOf<D> = dyn ComponentWrap<<D as Desc>::Child>;

pub trait ComponentWrap<T: ?Sized>: crate::StateMachineChild + DynClone {
    fn layout(
        &self,
        state: &mut StateManager,
        driver: &graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>>;
}

dyn_clone::clone_trait_object!(<T> ComponentWrap<T> where T:?Sized);

impl<U: ?Sized, C: Component> ComponentWrap<U> for C
where
    for<'a> &'a U: From<&'a <C as Component>::Props>,
    <C as Component>::Props: Sized + 'static,
{
    fn layout(
        &self,
        state: &mut StateManager,
        driver: &graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<U>> {
        Box::new(Component::layout(self, state, driver, window))
    }
}

impl<T: Component + 'static, U> From<Box<T>> for Box<dyn ComponentWrap<U>>
where
    for<'a> &'a U: std::convert::From<&'a <T as Component>::Props>,
    <T as Component>::Props: Sized,
{
    fn from(value: Box<T>) -> Self {
        value
    }
}

// Stores the root node for the various trees.

pub struct RootState {
    pub(crate) id: Arc<SourceID>,
    layout_tree: Option<Box<dyn crate::layout::Layout<crate::PxDim>>>,
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
    // We currently rely on window-specific functions, so there's no point trying to make this more
    // general right now.
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
        on_driver: &mut Option<Box<dyn FnOnce(std::sync::Weak<graphics::Driver>)>>,
        instance: &wgpu::Instance,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> eyre::Result<()> {
        // Initialize any states that need to be initialized before calling the layout
        // function TODO: make this actually efficient by performing the
        // initialization when a new component is initialized
        for (_, window) in self.children.iter() {
            let window = window.as_ref().unwrap();
            window.init_custom(manager, driver, instance, event_loop, on_driver)?;
            let state: &WindowStateMachine = manager.get(&window.id())?;
            let id = state.state.window.id();
            self.states
                .entry(id)
                .or_insert_with(|| RootState::new(window.id().clone()));

            let root = self
                .states
                .get_mut(&id)
                .ok_or_eyre("Couldn't find window state")?;
            let driver = state.state.driver.clone();
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
            let id = state.state.window.id();
            let root = self
                .states
                .get_mut(&id)
                .ok_or_eyre("Couldn't find window state")?;
            if let Some(layout) = root.layout_tree.as_ref() {
                let layout: &dyn Layout<dyn root::Prop> = &layout.as_ref();
                let staging =
                    layout.stage(Default::default(), Default::default(), &mut state.state);
                root.rtree = staging.get_rtree();
                root.staging = Some(staging);
                state.state.window.request_redraw();
            }
        }
        Ok(())
    }

    pub fn validate_ids(&self) -> eyre::Result<()> {
        struct Validator(std::collections::HashSet<Arc<SourceID>>);
        impl Validator {
            fn f(&mut self, x: &dyn StateMachineChild) -> eyre::Result<()> {
                let id = x.id();
                if !self.0.insert(id.clone()) {
                    return Err(eyre::eyre!(
                        "Duplicate ID found! Did you forget to add a child index to an ID? {}",
                        x.id()
                    ));
                }

                x.apply_children(&mut |x| self.f(x))
            }
        }
        let mut v = Validator(std::collections::HashSet::new());
        for (_, child) in &self.children {
            if let Some(window) = child {
                v.f(window)?;
            }
        }

        Ok(())
    }
}
