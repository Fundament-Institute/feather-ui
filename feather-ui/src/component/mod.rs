// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

pub mod line;
pub mod region;
pub mod window;

use crate::component::window::Window;
use crate::layout::{Desc, DynLayout, Layout, Staged};
use crate::reactive::MutableSignal;
use crate::{
    DispatchPair, Dispatchable, InputResult, PxRect, Slot, SourceID, StateMachineChild,
    StateManager, graphics, rtree,
};
use eyre::{OptionExt, Result};
use smallvec::SmallVec;
use std::any::Any;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

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

pub trait EventRouter
where
    // : zerocopy::Immutable
    Self: Sized,
{
    type Input: Dispatchable;
    type Output: Dispatchable;

    #[allow(unused_variables)]
    #[allow(clippy::type_complexity)]
    fn process(
        state: crate::AccessCell<Self>,
        input: Self::Input,
        area: PxRect,
        extent: PxRect,
        dpi: crate::RelDim,
        driver: &std::sync::Weak<crate::Driver>,
    ) -> InputResult<SmallVec<[Self::Output; 1]>> {
        InputResult::Forward(SmallVec::new())
    }
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
pub trait Component {
    type Props;
    type R: Layout<Props = Self::Props> + 'static;

    fn layout(&self, driver: Arc<graphics::Driver>, dpi: MutableSignal<crate::RelDim>) -> Self::R;
}

pub type ChildOf<D> = dyn DynComponent<<D as Desc>::Child>;

pub trait DynComponent<T: ?Sized> {
    fn layout(
        &self,
        driver: Arc<graphics::Driver>,
        dpi: MutableSignal<crate::RelDim>,
    ) -> Rc<dyn DynLayout<T>>;
}

impl<U: ?Sized, C: Component> DynComponent<U> for C
where
    for<'a> &'a U: From<&'a <C as Component>::Props>,
    <C as Component>::Props: Sized + 'static,
{
    fn layout(
        &self,
        driver: Arc<graphics::Driver>,
        dpi: MutableSignal<crate::RelDim>,
    ) -> Rc<dyn DynLayout<U>> {
        Rc::new(Component::layout(self, driver, dpi))
    }
}

impl<T: Component + 'static, U> From<Box<T>> for Box<dyn DynComponent<U>>
where
    for<'a> &'a U: std::convert::From<&'a <T as Component>::Props>,
    <T as Component>::Props: Sized,
{
    fn from(value: Box<T>) -> Self {
        value
    }
}

#[derive(Clone)]
pub struct UI {
    //children: imbl::Vector<Rc<dyn DynComponent<dyn root::Prop>>>,
    pub children: crate::DynSignal<imbl::Vector<Rc<Window>>>,
}
