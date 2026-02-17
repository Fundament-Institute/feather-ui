// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

pub mod button;
pub mod line;
pub mod mouse_area;
pub mod region;
pub mod shape;
pub mod text;
pub mod window;

use crate::component::window::Window;
use crate::graphics;
use crate::layout::{Desc, DynLayout, Layout};
use crate::reactive::MutableSignal;
use std::rc::Rc;
use std::sync::Arc;

pub trait ComponentMarker {}

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

    fn layout(&self, driver: &Arc<graphics::Driver>, dpi: MutableSignal<crate::RelDim>) -> Self::R;
}

impl<T: Component> ComponentMarker for T {}

pub type ChildOf<D> = dyn DynComponent<<D as Desc>::Child>;

pub trait DynComponent<T: ?Sized> {
    fn layout(
        &self,
        driver: &Arc<graphics::Driver>,
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
        driver: &Arc<graphics::Driver>,
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

/*impl<U: ?Sized> DynComponent<U> for () {
    fn layout(
        &self,
        _: &Arc<graphics::Driver>,
        _: MutableSignal<crate::RelDim>,
    ) -> Rc<dyn DynLayout<U>> {
        panic!("Component was already processed!")
    }
}*/

#[derive(Clone)]
pub struct UI {
    //children: imbl::Vector<Rc<dyn DynComponent<dyn root::Prop>>>,
    pub children: crate::DynSignal<imbl::Vector<Rc<Window>>>,
}
