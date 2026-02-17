// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::mouse_area::MouseArea;

use crate::component::ChildOf;
use crate::component::mouse_area::{MouseAreaEvent, MouseAreaState};
use crate::event;
use crate::layout::fixed;
use crate::reactive::{DynSignal, MutableSignal, Signal, SignalProvider};
use std::rc::Rc;
use std::sync::Arc;

// A button component that contains a mousearea alongside it's children
pub struct Button<T> {
    props: Rc<T>,
    children: DynSignal<imbl::Vector<Rc<ChildOf<dyn fixed::Prop>>>>,
}

impl<T: fixed::Prop> Button<T> {
    pub fn new<P1: SignalProvider<Item = f32> + ?Sized + 'static>(
        props: T,
        deadzone: Signal<P1>,
        children: DynSignal<imbl::Vector<Rc<ChildOf<dyn fixed::Prop>>>>,
    ) -> (
        Self,
        impl event::EventStream<'static, MouseAreaEvent>,
        Signal<impl SignalProvider<Item = MouseAreaState>>,
    ) {
        let (marea, evt, export) = MouseArea::new(crate::FILL_DRECT, deadzone);
        let marea = Rc::new(marea);
        (
            Self {
                props: props.into(),
                children: children
                    .map_modify(move |x| {
                        let mut v = x.clone();
                        v.push_back(marea.clone());
                        v
                    })
                    .into(),
            },
            evt,
            export,
        )
    }
}

impl<T: fixed::Prop + 'static> super::Component for Button<T>
where
    for<'a> &'a T: Into<&'a (dyn fixed::Prop + 'static)>,
{
    type Props = T;
    type R = fixed::Layer<T, ()>;

    fn layout(
        &self,
        driver: &Arc<crate::graphics::Driver>,
        dpi2: MutableSignal<crate::RelDim>,
    ) -> Self::R {
        let wdriver = Arc::downgrade(&driver);
        let dpi = dpi2.clone();
        let children = self.children.clone().map_elements(
            move |child: &Rc<ChildOf<dyn fixed::Prop>>| {
                child.layout(&wdriver.upgrade().unwrap(), dpi.clone())
            },
            |x| crate::reactive::Identity(x.clone()),
        );

        Self::R {
            props: self.props.clone(),
            children: children.into(),
            renderable: None,
            layer: None,
            machine: None,
        }
        .into()
    }
}
