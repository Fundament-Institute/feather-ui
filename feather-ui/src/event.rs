// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use either::Either;
use smallvec::SmallVec;
use std::{cell::RefCell, collections::HashSet, rc::Rc};

use crate::{AccessCell, Dispatchable, InputResult, PxRect};

/// An event functions very similar to a Result except it also has an explicit End variant
pub enum Event<T> {
    Value(T),
    Err(eyre::ErrReport),
    End,
}

impl<T> Event<T> {
    fn map<U>(self, f: impl FnOnce(T) -> U) -> Event<U> {
        match self {
            Event::Value(v) => Event::Value(f(v)),
            Event::Err(report) => Event::Err(report),
            Event::End => Event::End,
        }
    }
}

// : zerocopy::Immutable
pub trait EventRouter
where
    Self: std::marker::Sized,
{
    type Input: Dispatchable + 'static;
    type Output: Dispatchable + 'static;

    #[allow(unused_variables)]
    #[allow(clippy::type_complexity)]
    fn process(
        state: AccessCell<Self>,
        input: Self::Input,
        area: PxRect,
        extent: PxRect,
        dpi: crate::RelDim,
        driver: &std::sync::Weak<crate::Driver>,
    ) -> InputResult<SmallVec<[Self::Output; 1]>> {
        InputResult::Forward(SmallVec::new())
    }
}

// TODO: May need an eventstreamwrap trait or object that encapsulates an eventstream which can translate
// from a DispatchPair to the specific input type.
// EventStreamWrap<Event, Input> where Input: std::convert::TryFrom<Event>

/// An event stream corresponds to a single output slot of an event router, which in feather's case
/// always corresponds to a specific enum type. Conceptually, an EventStream is essentially a push version
/// of an Iterator. As a result, most of it's methods mirror the Iterator's methods, with modified
/// signatures to account for EventStreams being push-based rather than pull-based.
pub trait EventStream<V: 'static> {
    fn buffer(&mut self, n: usize, sink: Rc<dyn Sink<[V]>>) {
        todo!()
    }
    fn unsubscribe(&mut self, sink: Rc<dyn Sink<Event<V>>>) -> bool;
    fn subscribe(&mut self, sink: Rc<dyn Sink<Event<V>>>) -> bool;
    fn on_value(&mut self, sink: Rc<dyn Sink<V>>) -> bool;
    fn on_error(&mut self, sink: Rc<dyn Sink<eyre::ErrReport>>) -> bool;
    fn on_end(&mut self, sink: Rc<dyn Sink<()>>) -> bool;
    fn scan(&mut self);
    fn merge<U: 'static>(
        &mut self,
        other: &mut dyn EventStream<U>,
    ) -> impl EventStream<Either<V, U>>
    where
        Self: Sized,
    {
        let e: Rc<EventPass<Either<V, U>>> = Rc::new(EventPass::new());

        let sink: Rc<dyn Sink<Event<Either<V, U>>>> = e.clone();
        let (l, r) = sink.split();
        self.subscribe(Rc::new(l));
        other.subscribe(Rc::new(r));
        e
    }

    //fn map<T>(&self, f: impl Fn(V) -> T) -> impl EventStream<T> {}
}

/// An EventBus is simply an eventstream that you can push to.
trait EventBus<V: 'static>: EventStream<V> {
    fn plug(&self, evt: &mut dyn EventStream<V>);
    fn end(&self);
}

//impl<V> EventStream<V> for dyn Iterator<Item = V> {}

/// This represents anything that can subscribe to a particular type.
trait Sink<T> {
    fn push(&self, value: T) -> bool;
}

impl<T, F: Fn(T) -> bool> Sink<T> for F {
    fn push(&self, value: T) -> bool {
        self(value)
    }
}

#[derive(Clone)]
enum EventListener<V> {
    Event(Rc<dyn Sink<Event<V>>>),
    Value(Rc<dyn Sink<V>>),
    Error(Rc<dyn Sink<eyre::ErrReport>>),
    End(Rc<dyn Sink<()>>),
}

impl<T> PartialEq for EventListener<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Event(l0), Self::Event(r0)) => Rc::ptr_eq(l0, r0),
            (Self::Value(l0), Self::Value(r0)) => Rc::ptr_eq(l0, r0),
            (Self::Error(l0), Self::Error(r0)) => Rc::ptr_eq(l0, r0),
            (Self::End(l0), Self::End(r0)) => Rc::ptr_eq(l0, r0),
            _ => false,
        }
    }
}

impl<T> Eq for EventListener<T> {}

impl<T> std::hash::Hash for EventListener<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            EventListener::Event(sink) => Rc::as_ptr(sink).hash(state),
            EventListener::Value(sink) => Rc::as_ptr(sink).hash(state),
            EventListener::Error(sink) => Rc::as_ptr(sink).hash(state),
            EventListener::End(sink) => Rc::as_ptr(sink).hash(state),
        }
    }
}

pub struct EventPass<V>(RefCell<Option<Rc<dyn Sink<Event<V>>>>>);

impl<V> EventPass<V> {
    pub fn new() -> Self {
        Self(RefCell::new(None))
    }
}

impl<V: 'static> EventStream<V> for Rc<EventPass<V>> {
    fn unsubscribe(&mut self, sink: Rc<dyn Sink<Event<V>>>) -> bool {
        if let Some(r) = self.0.borrow_mut().as_mut()
            && Rc::ptr_eq(&r, &sink)
        {
            self.0.borrow_mut().take();
            true
        } else {
            false
        }
    }

    fn subscribe(&mut self, sink: Rc<dyn Sink<Event<V>>>) -> bool {
        if self.0.borrow().is_none() {
            self.0.borrow_mut().replace(sink);
            true
        } else {
            false
        }
    }

    fn on_value(&mut self, sink: Rc<dyn Sink<V>>) -> bool {
        todo!()
    }

    fn on_error(&mut self, sink: Rc<dyn Sink<eyre::ErrReport>>) -> bool {
        todo!()
    }

    fn on_end(&mut self, sink: Rc<dyn Sink<()>>) -> bool {
        todo!()
    }

    fn scan(&mut self) {
        todo!()
    }
}

impl<V> Sink<Event<V>> for EventPass<V> {
    fn push(&self, value: Event<V>) -> bool {
        if let Some(sink) = self.0.borrow().as_ref() {
            sink.push(value)
        } else {
            false
        }
    }
}

impl<V: 'static> EventBus<V> for Rc<EventPass<V>> {
    fn plug(&self, evt: &mut dyn EventStream<V>) {
        evt.subscribe(self.clone());
    }

    fn end(&self) {
        if let Some(sink) = self.0.borrow().as_ref() {
            sink.push(Event::End);
        }
    }
}
// TODO: eventimpl should really be an eventrouter
pub struct EventHop<V> {
    listeners: std::cell::RefCell<HashSet<EventListener<V>>>,
}

impl<V> EventHop<V> {}

impl<V: 'static> EventStream<V> for Rc<EventHop<V>> {
    fn unsubscribe(&mut self, sink: Rc<dyn Sink<Event<V>>>) -> bool {
        self.listeners
            .borrow_mut()
            .remove(&EventListener::Event(sink))
    }

    fn subscribe(&mut self, sink: Rc<dyn Sink<Event<V>>>) -> bool {
        self.listeners
            .borrow_mut()
            .insert(EventListener::Event(sink))
    }

    fn on_value(&mut self, sink: Rc<dyn Sink<V>>) -> bool {
        self.listeners
            .borrow_mut()
            .insert(EventListener::Value(sink))
    }

    fn on_error(&mut self, sink: Rc<dyn Sink<eyre::ErrReport>>) -> bool {
        self.listeners
            .borrow_mut()
            .insert(EventListener::Error(sink))
    }

    fn on_end(&mut self, sink: Rc<dyn Sink<()>>) -> bool {
        self.listeners.borrow_mut().insert(EventListener::End(sink))
    }

    fn scan(&mut self) {
        todo!()
    }
}

impl<V: Clone + 'static> EventBus<V> for Rc<EventHop<V>> {
    fn plug(&self, evt: &mut dyn EventStream<V>) {
        evt.on_value(self.clone());
    }

    fn end(&self) {
        for l in self.listeners.borrow().iter() {
            if let EventListener::End(v) = l {
                v.push(());
            }
        }
    }
}

impl<V: Clone> Sink<V> for EventHop<V> {
    fn push(&self, value: V) -> bool {
        for v in self.listeners.borrow().iter() {
            match v {
                EventListener::Value(sink) => sink.push(value.clone()),
                EventListener::Event(sink) => sink.push(Event::Value(value.clone())),
                _ => false,
            };
        }

        return true;
    }
}

pub struct LSink<L, R>(Rc<dyn Sink<Event<Either<L, R>>>>);
pub struct RSink<L, R>(Rc<dyn Sink<Event<Either<L, R>>>>);

impl<L, R> Sink<Event<L>> for LSink<L, R> {
    fn push(&self, value: Event<L>) -> bool {
        self.0.push(value.map(Either::Left))
    }
}

impl<L, R> Sink<Event<R>> for RSink<L, R> {
    fn push(&self, value: Event<R>) -> bool {
        self.0.push(value.map(Either::Right))
    }
}

impl<L, R> dyn Sink<Event<Either<L, R>>> {
    fn split(self: Rc<Self>) -> (LSink<L, R>, RSink<L, R>) {
        (LSink(self.clone()), RSink(self.clone()))
    }
}
