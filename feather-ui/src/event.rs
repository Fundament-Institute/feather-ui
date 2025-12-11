// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::{any::Any, cell::RefCell, collections::HashMap, marker::PhantomData, rc::Rc};

use crate::reactive::{SignalMap, SignalProvider};

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub struct EventRes {
    pub cancel: bool,
    pub claim: bool,
}

impl std::ops::BitOr for EventRes {
    type Output = EventRes;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            cancel: self.cancel | rhs.cancel,
            claim: self.claim | rhs.claim,
        }
    }
}

pub trait StreamCallback<T> {
    fn send(&mut self, x: T) -> EventRes;
}

pub trait Unsubscribe<T, S: ?Sized, H: StreamCallback<T>> {
    fn unsubscribe(self) -> (S, H)
    where
        S: Sized,
        H: Sized;
}

pub trait EventStream<'l, T> {
    type Subscription<H: StreamCallback<T> + 'l>: Unsubscribe<T, Self, H>;
    fn subscribe<H: StreamCallback<T> + 'l>(self, h: H) -> Self::Subscription<H>;
}

struct NeverSubscription<T, H> {
    t: PhantomData<T>,
    h: H,
}
impl<T, H: StreamCallback<T>> Unsubscribe<T, NeverStream<T>, H> for NeverSubscription<T, H> {
    fn unsubscribe(self) -> (NeverStream<T>, H)
    where
        H: Sized,
    {
        (NeverStream { t: PhantomData }, self.h)
    }
}
struct NeverStream<T> {
    t: PhantomData<T>,
}
impl<'l, T> EventStream<'l, T> for NeverStream<T> {
    type Subscription<H: StreamCallback<T> + 'l> = NeverSubscription<T, H>;

    fn subscribe<H: StreamCallback<T> + 'l>(self, h: H) -> Self::Subscription<H> {
        NeverSubscription {
            t: PhantomData,
            h: h,
        }
    }
}

struct MapCallback<T, R, F: Fn(T) -> R, H: StreamCallback<R>> {
    f: F,
    h: H,
    phantom: PhantomData<(T, R)>,
}

impl<T, R, F: Fn(T) -> R, H: StreamCallback<R>> StreamCallback<T> for MapCallback<T, R, F, H> {
    fn send(&mut self, x: T) -> EventRes {
        self.h.send((self.f)(x))
    }
}

struct MapSubscription<
    'a,
    T: 'a,
    R: 'a,
    F: 'a + Fn(T) -> R,
    H: 'a + StreamCallback<R>,
    S: EventStream<'a, T>,
> {
    origin: <S as EventStream<'a, T>>::Subscription<MapCallback<T, R, F, H>>,
}

impl<'a, T, R, F: Fn(T) -> R, H: StreamCallback<R>, S: EventStream<'a, T>>
    Unsubscribe<R, MapStream<'a, T, R, F, S>, H> for MapSubscription<'a, T, R, F, H, S>
{
    fn unsubscribe(self) -> (MapStream<'a, T, R, F, S>, H)
    where
        MapStream<'a, T, R, F, S>: Sized,
    {
        let (s, h) = self.origin.unsubscribe();
        (
            MapStream {
                origin: s,
                f: h.f,
                phantom: PhantomData,
            },
            h.h,
        )
    }
}

struct MapStream<'l, T, R, F: Fn(T) -> R, S: EventStream<'l, T>> {
    origin: S,
    f: F,
    phantom: PhantomData<(&'l (), R, T)>,
}

impl<'l, T: 'l, R: 'l, F: 'l + Fn(T) -> R, S: EventStream<'l, T>> EventStream<'l, R>
    for MapStream<'l, T, R, F, S>
{
    type Subscription<H: 'l + StreamCallback<R> + 'l> = MapSubscription<'l, T, R, F, H, S>;

    fn subscribe<H: StreamCallback<R> + 'l>(self, h: H) -> Self::Subscription<H> {
        Self::Subscription {
            origin: self.origin.subscribe(MapCallback {
                f: self.f,
                h: h,
                phantom: PhantomData,
            }),
        }
    }
}

struct FilterCallback<T, F: Fn(&T) -> bool, H: StreamCallback<T>> {
    f: F,
    h: H,
    phantom: PhantomData<T>,
}

impl<T, F: Fn(&T) -> bool, H: StreamCallback<T>> StreamCallback<T> for FilterCallback<T, F, H> {
    fn send(&mut self, x: T) -> EventRes {
        if (self.f)(&x) {
            self.h.send(x)
        } else {
            EventRes {
                cancel: false,
                claim: false,
            }
        }
    }
}

struct FilterSubscription<
    'l,
    T: 'l,
    F: 'l + Fn(&T) -> bool,
    H: 'l + StreamCallback<T>,
    S: EventStream<'l, T>,
> {
    origin: S::Subscription<FilterCallback<T, F, H>>,
}

impl<'l, T, F: Fn(&T) -> bool, H: StreamCallback<T>, S: EventStream<'l, T>>
    Unsubscribe<T, FilterStream<'l, T, F, S>, H> for FilterSubscription<'l, T, F, H, S>
{
    fn unsubscribe(self) -> (FilterStream<'l, T, F, S>, H)
    where
        FilterStream<'l, T, F, S>: Sized,
    {
        let (s, h) = self.origin.unsubscribe();
        (
            FilterStream {
                origin: s,
                f: h.f,
                phantom: PhantomData,
            },
            h.h,
        )
    }
}

struct FilterStream<'l, T, F: Fn(&T) -> bool, S: EventStream<'l, T>> {
    origin: S,
    f: F,
    phantom: PhantomData<&'l T>,
}

impl<'l, T: 'l, F: 'l + Fn(&T) -> bool, S: EventStream<'l, T>> EventStream<'l, T>
    for FilterStream<'l, T, F, S>
{
    type Subscription<H: 'l + StreamCallback<T>> = FilterSubscription<'l, T, F, H, S>;

    fn subscribe<H: StreamCallback<T>>(self, h: H) -> Self::Subscription<H> {
        Self::Subscription {
            origin: self.origin.subscribe(FilterCallback {
                f: self.f,
                h: h,
                phantom: PhantomData,
            }),
        }
    }
}

struct ClaimCallback<T, H: StreamCallback<T>> {
    h: H,
    phantom: PhantomData<T>,
}

impl<T, H: StreamCallback<T>> StreamCallback<T> for ClaimCallback<T, H> {
    fn send(&mut self, x: T) -> EventRes {
        let mut r = self.h.send(x);
        r.claim = true;
        r
    }
}

struct ClaimSubscription<'l, T: 'l, H: 'l + StreamCallback<T>, S: EventStream<'l, T>> {
    origin: S::Subscription<ClaimCallback<T, H>>,
}

impl<'l, T, H: StreamCallback<T>, S: EventStream<'l, T>> Unsubscribe<T, ClaimStream<'l, T, S>, H>
    for ClaimSubscription<'l, T, H, S>
{
    fn unsubscribe(self) -> (ClaimStream<'l, T, S>, H)
    where
        ClaimStream<'l, T, S>: Sized,
    {
        let (s, h) = self.origin.unsubscribe();
        (
            ClaimStream {
                origin: s,
                phantom: PhantomData,
            },
            h.h,
        )
    }
}

struct ClaimStream<'l, T, S: EventStream<'l, T>> {
    origin: S,
    phantom: PhantomData<&'l T>,
}

impl<'l, T: 'l, S: EventStream<'l, T>> EventStream<'l, T> for ClaimStream<'l, T, S> {
    type Subscription<H: 'l + StreamCallback<T> + 'l> = ClaimSubscription<'l, T, H, S>;

    fn subscribe<H: StreamCallback<T>>(self, h: H) -> Self::Subscription<H> {
        Self::Subscription {
            origin: self.origin.subscribe(ClaimCallback {
                h: h,
                phantom: PhantomData,
            }),
        }
    }
}

struct FlattenInnerCallback<T, H: StreamCallback<T>> {
    h: Rc<RefCell<H>>,
    phantom: PhantomData<T>,
}

impl<T, H: StreamCallback<T>> StreamCallback<T> for FlattenInnerCallback<T, H> {
    fn send(&mut self, x: T) -> EventRes {
        self.h.borrow_mut().send(x)
    }
}

struct FlattenOuterCallback<'l, T: EventStream<'l, R>, R: 'l, H: 'l + StreamCallback<R>> {
    h: Rc<RefCell<H>>,
    tracker: Vec<T::Subscription<FlattenInnerCallback<R, H>>>,
}

impl<'l, T: EventStream<'l, R>, R, H: StreamCallback<R>> StreamCallback<T>
    for FlattenOuterCallback<'l, T, R, H>
{
    fn send(&mut self, x: T) -> EventRes {
        let sub = x.subscribe(FlattenInnerCallback {
            h: self.h.clone(),
            phantom: PhantomData,
        });
        self.tracker.push(sub);
        EventRes {
            cancel: false,
            claim: false,
        }
    }
}

struct FlattenSubscription<
    'l,
    T: 'l + EventStream<'l, R>,
    R: 'l,
    H: 'l + StreamCallback<R>,
    S: EventStream<'l, T>,
> {
    origin: S::Subscription<FlattenOuterCallback<'l, T, R, H>>,
}

impl<'l, T: EventStream<'l, R>, R, H: StreamCallback<R>, S: EventStream<'l, T>>
    Unsubscribe<R, FlattenStream<'l, T, R, S>, H> for FlattenSubscription<'l, T, R, H, S>
{
    fn unsubscribe(self) -> (FlattenStream<'l, T, R, S>, H)
    where
        FlattenStream<'l, T, R, S>: Sized,
    {
        let (s, h) = self.origin.unsubscribe();
        let tracker = h
            .tracker
            .into_iter()
            .map(|sub| sub.unsubscribe().0)
            .collect();
        let hh = Rc::try_unwrap(h.h)
            .map_err(|_| ())
            .expect("There Should only be one reference left by now")
            .into_inner();
        (
            FlattenStream {
                origin: s,
                tracker: tracker,
                phantom: PhantomData,
            },
            hh,
        )
    }
}

struct FlattenStream<'l, T: EventStream<'l, R>, R, S: EventStream<'l, T>> {
    origin: S,
    tracker: Vec<T>,
    phantom: PhantomData<&'l R>,
}

impl<'l, T: 'l + EventStream<'l, R>, R: 'l, S: EventStream<'l, T>> EventStream<'l, R>
    for FlattenStream<'l, T, R, S>
{
    type Subscription<H: 'l + StreamCallback<R>> = FlattenSubscription<'l, T, R, H, S>;

    fn subscribe<H: StreamCallback<R>>(self, h: H) -> Self::Subscription<H> {
        let hh = Rc::new(RefCell::new(h));
        let tracker = self
            .tracker
            .into_iter()
            .map(|x| {
                x.subscribe(FlattenInnerCallback {
                    h: hh.clone(),
                    phantom: PhantomData,
                })
            })
            .collect();
        Self::Subscription {
            origin: self.origin.subscribe(FlattenOuterCallback {
                h: hh,
                tracker: tracker,
            }),
        }
    }
}

struct EachCallback<T, F: FnMut(T)> {
    f: F,
    phantom: PhantomData<T>,
}

impl<T, F: FnMut(T)> StreamCallback<T> for EachCallback<T, F> {
    fn send(&mut self, x: T) -> EventRes {
        (self.f)(x);
        EventRes {
            cancel: false,
            claim: false,
        }
    }
}

struct EachSubscription<'l, T: 'l, F: 'l + FnMut(T), S: EventStream<'l, T>> {
    origin: S::Subscription<EachCallback<T, F>>,
    phantom: PhantomData<&'l ()>,
}

impl<'l, T, F: FnMut(T), S: EventStream<'l, T>> EachSubscription<'l, T, F, S> {
    fn unsubscribe(self) -> (S, F) {
        let (s, h) = self.origin.unsubscribe();
        (s, h.f)
    }
}

// FIXME: meaningless contents to make rust build it -- erry
struct Bubbler<'l, T: Clone, Priority: Ord, S: EventStream<'l, T>> {
    t: T,
    pri: Priority,
    s: S,
    phantom: PhantomData<&'l ()>,
}

impl<'l, T: Clone, Priority: Ord, S: EventStream<'l, T>> Bubbler<'l, T, Priority, S> {
    // Get the derived stream with a specific priority; BubbleStream will attempt to dispatch the event to each child Stream in priority order until one of them claims it.
    fn get(self, _: Priority) -> impl EventStream<'l, T> {
        NeverStream { t: PhantomData }
    }
}

trait DynamicStreamCallback<T>: StreamCallback<T> + Any {}
impl<T, H> DynamicStreamCallback<T> for H
where
    H: StreamCallback<T> + 'static,
    H: Any,
{
}

// TODO: Find an alternative to smolset due to `retain()` bug
#[derive(Clone)]
struct DupCallback<T> {
    handlers: Rc<
        RefCell<HashMap<*const dyn DynamicStreamCallback<T>, Box<dyn DynamicStreamCallback<T>>>>,
    >,
}

impl<T: Clone> StreamCallback<T> for DupCallback<T> {
    fn send(&mut self, x: T) -> EventRes {
        let mut claim = false;
        let mut handlers = self.handlers.borrow_mut();
        (*handlers).retain(|_k, handler| {
            let res = handler.as_mut().send(x.clone());
            claim |= res.claim;
            !res.cancel
        });
        EventRes {
            cancel: handlers.is_empty(),
            claim: claim,
        }
    }
}

struct DupSubscription<
    T: Clone + 'static,
    H: StreamCallback<T> + 'static,
    S: EventStream<'static, T>,
> {
    callbackid: *const dyn DynamicStreamCallback<T>,
    state: Rc<RefCell<DupState<T, S>>>,
    phantom: PhantomData<H>,
}

enum DupState<T: Clone + 'static, S: EventStream<'static, T>> {
    NoSubscriptions(S),
    Active(S::Subscription<DupCallback<T>>, DupCallback<T>),
    Invalid,
}

impl<T: Clone + 'static, H: StreamCallback<T> + 'static, S: EventStream<'static, T>>
    Unsubscribe<T, DupStream<T, S>, H> for DupSubscription<T, H, S>
{
    fn unsubscribe(self) -> (DupStream<T, S>, H)
    where
        DupStream<T, S>: Sized,
    {
        let typed_handler = {
            let state = self.state.borrow_mut();
            match &*state {
                DupState::NoSubscriptions(_) => panic!(),
                state_mut @ DupState::Active(s, dup_callback) => {
                    // dup_callback is a &DupCallback<_>
                    let handler = dup_callback
                        .handlers
                        .borrow_mut()
                        .remove(&self.callbackid)
                        .expect("not subscribed?");
                    (handler as Box<dyn Any>)
                        .downcast::<H>()
                        .expect("type error")
                }
                DupState::Invalid => panic!(),
            }
        };

        let stream = DupStream { state: self.state };
        (stream, *typed_handler)
    }
}

struct DupStream<T: Clone + 'static, S: EventStream<'static, T>> {
    state: Rc<RefCell<DupState<T, S>>>,
}

impl<T: Clone, S: EventStream<'static, T>> Clone for DupStream<T, S> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

/*
trait EventStreamOps<T>: EventStream<T> {
    // ... other methods ...

    fn dup(self) -> impl EventStream<T> + Clone
    where
        T: Clone,
        Self: Sized,
        // This constraint only applies when using dup()
        Self: for<'a> EventStream<T, Subscription<&'a mut (dyn StreamCallback<T> + 'static)> = impl Unsubscribe<T, Self, &'a mut (dyn StreamCallback<T> + 'static)>>
    {
        // Implementation
    }
}
*/

impl<T: Clone + 'static, S: EventStream<'static, T>> EventStream<'static, T> for DupStream<T, S> {
    type Subscription<H: StreamCallback<T> + 'static> = DupSubscription<T, H, S>;

    fn subscribe<H: StreamCallback<T> + 'static>(self, h: H) -> Self::Subscription<H> {
        use std::ops::Deref;
        use std::ops::DerefMut;
        let callback: Box<dyn DynamicStreamCallback<T>> = Box::new(h);
        let callbackid: *const dyn DynamicStreamCallback<T> = callback.deref();
        let mut state = self.state.borrow_mut();
        match state.deref_mut() {
            DupState::Active(_s, h) => {
                h.handlers.borrow_mut().insert(callbackid, callback);
            }
            DupState::Invalid => panic!(),
            state_mut @ DupState::NoSubscriptions(_) => {
                let mut handlers = HashMap::new();
                handlers.insert(callbackid, callback);
                let mycallback = DupCallback {
                    handlers: Rc::new(RefCell::new(handlers)),
                };
                let DupState::NoSubscriptions(s) = std::mem::replace(state_mut, DupState::Invalid)
                else {
                    panic!()
                };
                let subscription = s.subscribe(mycallback.clone());
                let DupState::Invalid =
                    std::mem::replace(state_mut, DupState::Active(subscription, mycallback))
                else {
                    panic!()
                };
            }
        }
        return DupSubscription {
            callbackid: callbackid,
            state: self.state.clone(),
            phantom: PhantomData,
        };
    }
}

trait EventStreamExt<'l, T>: EventStream<'l, T> {
    fn map<R: 'l>(self, f: impl Fn(T) -> R + 'l) -> impl EventStream<'l, R>;
    fn filter(self, f: impl Fn(&T) -> bool + 'l) -> impl EventStream<'l, T>;
    fn claim(self) -> impl EventStream<'l, T>;
    fn flatten<R: 'l>(self) -> impl EventStream<'l, R>
    where
        T: for<'a> EventStream<'a, R>;
    fn each<F: FnMut(T) + 'l>(self, f: F) -> EachSubscription<'l, T, F, Self>
    where
        Self: Sized;
}

trait EventStreamClone<'l, T: Clone>: EventStream<'l, T> {
    fn bubble<Priority: Ord>(self) -> Bubbler<'l, T, Priority, Self>
    where
        Self: Sized;
}

trait EventStreamDup<T: Clone + 'static>: EventStream<'static, T> + 'static {
    fn dup(self) -> impl EventStream<'static, T> + Clone;
}

// core operations with no lifetime constraints
impl<'l, S, T: 'l> EventStreamExt<'l, T> for S
where
    S: EventStream<'l, T>,
{
    fn map<R: 'l>(self, f: impl Fn(T) -> R + 'l) -> impl EventStream<'l, R> {
        MapStream {
            origin: self,
            f: f,
            phantom: PhantomData,
        }
    }
    fn filter(self, f: impl Fn(&T) -> bool + 'l) -> impl EventStream<'l, T> {
        FilterStream {
            origin: self,
            f: f,
            phantom: PhantomData,
        }
    }
    fn claim(self) -> impl EventStream<'l, T> {
        ClaimStream {
            origin: self,
            phantom: PhantomData,
        }
    }

    fn flatten<R: 'l>(self) -> impl EventStream<'l, R>
    where
        T: for<'a> EventStream<'a, R>,
    {
        FlattenStream {
            origin: self,
            tracker: Vec::new(),
            phantom: PhantomData,
        }
    }

    fn each<F: FnMut(T)>(self, f: F) -> EachSubscription<'l, T, F, Self>
    where
        Self: Sized,
    {
        EachSubscription {
            origin: self.subscribe(EachCallback {
                f: f,
                phantom: PhantomData,
            }),
            phantom: PhantomData,
        }
    }
}

// operations requiring T: Clone
impl<'l, S, T: Clone> EventStreamClone<'l, T> for S
where
    S: EventStream<'l, T>,
{
    fn bubble<Priority: Ord>(self) -> Bubbler<'l, T, Priority, Self>
    where
        Self: Sized,
    {
        todo!()
    }
}

// operations requiring T: 'static
impl<S, T: Clone + 'static> EventStreamDup<T> for S
where
    S: EventStream<'static, T> + 'static,
{
    fn dup(self) -> impl EventStream<'static, T> + Clone {
        DupStream {
            state: Rc::new(RefCell::new(DupState::NoSubscriptions(self))),
        }
    }
}

pub trait StateMachine<InputEvent, InputState: Clone, OutputEvent, OutputState: Clone> {
    fn update(&mut self, ie: InputEvent, is: InputState) -> impl IntoIterator<Item = OutputEvent>;
    fn get(&self, is: InputState) -> OutputState;
}

fn split_parity(
    num: impl EventStream<'static, i32> + 'static,
) -> (
    impl EventStream<'static, i32>,
    impl EventStream<'static, i32>,
) {
    let nums = num.dup();
    (
        nums.clone().filter(|n| n % 2 == 0),
        nums.clone().filter(|n| n % 2 == 1),
    )
}

pub(crate) fn statemachine<
    'a,
    InputEvent: 'a,
    InputState: Clone + 'a,
    OutputEvent: 'a,
    OutputState: Clone + 'a,
>(
    machine: Rc<RefCell<impl StateMachine<InputEvent, InputState, OutputEvent, OutputState> + 'a>>,
    events: impl EventStream<'a, InputEvent>,
    state: crate::reactive::Signal<impl SignalProvider<Item = InputState> + ?Sized + 'a>,
) -> (
    impl EventStream<'a, OutputEvent>,
    crate::reactive::Signal<impl crate::reactive::SignalProvider<Item = OutputState>>,
) {
    let sig = state;
    let sig2 = sig.clone();
    let machine2 = machine.clone();
    (
        StateMachineStream {
            origin: events,
            m: machine,
            s: sig,
            phantom: PhantomData,
        },
        sig2.map(move |x| machine2.borrow_mut().get(x.clone())),
    )
}

struct StateMachineCallback<
    'a,
    InputEvent: 'a,
    InputState: Clone + 'a,
    OutputEvent: 'a,
    OutputState: Clone + 'a,
    M: StateMachine<InputEvent, InputState, OutputEvent, OutputState> + 'a,
    H: StreamCallback<OutputEvent>,
    P: crate::reactive::SignalProvider<Item = InputState> + ?Sized,
> {
    m: Rc<RefCell<M>>,
    h: H,
    s: crate::reactive::Signal<P>,
    phantom: PhantomData<&'a (InputEvent, OutputEvent, OutputState)>,
}

impl<
    'a,
    InputEvent: 'a,
    IS: Clone + 'a,
    OutputEvent: 'a,
    OS: Clone + 'a,
    M: StateMachine<InputEvent, IS, OutputEvent, OS> + 'a,
    H: StreamCallback<OutputEvent>,
    P: crate::reactive::SignalProvider<Item = IS> + ?Sized,
> StreamCallback<InputEvent>
    for StateMachineCallback<'a, InputEvent, IS, OutputEvent, OS, M, H, P>
{
    fn send(&mut self, x: InputEvent) -> EventRes {
        let mut res = EventRes {
            cancel: false,
            claim: false,
        };
        let mut machine = self.m.borrow_mut();
        for e in machine.update(x, crate::reactive::sample(&self.s).clone()) {
            res = res | self.h.send(e);
        }
        res
    }
}

struct StateMachineSubscription<
    'a,
    InputEvent: 'a,
    IS: Clone + 'a,
    OutputEvent: 'a,
    OS: Clone + 'a,
    M: StateMachine<InputEvent, IS, OutputEvent, OS> + 'a,
    H: StreamCallback<OutputEvent> + 'a,
    P: crate::reactive::SignalProvider<Item = IS> + ?Sized + 'a,
    S: EventStream<'a, InputEvent>,
> {
    origin: <S as EventStream<'a, InputEvent>>::Subscription<
        StateMachineCallback<'a, InputEvent, IS, OutputEvent, OS, M, H, P>,
    >,
}

impl<
    'a,
    InputEvent: 'a,
    IS: Clone + 'a,
    OutputEvent: 'a,
    OS: Clone + 'a,
    M: StateMachine<InputEvent, IS, OutputEvent, OS> + 'a,
    H: StreamCallback<OutputEvent> + 'a,
    P: crate::reactive::SignalProvider<Item = IS> + ?Sized + 'a,
    S: EventStream<'a, InputEvent>,
> Unsubscribe<OutputEvent, StateMachineStream<'a, InputEvent, IS, OutputEvent, OS, M, P, S>, H>
    for StateMachineSubscription<'a, InputEvent, IS, OutputEvent, OS, M, H, P, S>
{
    fn unsubscribe(
        self,
    ) -> (
        StateMachineStream<'a, InputEvent, IS, OutputEvent, OS, M, P, S>,
        H,
    )
    where
        StateMachineStream<'a, InputEvent, IS, OutputEvent, OS, M, P, S>: Sized,
    {
        let (s, h) = self.origin.unsubscribe();
        (
            StateMachineStream {
                origin: s,
                m: h.m,
                s: h.s,
                phantom: PhantomData,
            },
            h.h,
        )
    }
}

struct StateMachineStream<
    'l,
    InputEvent: 'l,
    IS: Clone + 'l,
    OutputEvent: 'l,
    OS: Clone + 'l,
    M: StateMachine<InputEvent, IS, OutputEvent, OS> + 'l,
    P: crate::reactive::SignalProvider<Item = IS> + ?Sized,
    S: EventStream<'l, InputEvent>,
> {
    origin: S,
    m: Rc<RefCell<M>>,
    s: crate::reactive::Signal<P>,
    phantom: PhantomData<&'l (InputEvent, OutputEvent, OS)>,
}

impl<
    'l,
    InputEvent: 'l,
    IS: Clone + 'l,
    OutputEvent: 'l,
    OS: Clone + 'l,
    M: StateMachine<InputEvent, IS, OutputEvent, OS> + 'l,
    P: crate::reactive::SignalProvider<Item = IS> + ?Sized + 'l,
    S: EventStream<'l, InputEvent>,
> EventStream<'l, OutputEvent>
    for StateMachineStream<'l, InputEvent, IS, OutputEvent, OS, M, P, S>
{
    type Subscription<H: 'l + StreamCallback<OutputEvent> + 'l> =
        StateMachineSubscription<'l, InputEvent, IS, OutputEvent, OS, M, H, P, S>;

    fn subscribe<H: StreamCallback<OutputEvent> + 'l>(self, h: H) -> Self::Subscription<H> {
        Self::Subscription {
            origin: self.origin.subscribe(StateMachineCallback {
                m: self.m,
                h: h,
                s: self.s,
                phantom: PhantomData,
            }),
        }
    }
}
