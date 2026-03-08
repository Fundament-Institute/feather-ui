// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use small_map::SmallMap;

use crate::reactive::{self, Signal, SignalProvider};
use crate::{DispatchCallback, Dispatchable};

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub struct EventRes {
    pub cancel: bool,
    pub claim: bool,
}

pub const CONSUME: EventRes = EventRes {
    cancel: false,
    claim: true,
};

pub const FORWARD: EventRes = EventRes {
    cancel: false,
    claim: false,
};

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

impl<U: FnMut(T) -> EventRes, T> StreamCallback<T> for U {
    fn send(&mut self, x: T) -> EventRes {
        self(x)
    }
}

pub trait Unsubscribe<T, S: ?Sized, H: StreamCallback<T> + ?Sized> {
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
    type Subscription<H: StreamCallback<R> + 'l> = MapSubscription<'l, T, R, F, H, S>;

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
    type Subscription<H: StreamCallback<T> + 'l> = ClaimSubscription<'l, T, H, S>;

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

pub struct EachSubscription<'l, T: 'l, F: 'l + FnMut(T), S: EventStream<'l, T>> {
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
pub struct Bubbler<'l, T: Clone, Priority: Ord, S: EventStream<'l, T>> {
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

// TODO: replace with smallmap form https://github.com/rinde/more_collections
struct DupCallback<'l, T> {
    queue: Rc<
        RefCell<SmallMap<4, *const (dyn StreamCallback<T> + 'l), Box<dyn StreamCallback<T> + 'l>>>,
    >,
}

impl<'l, T> DupCallback<'l, T> {
    fn new() -> Self {
        Self {
            queue: Rc::new(SmallMap::new().into()),
        }
    }
}

impl<'l, T: Clone> StreamCallback<T> for DupCallback<'l, T> {
    fn send(&mut self, x: T) -> EventRes {
        let mut claim = false;
        self.queue.borrow_mut().retain(|_, h| {
            let res = h.send(x.clone());
            claim |= res.claim;
            !res.cancel
        });
        EventRes {
            cancel: self.queue.borrow().is_empty(),
            claim: claim,
        }
    }
}

enum DupState<'l, T: Clone + 'l, S: EventStream<'l, T>> {
    NoSubscriptions(S),
    Active(
        S::Subscription<DupCallback<'l, T>>,
        std::rc::Weak<
            RefCell<
                SmallMap<4, *const (dyn StreamCallback<T> + 'l), Box<dyn StreamCallback<T> + 'l>>,
            >,
        >,
    ),
    Invalid,
}

impl<'l, T: Clone + 'l, S: EventStream<'l, T>> DupState<'l, T, S> {
    fn deactivate(&mut self) {
        let mut state = DupState::Invalid;
        std::mem::swap(self, &mut state);
        match state {
            DupState::Active(s, _) => {
                let (dup, _) = s.unsubscribe();
                state = Self::NoSubscriptions(dup);
            }
            _ => (),
        }
        std::mem::swap(self, &mut state);
    }

    fn unsubscribe<H>(&mut self, k: *const (dyn StreamCallback<T> + 'l)) -> Option<H> {
        match self {
            DupState::NoSubscriptions(_) | DupState::Invalid => None,
            DupState::Active(_, callback) => {
                let c = callback.upgrade()?;
                let boxed = c.borrow_mut().remove(&k)?;

                if c.borrow().is_empty() {
                    self.deactivate();
                }

                let raw = Box::into_raw(boxed) as *mut H;
                Some(*unsafe { Box::<H>::from_raw(raw) })
            }
        }
    }
}

struct DupSubscription<'l, T: Clone + 'l, H: StreamCallback<T> + 'l, S: EventStream<'l, T>> {
    state: Rc<RefCell<DupState<'l, T, S>>>,
    key: *const (dyn StreamCallback<T> + 'l),
    phantom: PhantomData<H>,
}

impl<'l, T: Clone + 'l, H: StreamCallback<T> + 'l, S: EventStream<'l, T>>
    Unsubscribe<T, DupStream<'l, T, S>, H> for DupSubscription<'l, T, H, S>
{
    fn unsubscribe(self) -> (DupStream<'l, T, S>, H) {
        let h = self
            .state
            .borrow_mut()
            .unsubscribe(self.key)
            .expect("Tried to unsubscribe from invalid DupState!");

        (DupStream { state: self.state }, h)
    }
}

struct DupStream<'l, T: Clone + 'l, S: EventStream<'l, T>> {
    state: Rc<RefCell<DupState<'l, T, S>>>,
}

impl<'l, T: Clone, S: EventStream<'l, T>> Clone for DupStream<'l, T, S> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<'l, T: Clone + 'l, S: EventStream<'l, T>> EventStream<'l, T> for DupStream<'l, T, S> {
    type Subscription<H: StreamCallback<T> + 'l> = DupSubscription<'l, T, H, S>;

    fn subscribe<H: StreamCallback<T> + 'l>(self, h: H) -> Self::Subscription<H> {
        let boxed: Box<dyn StreamCallback<T> + 'l> = Box::new(h);
        let p = boxed.as_ref() as *const dyn StreamCallback<T>;
        let mut state = DupState::Invalid;
        std::mem::swap(&mut *self.state.borrow_mut(), &mut state);
        match state {
            DupState::NoSubscriptions(s) => {
                let callback = DupCallback::new();
                callback.queue.borrow_mut().insert(p, boxed);

                let queue = Rc::downgrade(&callback.queue);
                state = DupState::Active(s.subscribe(callback), queue);
            }
            DupState::Active(_, ref mut queue) => {
                let q = queue
                    .upgrade()
                    .expect("Tried to subscribe to nonexistent callback!");
                q.borrow_mut().insert(p, boxed);
            }
            DupState::Invalid => panic!("Tried to subscribe to invalid dupstream!"),
        }

        std::mem::swap(&mut *self.state.borrow_mut(), &mut state);

        DupSubscription {
            state: self.state.clone(),
            key: p,
            phantom: PhantomData,
        }
    }
}

pub trait EventStreamExt<'l, T>: EventStream<'l, T> {
    fn map<R: 'l>(self, f: impl Fn(T) -> R + 'l) -> impl EventStream<'l, R>;
    fn filter(self, f: impl Fn(&T) -> bool + 'l) -> impl EventStream<'l, T>;
    fn claim(self) -> impl EventStream<'l, T>;
    fn flatten<R: 'l>(self) -> impl EventStream<'l, R>
    where
        T: for<'a> EventStream<'a, R>;
    fn each<F: FnMut(T) + 'l>(self, f: F) -> EachSubscription<'l, T, F, Self>
    where
        Self: Sized;
    fn narrow<U: core::convert::TryFrom<T>>(self) -> NarrowStream<'l, T, U, Self>
    where
        Self: Sized;
}

pub trait EventStreamClone<'l, T: Clone>: EventStream<'l, T> {
    fn bubble<Priority: Ord>(self) -> Bubbler<'l, T, Priority, Self>
    where
        Self: Sized;
    fn dup(self) -> impl EventStream<'l, T> + Clone;
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

    fn narrow<U: core::convert::TryFrom<T>>(self) -> NarrowStream<'l, T, U, Self>
    where
        Self: Sized,
    {
        NarrowStream {
            origin: self,
            phantom: PhantomData,
        }
    }
}

// operations requiring T: Clone
impl<'l, S, T: Clone + 'l> EventStreamClone<'l, T> for S
where
    S: EventStream<'l, T>,
{
    fn bubble<Priority: Ord>(self) -> Bubbler<'l, T, Priority, Self>
    where
        Self: Sized,
    {
        todo!()
    }
    fn dup(self) -> impl EventStream<'l, T> + Clone {
        DupStream {
            state: Rc::new(RefCell::new(DupState::NoSubscriptions(self))),
        }
    }
}

struct NarrowCallback<T, U, H: StreamCallback<U>> {
    h: H,
    phantom: PhantomData<(T, U)>,
}

impl<T, U: core::convert::TryFrom<T>, H: StreamCallback<U>> StreamCallback<T>
    for NarrowCallback<T, U, H>
{
    fn send(&mut self, x: T) -> EventRes {
        if let Ok(v) = x.try_into() {
            self.h.send(v)
        } else {
            FORWARD
        }
    }
}

pub struct NarrowSubscription<
    'a,
    T: 'a,
    R: 'a + core::convert::TryFrom<T>,
    H: 'a + StreamCallback<R>,
    S: EventStream<'a, T>,
> {
    origin: <S as EventStream<'a, T>>::Subscription<NarrowCallback<T, R, H>>,
}

impl<'a, T, R: core::convert::TryFrom<T>, H: StreamCallback<R>, S: EventStream<'a, T>>
    Unsubscribe<R, NarrowStream<'a, T, R, S>, H> for NarrowSubscription<'a, T, R, H, S>
{
    fn unsubscribe(self) -> (NarrowStream<'a, T, R, S>, H)
    where
        NarrowStream<'a, T, R, S>: Sized,
    {
        let (s, h) = self.origin.unsubscribe();
        (
            NarrowStream {
                origin: s,
                phantom: PhantomData,
            },
            h.h,
        )
    }
}

pub struct NarrowStream<'l, T, R, S: EventStream<'l, T>> {
    origin: S,
    phantom: PhantomData<(&'l R, T)>,
}

impl<'l, T: 'l, R: 'l + core::convert::TryFrom<T>, S: EventStream<'l, T>> EventStream<'l, R>
    for NarrowStream<'l, T, R, S>
{
    type Subscription<H: StreamCallback<R> + 'l> = NarrowSubscription<'l, T, R, H, S>;

    fn subscribe<H: StreamCallback<R> + 'l>(self, h: H) -> Self::Subscription<H> {
        Self::Subscription {
            origin: self.origin.subscribe(NarrowCallback {
                h: h,
                phantom: PhantomData,
            }),
        }
    }
}

enum PrismState<'l, T: 'l + Dispatchable, S: 'l + EventStream<'l, T>> {
    Origin(S),
    Subscription(S::Subscription<PrismCallback<'l, T, S>>, T::Callback, usize),
    Invalid,
}

impl<'l, T: 'l + Dispatchable, S: EventStream<'l, T>> PrismState<'l, T, S> {
    fn subscribe(this: &Rc<RefCell<Self>>) {
        match &mut *this.borrow_mut() {
            Self::Origin(_) => (),
            Self::Subscription(_, _, count) => {
                *count += 1;
                return;
            }
            _ => return,
        }

        let mut state = PrismState::Invalid;
        std::mem::swap(&mut *this.borrow_mut(), &mut state);
        if let Self::Origin(origin) = state {
            state = Self::Subscription(
                origin.subscribe(PrismCallback {
                    state: this.clone(),
                }),
                T::callback(),
                1,
            );
        }
        std::mem::swap(&mut *this.borrow_mut(), &mut state);
    }

    fn unsubscribe(&mut self) {
        match self {
            Self::Subscription(_, _, count) => {
                *count -= 1;
                if *count > 0 {
                    return;
                }
            }
            Self::Origin(_) => panic!("Tried to double-unsubscribe from PrismState!"),
            Self::Invalid => panic!("Invalid PrismState!"),
        }

        let mut state = PrismState::Invalid;
        std::mem::swap(self, &mut state);
        if let Self::Subscription(s, _, count) = state {
            assert_eq!(count, 0);
            let (origin, _) = s.unsubscribe();
            state = Self::Origin(origin);
        }
        std::mem::swap(self, &mut state);
    }
}

/// Due to restrictions in the type system, we can't have some eventstreams impose an extra std::any::Any restriction
/// on EventStreams, so instead we use a Box paired with a type_id to build our own knockoff Any.
pub struct BoxedCallback<T>(pub Box<dyn StreamCallback<T>>, std::any::TypeId);

impl<T> std::fmt::Debug for BoxedCallback<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ptr = self.0.as_ref() as *const dyn StreamCallback<T>;
        f.debug_tuple(format!("BoxedCallback<{}>", std::any::type_name::<T>()).as_str())
            .field(&ptr)
            .field(&self.1)
            .finish()
    }
}

impl<T> BoxedCallback<T> {
    pub fn new<H: StreamCallback<T> + 'static>(h: H) -> Self {
        let boxed: Box<dyn StreamCallback<T>> = Box::new(h);

        Self(boxed, std::any::TypeId::of::<H>())
    }

    pub fn unbox<H: StreamCallback<T> + 'static>(self) -> H {
        // This ensures the pointer we extract out is the type we expect
        assert_eq!(self.1, std::any::TypeId::of::<H>());
        let raw = Box::into_raw(self.0) as *mut H;

        // Using from_raw here is important because manually deallocating will segfault if H is zero-sized.
        *unsafe { Box::<H>::from_raw(raw) }
    }
}

struct PrismCallback<'l, T: 'l + Dispatchable, S: EventStream<'l, T>> {
    state: Rc<RefCell<PrismState<'l, T, S>>>,
}

impl<'l, T: 'l + Dispatchable, S: EventStream<'l, T>> StreamCallback<T>
    for PrismCallback<'l, T, S>
{
    fn send(&mut self, x: T) -> EventRes {
        if let PrismState::Subscription(_, h, _) = &mut *self.state.borrow_mut() {
            T::send(x, h)
        } else {
            FORWARD
        }
    }
}

pub struct PrismSubscription<'l, T: Dispatchable, R, H: StreamCallback<R>, S: EventStream<'l, T>> {
    origin: Rc<RefCell<PrismState<'l, T, S>>>,
    phantom: PhantomData<(R, H)>,
}

impl<
    'a,
    T: Dispatchable,
    R: DispatchCallback<T>,
    H: 'static + StreamCallback<R>,
    S: EventStream<'a, T>,
> Unsubscribe<R, PrismStream<'a, T, R, S>, H> for PrismSubscription<'a, T, R, H, S>
{
    fn unsubscribe(self) -> (PrismStream<'a, T, R, S>, H)
    where
        PrismStream<'a, T, R, S>: Sized,
    {
        let h = if let PrismState::Subscription(_, callback, _) = &mut *self.origin.borrow_mut() {
            R::unsubscribe::<H>(callback)
        } else {
            panic!("Invalid prism state!")
        };

        self.origin.borrow_mut().unsubscribe();
        return (
            PrismStream::<'a, T, R, S> {
                origin: self.origin.clone(),
                phantom: PhantomData,
            },
            h,
        );
    }
}

pub struct PrismStream<'l, T: Dispatchable, R, S: EventStream<'l, T>> {
    origin: Rc<RefCell<PrismState<'l, T, S>>>,
    phantom: PhantomData<R>,
}

#[repr(transparent)]
pub struct PrismInternal<'l, T: 'l + Dispatchable, S: 'l + EventStream<'l, T>>(
    Rc<RefCell<PrismState<'l, T, S>>>,
);

impl<'l, T: Dispatchable, S: EventStream<'l, T>> PrismInternal<'l, T, S> {
    pub fn new(s: S) -> Self {
        Self(Rc::new(RefCell::new(PrismState::Origin(s))))
    }
}

impl<'l, T: 'l + Dispatchable, R: 'l, S: EventStream<'l, T>> PrismStream<'l, T, R, S> {
    pub fn new(s: &PrismInternal<'l, T, S>) -> Self {
        Self {
            origin: s.0.clone(),
            phantom: PhantomData,
        }
    }
}

impl<'l, T: 'l + Dispatchable, R: 'l + DispatchCallback<T>, S: EventStream<'l, T>>
    EventStream<'static, R> for PrismStream<'l, T, R, S>
{
    type Subscription<H: StreamCallback<R> + 'static> = PrismSubscription<'l, T, R, H, S>;

    fn subscribe<H: StreamCallback<R> + 'static>(self, h: H) -> Self::Subscription<H> {
        PrismState::subscribe(&self.origin);

        if let PrismState::Subscription(_, callback, _) = &mut *self.origin.borrow_mut() {
            R::subscribe(h, callback);
        }

        Self::Subscription {
            origin: self.origin,
            phantom: PhantomData,
        }
    }
}

pub trait EventStreamPrism<'l, S: EventStream<'l, T>, T: Dispatchable>: EventStream<'l, T> {
    fn prism(self) -> T::Prism<'l, S>
    where
        Self: Sized;
}

// operations requiring T: Dispatchable
impl<'l, S, T: Dispatchable + 'l> EventStreamPrism<'l, S, T> for S
where
    S: EventStream<'l, T>,
{
    fn prism(self) -> T::Prism<'l, S>
    where
        Self: Sized,
    {
        T::prism(self)
    }
}

impl Dispatchable for std::convert::Infallible {
    type Prism<'l, T: 'l + EventStream<'l, Self>> = ();
    type Callback = ();

    fn callback() -> Self::Callback {
        ()
    }

    fn send(self, _: &mut Self::Callback) -> crate::event::EventRes {
        FORWARD
    }

    fn prism<'a, S: 'a + crate::event::EventStream<'a, Self>>(_: S) -> Self::Prism<'a, S> {
        ()
    }
}

pub trait StateMachine<InputEvent, InputState, OutputEvent, OutputState> {
    fn update(
        &mut self,
        event: InputEvent,
        input: &InputState,
    ) -> (impl IntoIterator<Item = OutputEvent>, EventRes);
    fn get(&self, input: &InputState) -> OutputState;
}

pub(crate) fn statemachine<
    'a,
    InputEvent: 'a,
    InputState: 'a,
    OutputEvent: 'a,
    OutputState: Clone + reactive::SignalDebug + 'a,
>(
    machine: impl StateMachine<InputEvent, InputState, OutputEvent, OutputState> + 'a,
    events: impl EventStream<'a, InputEvent>,
    state: Signal<impl SignalProvider<Item = InputState> + ?Sized + 'a>,
) -> (
    impl EventStream<'a, OutputEvent>,
    Signal<impl SignalProvider<Item = OutputState>>,
) {
    let input = state;
    let m = Rc::new(RefCell::new(machine));
    let machine2 = m.clone();
    let output = input.clone().map_ex(move |x| machine2.borrow_mut().get(x));
    (
        StateMachineStream {
            origin: events,
            m,
            input,
            output: output.clone(),
            phantom: PhantomData,
        },
        output,
    )
}

struct StateMachineCallback<
    'a,
    InputEvent: 'a,
    InputState: 'a,
    OutputEvent: 'a,
    OutputState: Clone + 'a,
    M: StateMachine<InputEvent, InputState, OutputEvent, OutputState> + 'a,
    H: StreamCallback<OutputEvent>,
    IP: SignalProvider<Item = InputState> + ?Sized + 'a,
    OP: SignalProvider<Item = OutputState> + ?Sized + 'a,
> {
    m: Rc<RefCell<M>>,
    h: H,
    input: Signal<IP>,
    output: Signal<OP>,
    phantom: PhantomData<&'a (InputEvent, OutputEvent)>,
}

impl<
    'a,
    InputEvent: 'a,
    IS: 'a,
    OutputEvent: 'a,
    OS: Clone + 'a,
    M: StateMachine<InputEvent, IS, OutputEvent, OS> + 'a,
    H: StreamCallback<OutputEvent>,
    IP: SignalProvider<Item = IS> + ?Sized,
    OP: SignalProvider<Item = OS> + ?Sized,
> StreamCallback<InputEvent>
    for StateMachineCallback<'a, InputEvent, IS, OutputEvent, OS, M, H, IP, OP>
{
    fn send(&mut self, x: InputEvent) -> EventRes {
        let mut machine = self.m.borrow_mut();
        let input = crate::reactive::sample(&self.input);
        let (events, mut res) = machine.update(x, &input);
        for e in events {
            res = res | self.h.send(e);
        }
        std::mem::drop(input);
        crate::reactive::notify_change(&self.output);
        res
    }
}

struct StateMachineSubscription<
    'a,
    InputEvent: 'a,
    IS: 'a,
    OutputEvent: 'a,
    OS: Clone + 'a,
    M: StateMachine<InputEvent, IS, OutputEvent, OS> + 'a,
    H: StreamCallback<OutputEvent> + 'a,
    IP: SignalProvider<Item = IS> + ?Sized + 'a,
    OP: SignalProvider<Item = OS> + ?Sized + 'a,
    S: EventStream<'a, InputEvent>,
> {
    origin: <S as EventStream<'a, InputEvent>>::Subscription<
        StateMachineCallback<'a, InputEvent, IS, OutputEvent, OS, M, H, IP, OP>,
    >,
}

impl<
    'a,
    InputEvent: 'a,
    IS: 'a,
    OutputEvent: 'a,
    OS: Clone + 'a,
    M: StateMachine<InputEvent, IS, OutputEvent, OS> + 'a,
    H: StreamCallback<OutputEvent> + 'a,
    IP: SignalProvider<Item = IS> + ?Sized + 'a,
    OP: SignalProvider<Item = OS> + ?Sized + 'a,
    S: EventStream<'a, InputEvent>,
> Unsubscribe<OutputEvent, StateMachineStream<'a, InputEvent, IS, OutputEvent, OS, M, IP, OP, S>, H>
    for StateMachineSubscription<'a, InputEvent, IS, OutputEvent, OS, M, H, IP, OP, S>
{
    fn unsubscribe(
        self,
    ) -> (
        StateMachineStream<'a, InputEvent, IS, OutputEvent, OS, M, IP, OP, S>,
        H,
    )
    where
        StateMachineStream<'a, InputEvent, IS, OutputEvent, OS, M, IP, OP, S>: Sized,
    {
        let (s, h) = self.origin.unsubscribe();
        (
            StateMachineStream {
                origin: s,
                m: h.m,
                input: h.input,
                output: h.output,
                phantom: PhantomData,
            },
            h.h,
        )
    }
}

struct StateMachineStream<
    'l,
    InputEvent: 'l,
    IS: 'l,
    OutputEvent: 'l,
    OS: Clone + 'l,
    M: StateMachine<InputEvent, IS, OutputEvent, OS> + 'l,
    IP: SignalProvider<Item = IS> + ?Sized,
    OP: SignalProvider<Item = OS> + ?Sized,
    S: EventStream<'l, InputEvent>,
> {
    origin: S,
    m: Rc<RefCell<M>>,
    input: Signal<IP>,
    output: Signal<OP>,
    phantom: PhantomData<&'l (InputEvent, OutputEvent, OS)>,
}

impl<
    'l,
    InputEvent: 'l,
    IS: 'l,
    OutputEvent: 'l,
    OS: Clone + 'l,
    M: StateMachine<InputEvent, IS, OutputEvent, OS> + 'l,
    IP: SignalProvider<Item = IS> + ?Sized + 'l,
    OP: SignalProvider<Item = OS> + ?Sized + 'l,
    S: EventStream<'l, InputEvent>,
> EventStream<'l, OutputEvent>
    for StateMachineStream<'l, InputEvent, IS, OutputEvent, OS, M, IP, OP, S>
{
    type Subscription<H: 'l + StreamCallback<OutputEvent> + 'l> =
        StateMachineSubscription<'l, InputEvent, IS, OutputEvent, OS, M, H, IP, OP, S>;

    fn subscribe<H: StreamCallback<OutputEvent> + 'l>(self, h: H) -> Self::Subscription<H> {
        Self::Subscription {
            origin: self.origin.subscribe(StateMachineCallback {
                m: self.m,
                h: h,
                input: self.input,
                output: self.output,
                phantom: PhantomData,
            }),
        }
    }
}

enum BounceState<'a, T: 'a, S: EventStream<'a, T>> {
    None(PhantomData<&'a T>),
    Initialized(S),
    Pending(Box<dyn FnOnce(S) -> *mut () + 'a>),
    Subscribed(*mut ()),
}

impl<'a, T: 'a, S: EventStream<'a, T>> std::fmt::Debug for BounceState<'a, T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None(arg0) => f.debug_tuple("None").field(arg0).finish(),
            Self::Initialized(_) => f.debug_tuple("Initialized").finish_non_exhaustive(),
            Self::Pending(_) => f.debug_tuple("Pending").finish_non_exhaustive(),
            Self::Subscribed(arg0) => f.debug_tuple("Subscribed").field(arg0).finish(),
        }
    }
}

pub struct BounceSubscription<'a, T: 'a, H: 'a + StreamCallback<T>, S: EventStream<'a, T>> {
    origin: Rc<RefCell<BounceState<'a, T, S>>>,
    phantom: PhantomData<H>,
}

impl<'a, T: 'a, H: 'a + StreamCallback<T>, S: EventStream<'a, T>> BounceSubscription<'a, T, H, S> {
    fn extract(&mut self) -> Option<H> {
        match *self.origin.borrow_mut() {
            BounceState::None(_) | BounceState::Initialized(_) | BounceState::Pending(_) => None,
            ref mut state @ BounceState::Subscribed(origin) => {
                let boxed = unsafe {
                    Box::from_raw(origin as *mut <S as EventStream<'a, T>>::Subscription<H>)
                };

                let (s, h) = boxed.unsubscribe();
                *state = BounceState::Initialized(s);
                Some(h)
            }
        }
    }
}

// This custom drop ensures we never leave an invalid dangling pointer in the shared state.
impl<'a, T: 'a, H: 'a + StreamCallback<T>, S: EventStream<'a, T>> Drop
    for BounceSubscription<'a, T, H, S>
{
    fn drop(&mut self) {
        let _ = self.extract();
    }
}

impl<'a, T, H: StreamCallback<T>, S: EventStream<'a, T>> Unsubscribe<T, BounceStream<'a, T, S>, H>
    for BounceSubscription<'a, T, H, S>
{
    fn unsubscribe(mut self) -> (BounceStream<'a, T, S>, H) {
        let h = self
            .extract()
            .expect("Tried to unsubscribe from invalid state!");
        (
            BounceStream {
                origin: self.origin.clone(),
            },
            h,
        )
    }
}

pub struct BounceStream<'l, T, S: EventStream<'l, T>> {
    origin: Rc<RefCell<BounceState<'l, T, S>>>,
}

impl<'l, T, S: EventStream<'l, T>> BounceStream<'l, T, S> {
    pub fn new() -> (Self, AntiStream<'l, T, S>) {
        let origin = Rc::new(RefCell::new(BounceState::None(PhantomData)));
        (
            Self {
                origin: origin.clone(),
            },
            AntiStream { origin },
        )
    }
}

#[derive(Clone, Debug)]
pub struct AntiStream<'l, T, S: EventStream<'l, T>> {
    origin: Rc<RefCell<BounceState<'l, T, S>>>,
}

impl<'l, T, S: EventStream<'l, T>> AntiStream<'l, T, S> {
    /// Consumes the AntiStream, which sends the eventstream to the corresponding BounceStream.
    pub fn connect(&self, target: S) -> eyre::Result<()> {
        let origin = &self.origin;
        let mut state = BounceState::None(PhantomData);
        std::mem::swap(&mut *origin.borrow_mut(), &mut state);
        state = match state {
            BounceState::Initialized(_) | BounceState::Subscribed(_) => {
                return Err(eyre::eyre!("This stream was already connected!"));
            }
            BounceState::None(_) => BounceState::Initialized(target),
            BounceState::Pending(f) => {
                let p = f(target);
                BounceState::Subscribed(p)
            }
        };
        std::mem::swap(&mut *origin.borrow_mut(), &mut state);
        Ok(())
    }
}

trait BouncePending<'l, T: 'l, S: EventStream<'l, T>> {
    fn subscribe(self, s: S) -> *mut ();
}

struct BouncePendingImpl<'l, H: 'l>(H, PhantomData<&'l ()>);

impl<'l, T: 'l, H: 'l + StreamCallback<T>, S: EventStream<'l, T>> BouncePending<'l, T, S>
    for BouncePendingImpl<'l, H>
{
    fn subscribe(self, s: S) -> *mut () {
        Box::into_raw(Box::new(s.subscribe(self.0))) as *mut ()
    }
}

impl<'l, T: 'l, S: EventStream<'l, T>> EventStream<'l, T> for BounceStream<'l, T, S> {
    type Subscription<H: StreamCallback<T> + 'l> = BounceSubscription<'l, T, H, S>;

    fn subscribe<H: StreamCallback<T> + 'l>(self, h: H) -> Self::Subscription<H> {
        let mut state = BounceState::None(PhantomData);
        std::mem::swap(&mut *self.origin.borrow_mut(), &mut state);
        state = match state {
            // Pending is safe to recover from, because the lambda knows how to drop itself
            BounceState::None(_) | BounceState::Pending(_) => {
                BounceState::Pending(Box::new(move |s: S| {
                    Box::into_raw(Box::new(s.subscribe(h))) as *mut ()
                }))
            }
            BounceState::Initialized(s) => {
                BounceState::Subscribed(Box::into_raw(Box::new(s.subscribe(h))) as *mut ())
            }
            // Subscribe is NOT safe to recover from, because we don't know how to drop the internal pointer
            BounceState::Subscribed(_) => panic!("BounceStream was already subscribed to!"),
        };

        std::mem::swap(&mut *self.origin.borrow_mut(), &mut state);

        Self::Subscription {
            origin: self.origin,
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
fn split_parity<'l>(
    num: impl EventStream<'l, i32> + 'l,
) -> (impl EventStream<'l, i32>, impl EventStream<'l, i32>) {
    let nums = num.dup();
    (
        nums.clone().filter(|n| n % 2 == 0),
        nums.clone().filter(|n| n % 2 == 1),
    )
}
