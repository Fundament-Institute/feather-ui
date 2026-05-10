// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::Limited;
use crate::smallset::SmallSet;
use backtrace::BytesOrWideString;
use imbl::Vector;
use stable_deref_trait::CloneStableDeref;
use std::cell::{OnceCell, Ref};
use std::cmp::{PartialEq, PartialOrd};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Deref;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::rc::Rc;
use tuple_list::{Tuple, TupleList};

use std::cell::RefCell;

#[cfg(feature = "signal-debug")]
macro_rules! rewrite_panic_debug {
    ($old:expr) => {
        Some($old as *const dyn std::fmt::Debug)
    };
}

#[cfg(not(feature = "signal-debug"))]
macro_rules! rewrite_panic_debug {
    ($old:expr) => {
        None
    };
}

#[cfg(any(debug_assertions, feature = "signal-backtrace"))]
macro_rules! rewrite_panic {
    ($frame:expr, $e:expr) => {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $e)) {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(Box::new(UnwindPayload {
                inner: e,
                frame: $frame.clone(),
                value: None,
            })),
        }
    };
    ($frame:expr, $e:expr, $old:expr) => {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $e)) {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(Box::new(UnwindPayload {
                inner: e,
                frame: $frame.clone(),
                value: rewrite_panic_debug!($old),
            })),
        }
    };
}

#[cfg(all(not(debug_assertions), not(feature = "signal-backtrace")))]
macro_rules! rewrite_panic {
    ($frame:expr, $e:expr) => {
        $e
    };
    ($frame:expr, $e:expr, $old:expr) => {
        $e
    };
}

#[cfg(feature = "signal-debug")]
pub trait SignalDebug: std::fmt::Debug {}

#[cfg(not(feature = "signal-debug"))]
pub trait SignalDebug {}

#[cfg(feature = "signal-debug")]
impl<T: std::fmt::Debug> SignalDebug for T {}

#[cfg(not(feature = "signal-debug"))]
impl<T> SignalDebug for T {}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum NodeColor {
    Ready,
    Changed,
    Check,
}

/// This is similar to OnceCell, but tracks active usages and will drop the inner value when usages drop to 0
pub struct MultiCell<T> {
    borrow: std::cell::Cell<i32>,
    inner: std::cell::UnsafeCell<Option<T>>,
}

impl<T> std::default::Default for MultiCell<T> {
    fn default() -> Self {
        Self {
            borrow: std::cell::Cell::new(0),
            inner: Default::default(),
        }
    }
}

impl<T> MultiCell<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

pub struct MultiRef<'b, T> {
    value: std::ptr::NonNull<Option<T>>,
    borrow: &'b std::cell::Cell<i32>,
}

impl<T> Deref for MultiRef<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: the value must be non-null as long as we hold our borrow.
        unsafe { self.value.as_ref().as_ref().unwrap_unchecked() }
    }
}

impl<T> Drop for MultiRef<'_, T> {
    fn drop(&mut self) {
        let borrow = self.borrow.get() - 1;
        self.borrow.set(borrow);

        if borrow == 0 {
            unsafe {
                self.value.as_mut().take();
            }
        }
    }
}

impl<T> MultiCell<T> {
    fn get_or_init(&self, t: T) -> MultiRef<'_, T> {
        let opt = self.inner.get();
        let borrow = self.borrow.get();
        if borrow == 0 {
            debug_assert!(unsafe { &*opt }.is_none());
            unsafe { *opt = Some(t) };
        }

        self.borrow.set(borrow + 1);
        MultiRef {
            value: unsafe { std::ptr::NonNull::new_unchecked(self.inner.get()) },
            borrow: &self.borrow,
        }
    }
}

/// This UnsafeRef strips the lifetime information from the type and allows casting back
/// into a DynRef<> type with *any* arbitrary lifetime. Obviously, this is EXTREMELY unsafe and is
/// only constructed internally for managing signal references.
#[repr(transparent)]
pub struct UnsafeRef<T>(DynRef<'static, ()>, PhantomData<T>);

impl<T> Deref for UnsafeRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { std::mem::transmute::<_, &DynRef<'_, T>>(&self.0) }
    }
}

/*mod internal {
    use super::{Deref, DynRef, PhantomData, UnsafeRef};

    impl<'a, T> From<DynRef<'a, T>> for UnsafeRef<T> {
        fn from(value: DynRef<'a, T>) -> Self {
            UnsafeRef(unsafe { std::mem::transmute(value) }, PhantomData)
        }
    }
}*/

fn into_unsafe_ref<'a, T>(value: DynRef<'a, T>) -> UnsafeRef<T> {
    UnsafeRef(unsafe { std::mem::transmute(value) }, PhantomData)
}

pub enum DynRef<'a, T> {
    Ref(&'a T),
    Cell(Ref<'a, T>),
    Multi(MultiRef<'a, T>),
    Flatten(MultiRef<'a, UnsafeRef<T>>),
}

impl<'a, T> Deref for DynRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            DynRef::Ref(v) => v,
            DynRef::Cell(v) => v,
            DynRef::Multi(v) => v,
            DynRef::Flatten(v) => v,
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Identity<T: CloneStableDeref>(pub T);
impl<T: CloneStableDeref> PartialEq for Identity<T> {
    fn eq(&self, other: &Self) -> bool {
        let self_ptr: *const _ = self.0.deref();
        let other_ptr: *const _ = other.0.deref();
        std::ptr::addr_eq(self_ptr, other_ptr)
    }
}
impl<T: CloneStableDeref> Eq for Identity<T> {}
impl<T: CloneStableDeref> PartialOrd for Identity<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let self_ptr: *const _ = self.0.deref();
        let other_ptr: *const _ = other.0.deref();
        self_ptr.cast::<()>().partial_cmp(&other_ptr.cast::<()>())
    }
}
impl<T: CloneStableDeref> Hash for Identity<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let self_ptr: *const _ = self.0.deref();
        self_ptr.hash(state);
    }
}

pub(crate) type SignalNodeId = Identity<Rc<RefCell<SignalNode>>>;

// Let's just be single threaded for now.
pub struct SignalNode {
    children: SmallSet<4, SignalNodeId>,
    color: NodeColor,
    callback: Option<Box<dyn Fn()>>,
    #[cfg(debug_assertions)]
    debug: &'static str,
}

impl fmt::Debug for SignalNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SignalNode")
            .field("children", &self.children)
            .field("color", &self.color)
            .finish()
    }
}

fn notify_check_node(nodeid: &SignalNodeId) {
    let mut node = nodeid.0.borrow_mut();
    if node.color == NodeColor::Ready {
        node.color = NodeColor::Check;
        node.children.iter().for_each(notify_check_node);

        if let Some(f) = &node.callback {
            f();
        }
    }
}

pub fn notify_check(signal: &Signal<impl SignalProvider + ?Sized>) {
    notify_check_node(signal.0.get_node())
}

fn notify_change_node(nodeid: &SignalNodeId) {
    let mut node = nodeid.0.borrow_mut();
    let was_ready = node.color == NodeColor::Ready;
    node.color = NodeColor::Changed;
    node.children.iter().for_each(notify_check_node);

    if was_ready
        && let Some(f) = &node.callback {
            f();
        }
}

pub fn notify_change(signal: &Signal<impl SignalProvider + ?Sized>) {
    notify_change_node(signal.0.get_node())
}

fn notify_children_change(nodeid: &SignalNodeId) {
    let node = nodeid.0.borrow();
    for child in node.children.iter() {
        assert!(!Rc::ptr_eq(&nodeid.0, &child.0));
        notify_change_node(child);
    }
    node.children.iter().for_each(notify_change_node);
}

fn add_dependency(parent: &SignalNodeId, child: SignalNodeId) {
    if parent.0.borrow().color != NodeColor::Ready && child.0.borrow().color == NodeColor::Ready {
        notify_check_node(&child);
    }
    parent.0.borrow_mut().children.insert(child);
}

fn remove_dependency(parent: &SignalNodeId, child: &SignalNodeId) {
    parent.0.borrow_mut().children.remove(child);
}

fn new_node<T>(color: NodeColor) -> SignalNodeId {
    Identity(Rc::new(RefCell::new(SignalNode {
        children: SmallSet::new(),
        color,
        callback: None,
        #[cfg(debug_assertions)]
        debug: std::any::type_name::<T>(),
    })))
}

// Signals may be in one of three states
// Const: Will never change again and do not need to be checked
// Discrete: Changes at specific intervals that are adequately captured by marking the graph node once something is known to change
// Continuous: Must be recomputed at every sampling, and can't just be notified to change each time, for example continuous time measurements

// TODO: Can I handle that by just refusing to update the graph node to mark it as updated after computing?
// Most things don't need to be continuous probably if the update minimization logic works correctly.

// Originally I wanted to use type level computation to enforce constant folding on statically known const signals, but is just having it check at runtime and depending on the optimizer good enough?
// At least for the prototype, I'm punting on that sophistication to simplify the code

// Being able to detect what the highest level of update frequency can be useful for determining what methods are applicable
// For example, making an event stream from changes on a Discrete Signal is possible, but not a Continuous one.
// On the other hand, Signals should usually be sampled by an event stream rather than notified from, and if we don't need continuous signals then that constraint vanishes.
// Triggering an event when a signal changes is useful for updating some external system in response to reactive computation, but if it can be ergonomically and performantly expressed in terms of sampling, we can drop the concept from the API

#[allow(dead_code)]
enum SignalNodeState {
    Const,
    Discrete(SignalNodeId),
}

pub trait SignalProvider {
    type Item;

    fn get_node(&self) -> &SignalNodeId;
    fn get_ref(&self) -> DynRef<'_, Self::Item>;
    fn update(&self);
}

//This wrapper type is required to make the trait resolution allow any constant to be used as a signal with AsSignal, which in addition to being a nice convenience feature is necessary for the idiom macro to work
// TODO: try to reduce amount of Rc required
pub struct Signal<Provider: SignalProvider + ?Sized>(Rc<Provider>);
impl<Provider: SignalProvider + ?Sized> Clone for Signal<Provider> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<Provider: SignalProvider + ?Sized> PartialEq for Signal<Provider> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl<Provider: SignalProvider + ?Sized> Eq for Signal<Provider> {}

impl<Provider: SignalProvider + ?Sized> std::hash::Hash for Signal<Provider> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl<Provider: SignalProvider + ?Sized> fmt::Debug for Signal<Provider>
where
    Provider::Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match std::panic::catch_unwind(AssertUnwindSafe(|| sample(self))) {
            Ok(v) if !std::thread::panicking() => {
                f.write_str("\u{00ab}")?;
                v.fmt(f)?;
                f.write_str("\u{00bb}")
            }
            _ => f
                .debug_struct("Signal")
                .field("node", self.0.get_node())
                .field("value", &*self.0.get_ref())
                .finish(),
        }
    }
}

use crate::Resolve;

impl<P1: SignalProvider + ?Sized> Signal<P1> {
    pub fn resolve<P2: SignalProvider + ?Sized>(
        self,
        factor: Signal<P2>,
    ) -> Signal<impl SignalProvider<Item = <P1::Item as Resolve<P2::Item>>::Output>>
    where
        P1::Item: Resolve<P2::Item>,
        P2::Item: Clone,
        <P1::Item as Resolve<P2::Item>>::Output: PartialEq,
    {
        map(
            move |arg| arg.0.resolve(arg.1.clone()),
            zip((self, factor)),
            PartialEq::eq,
            never_debug,
        )
    }

    pub fn limit(
        self,
        limits: Signal<impl SignalProvider<Item = crate::PxLimits> + ?Sized>,
    ) -> Signal<impl SignalProvider<Item = P1::Item>>
    where
        P1::Item: Limited + Clone + PartialEq,
    {
        map(
            move |arg| arg.0.clone().limit(*arg.1),
            zip((self, limits)),
            PartialEq::eq,
            never_debug,
        )
    }
}

// This has the same logic as From, but avoids needing manual type annotations in some situations.
impl<Provider: SignalProvider + 'static> Signal<Provider> {
    pub fn into_dyn(self) -> Signal<dyn SignalProvider<Item = Provider::Item>> {
        Signal(self.0)
    }
}

impl<Provider: SignalProvider + 'static> From<Signal<Provider>>
    for Signal<dyn SignalProvider<Item = Provider::Item>>
{
    fn from(value: Signal<Provider>) -> Self {
        Self(value.0)
    }
}

// This is absolutely optimizable to ensure const signals do not need to allocate a node
// A sketch of some of the components needed to do so are included
// But I leave actually implementing the type level optimizations to after prototyping
#[derive_where::derive_where(Debug; T: fmt::Debug)]
pub struct ConstProvider<T>(T, SignalNodeId);
impl<T> ConstProvider<T> {
    pub fn new(x: T) -> Self {
        Self(x, new_node::<Self>(NodeColor::Ready))
    }
}

pub fn const_new<T>(x: T) -> ConstSignal<T> {
    ConstSignal::<T>::new(x)
}

pub fn const_default<T: Default>() -> ConstSignal<T> {
    ConstSignal::<T>::default()
}

impl<T> SignalProvider for ConstProvider<T> {
    type Item = T;

    fn update(&self) {}

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.1
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, Self::Item> {
        DynRef::Ref(&self.0)
    }
}

pub trait ToSignal<T> {
    type Provider: SignalProvider<Item = T> + ?Sized;
    fn to_signal(self) -> Signal<Self::Provider>;
}

impl<T: num_traits::Num> From<T> for ConstSignal<T> {
    fn from(value: T) -> Self {
        Signal(Rc::new(ConstProvider::new(value)))
    }
}
impl<T: num_traits::Num> ToSignal<T> for T {
    type Provider = ConstProvider<T>;
    fn to_signal(self) -> Signal<Self::Provider> {
        Signal(Rc::new(ConstProvider::new(self)))
    }
}

// TODO: The Rc here should be removed by changing the API interface such that SignalProvider
// is implemented on Rc<TheProvider>, thus allowing ConstProvider to not store an Rc at all,
// but this will also require changing how updates are handled so that ConstProvider doesn't
// need to store an update node.
impl<T: Default> Default for Signal<ConstProvider<T>> {
    fn default() -> Self {
        Self(Rc::new(ConstProvider::new(T::default())))
    }
}

impl<T> Signal<ConstProvider<T>> {
    pub fn new(t: T) -> Self {
        Self(Rc::new(ConstProvider::new(t)))
    }
}

//TODO: constant fusion optimization to eliminate unnecessary intermediate storage

pub trait ProviderTupleList
where
    Self: Clone,
{
    type RefResult: TupleList;

    fn update(&self);
    fn build_ref(&self) -> Self::RefResult;
    fn add_dependency(&self, node: SignalNodeId);
}

impl ProviderTupleList for () {
    type RefResult = ();

    fn update(&self) {}
    fn build_ref(&self) -> Self::RefResult {
        
    }
    fn add_dependency(&self, _node: SignalNodeId) {}
}

impl<Head, Tail> ProviderTupleList for (Rc<Head>, Tail)
where
    Head: SignalProvider + ?Sized,
    Tail: ProviderTupleList,
    (UnsafeRef<Head::Item>, Tail::RefResult): TupleList,
{
    type RefResult = (UnsafeRef<Head::Item>, Tail::RefResult);

    fn update(&self) {
        self.0.update();
        self.1.update();
    }
    fn build_ref(&self) -> Self::RefResult {
        (into_unsafe_ref(self.0.get_ref()), self.1.build_ref())
    }
    fn add_dependency(&self, node: SignalNodeId) {
        add_dependency(self.0.get_node(), node.clone());
        self.1.add_dependency(node);
    }
}

pub struct ZipProvider<PList: ProviderTupleList> {
    providers: PList,
    res: MultiCell<<PList::RefResult as TupleList>::Tuple>,
    node: SignalNodeId,
}

impl<PList: ProviderTupleList> fmt::Debug for ZipProvider<PList> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZipSignal")
            .field("node", &self.node)
            .finish()
    }
}

impl<PList: ProviderTupleList> SignalProvider for ZipProvider<PList> {
    type Item = <PList::RefResult as TupleList>::Tuple;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        let color = self.node.0.borrow().color;
        if color != NodeColor::Ready {
            self.providers.update();
        }
        let color = self.node.0.borrow().color;
        if color == NodeColor::Changed {
            notify_children_change(&self.node);
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, Self::Item> {
        // We can get away with this precisely because, while the borrow is held, the underlying memory location cannot change. Therefore, we can re-use
        // the same memory location inside our struct for multiple borrows as long as a non-zero number of borrows exist.
        DynRef::Multi(
            self.res
                .get_or_init(self.providers.build_ref().into_tuple()),
        )
    }
}

pub trait SignalTupleList {
    type Result: TupleList + ProviderTupleList;

    fn as_providers(&self) -> Self::Result;
}

impl SignalTupleList for () {
    type Result = ();

    fn as_providers(&self) -> Self::Result {
        
    }
}

impl<Head, Tail> SignalTupleList for (Signal<Head>, Tail)
where
    Head: SignalProvider + ?Sized,
    Tail: SignalTupleList,
    (Rc<Head>, Tail::Result): TupleList + ProviderTupleList,
{
    type Result = (Rc<Head>, Tail::Result);

    fn as_providers(&self) -> Self::Result {
        (self.0.0.clone(), self.1.as_providers())
    }
}

pub fn zip<T: Tuple>(t: T) -> Signal<ZipProvider<<T::TupleList as SignalTupleList>::Result>>
where
    T::TupleList: SignalTupleList,
{
    let node =
        new_node::<ZipProvider<<T::TupleList as SignalTupleList>::Result>>(NodeColor::Changed);
    let providers = t.into_tuple_list().as_providers();
    providers.add_dependency(node.clone());
    verify_tree(&node);
    Signal(Rc::new(ZipProvider {
        providers,
        res: Default::default(),
        node,
    }))
}

pub trait UnsafeRefCloneTupleList {
    type Result: TupleList;

    fn get_val(&self) -> Self::Result;
}

impl UnsafeRefCloneTupleList for () {
    type Result = ();

    fn get_val(&self) -> Self::Result {
        
    }
}

impl<Head, Tail> UnsafeRefCloneTupleList for (UnsafeRef<Head>, Tail)
where
    Head: Clone,
    Tail: UnsafeRefCloneTupleList,
    (Head, Tail::Result): TupleList,
{
    type Result = (Head, Tail::Result);

    fn get_val(&self) -> Self::Result {
        (self.0.clone(), self.1.get_val())
    }
}

pub trait TupleBorrow<'a>: Tuple {
    type Result: Tuple;

    fn borrow(&'a self) -> Self::Result;
}

impl<'a, T1: 'a> TupleBorrow<'a> for (UnsafeRef<T1>,) {
    type Result = (&'a T1,);

    fn borrow(&'a self) -> Self::Result {
        (self.0.deref(),)
    }
}
/*
//It's not possible to use UnsafeRefTupleList in a generic way because of limitations in the rust borrow checker.

pub trait UnsafeRefTupleList<'a> {
    type Result: TupleList;

    fn borrow(&'a self) -> Self::Result;
}

impl UnsafeRefTupleList<'_> for () {
    type Result = ();

    fn borrow(&self) -> Self::Result {
        ()
    }
}

impl<'a, Head, Tail> UnsafeRefTupleList<'a> for (UnsafeRef<Head>, Tail)
where
    Head: 'a,
    Tail: UnsafeRefTupleList<'a>,
    (&'a Head, Tail::Result): TupleList,
{
    type Result = (&'a Head, Tail::Result);

    fn borrow(&'a self) -> Self::Result {
        (&self.0, self.1.borrow())
    }
}*/

pub struct ZipValueProvider<PList: ProviderTupleList>
where
    PList::RefResult: UnsafeRefCloneTupleList,
{
    provider: Rc<ZipProvider<PList>>,
    res: RefCell<
        Option<<<PList::RefResult as UnsafeRefCloneTupleList>::Result as TupleList>::Tuple>,
    >,
    node: SignalNodeId,
}

impl<PList: ProviderTupleList> fmt::Debug for ZipValueProvider<PList>
where
    PList::RefResult: UnsafeRefCloneTupleList,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZipSignal")
            .field("node", &self.node)
            .finish()
    }
}

impl<PList: ProviderTupleList> SignalProvider for ZipValueProvider<PList>
where
    PList::RefResult: UnsafeRefCloneTupleList,
{
    type Item = <<PList::RefResult as UnsafeRefCloneTupleList>::Result as TupleList>::Tuple;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        let color = self.node.0.borrow().color;
        if color != NodeColor::Ready {
            self.provider.update();
        }
        let color = self.node.0.borrow().color;
        if color == NodeColor::Changed {
            *self.res.borrow_mut() =
                Some(self.provider.providers.build_ref().get_val().into_tuple());
            notify_children_change(&self.node);
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, Self::Item> {
        DynRef::Cell(Ref::map(self.res.borrow(), |x| {
            x.as_ref().expect("Must be updated before getting value.")
        }))
    }
}

/*
pub struct ZipMapProvider<
    PList: ProviderTupleList,
    R,
    F: Fn(<<PList::RefResult as UnsafeRefTupleList>::Result as TupleList>::Tuple) -> R,
> where
    for<'a> <PList as ProviderTupleList>::RefResult: UnsafeRefTupleList<'a>,
{
    providers: PList,
    func: F,
    res: RefCell<Option<R>>,
    node: SignalNodeId,
}

impl<
    PList: ProviderTupleList,
    R,
    F: Fn(<<PList::RefResult as UnsafeRefTupleList>::Result as TupleList>::Tuple) -> R,
> std::fmt::Debug for ZipMapProvider<PList, R, F>
where
    for<'a> <PList as ProviderTupleList>::RefResult: UnsafeRefTupleList<'a>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZipMapProvider")
            .field("node", &self.node)
            .finish()
    }
}

impl<
    PList: ProviderTupleList,
    R,
    F: Fn(<<PList::RefResult as UnsafeRefTupleList>::Result as TupleList>::Tuple) -> R,
> SignalProvider for ZipMapProvider<PList, R, F>
where
    for<'a> <PList as ProviderTupleList>::RefResult: UnsafeRefTupleList<'a>,
{
    type Item = R;

#[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        let color = self.node.0.borrow().color;
        if color != NodeColor::Ready {
            self.providers.update();
        }
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Changed => {
                *self.res.borrow_mut() = Some((self.func)(
                    self.providers.build_ref().borrow().into_tuple(),
                ));
                notify_children_change(&self.node);
            }
            _ => {}
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

#[inline]
    fn get_ref(&self) -> DynRef<'_, Self::Item> {
        DynRef::Cell(Ref::map(self.res.borrow(), |x| {
            x.as_ref().expect("Must be updated before getting value.")
        }))
    }
}*/

impl<PList: ProviderTupleList> Signal<ZipProvider<PList>> {
    pub fn value(&self) -> Signal<ZipValueProvider<PList>>
    where
        <PList as ProviderTupleList>::RefResult: UnsafeRefCloneTupleList,
    {
        let node = new_node::<ZipValueProvider<PList>>(NodeColor::Changed);
        add_dependency(self.0.get_node(), node.clone());
        Signal(Rc::new(ZipValueProvider::<PList> {
            provider: self.0.clone(),
            node,
            res: Default::default(),
        }))
    }

    // It's impossible to write this function because current borrow checker limitations make it impossible to write a function signature
    // generic over all lifetimes that gets stored somewhere.
    /*pub fn flatmap<
        R,
        F: for<'a> Fn(<<<PList as ProviderTupleList>::RefResult as UnsafeRefTupleList<'a>>::Result as TupleList>::Tuple) -> R,
    >(
        &self,
        f: F,
    ) -> Signal<impl SignalProvider<Item = R>>
    where
        Self: Sized,
        <PList as ProviderTupleList>::RefResult: for<'a> UnsafeRefTupleList<'a>,
    {
        let node = new_node::<Self>(NodeColor::Changed);
        add_dependency(self.0.get_node(), node.clone());
        Signal(Rc::new(ZipMapProvider::<PList, R, F> {
            providers: self.0.providers.clone(),
            func: f,
            node,
            res: Default::default(),
        }))
    }*/
}

macro_rules! gen_flatmap {
    ($($x:ident),*) => (
        impl<$($x: SignalProvider + ?Sized),*> Signal<ZipProvider<tuple_list::tuple_list_type!($(Rc<$x>),*)>> {
            #[allow(non_snake_case)]
            pub fn flatmap<R: SignalDebug, F: Fn(($(&$x::Item),*)) -> R>(&self, f: F) -> Signal<impl SignalProvider<Item = R> + use<R, F, $($x),*>> {
                map(
                    move |($($x),*)| f(($(&*$x),*)),
                    self.clone(),
                    never_eq,
                    never_debug,
                )
            }
            #[allow(non_snake_case)]
            pub fn flatmap_mut<R: SignalDebug, F: Fn(($(&$x::Item),*), Option<R>) -> R>(
                &self,
                f: F,
            ) -> Signal<impl SignalProvider<Item = R> + use<R, F, $($x),*>> {
                map_mut(move |($($x),*), o| f(($(&*$x),*), o), self.clone())
            }
        }
    )
}

gen_flatmap!(P1, P2, P3);
gen_flatmap!(P1, P2, P3, P4);
gen_flatmap!(P1, P2, P3, P4, P5);
gen_flatmap!(P1, P2, P3, P4, P5, P6);
gen_flatmap!(P1, P2, P3, P4, P5, P6, P7);
gen_flatmap!(P1, P2, P3, P4, P5, P6, P7, P8);
gen_flatmap!(P1, P2, P3, P4, P5, P6, P7, P8, P9);
gen_flatmap!(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10);
gen_flatmap!(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11);
gen_flatmap!(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12);

impl<P1: SignalProvider + ?Sized, P2: SignalProvider + ?Sized>
    Signal<ZipProvider<(Rc<P1>, (Rc<P2>, ()))>>
{
    pub fn flatmap<R: SignalDebug, F: Fn((&P1::Item, &P2::Item)) -> R>(
        &self,
        f: F,
    ) -> Signal<impl SignalProvider<Item = R> + use<R, F, P1, P2>> {
        map(
            move |(a1, a2)| f((a1, a2)),
            self.clone(),
            never_eq,
            never_debug,
        )
    }

    pub fn flatmap_mut<R: SignalDebug, F: Fn((&P1::Item, &P2::Item), Option<R>) -> R>(
        &self,
        f: F,
    ) -> Signal<impl SignalProvider<Item = R> + use<R, F, P1, P2>> {
        map_mut(move |(a1, a2), o| f((a1, a2), o), self.clone())
    }
}

pub trait SignalZip<Input: Tuple>
where
    <Input as tuple_list::Tuple>::TupleList: SignalTupleList,
{
    fn zip(self) -> Signal<ZipProvider<<<Input as Tuple>::TupleList as SignalTupleList>::Result>>;
}
impl<Input: Tuple> SignalZip<Input> for Input
where
    <Input as tuple_list::Tuple>::TupleList: SignalTupleList,
{
    fn zip(self) -> Signal<ZipProvider<<<Input as Tuple>::TupleList as SignalTupleList>::Result>> {
        zip(self)
    }
}

pub struct MapProvider<
    P: SignalProvider + ?Sized,
    T2,
    F: Fn(&P::Item) -> T2,
    F2: Fn(&T2, &T2) -> bool,
    F3: Fn(&T2) -> Option<&dyn fmt::Debug>,
> {
    provider: Rc<P>,
    func: F,
    res: RefCell<Option<T2>>,
    node: SignalNodeId,
    eq: F2,
    debug: F3,
    #[cfg(any(debug_assertions, feature = "signal-backtrace"))]
    debug_frame: backtrace::BacktraceFrame,
}

impl<
    P: SignalProvider + ?Sized,
    T2,
    F: Fn(&P::Item) -> T2,
    F2: Fn(&T2, &T2) -> bool,
    F3: Fn(&T2) -> Option<&dyn fmt::Debug>,
> fmt::Debug for MapProvider<P, T2, F, F2, F3>
where
    T2: fmt::Debug,
    P: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dbg = f.debug_struct("MapSignal");
        dbg.field("node", &self.node)
            .field("provider", &self.provider)
            .field("f", &format_args!("{}", std::any::type_name::<F>()))
            .field("res", &self.res);
        dbg.finish()
    }
}

impl<
    P: SignalProvider + ?Sized,
    T2: SignalDebug,
    F: Fn(&P::Item) -> T2,
    F2: Fn(&T2, &T2) -> bool,
    F3: Fn(&T2) -> Option<&dyn fmt::Debug>,
> SignalProvider for MapProvider<P, T2, F, F2, F3>
{
    type Item = T2;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                rewrite_panic!(self.debug_frame, self.provider.update(), self.res.as_ptr());
            }
        }
        let color = self.node.0.borrow().color;
        if color == NodeColor::Changed {
            let res = rewrite_panic!(
                self.debug_frame,
                (self.func)(&self.provider.get_ref()),
                self.res.as_ptr()
            );

            let changed = if let Some(old) = &*self.res.borrow() {
                !(self.eq)(old, &res)
            } else {
                true
            };
            *self.res.borrow_mut() = Some(res);
            if changed {
                notify_children_change(&self.node);
            }
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, T2> {
        DynRef::Cell(Ref::map(self.res.borrow(), |x| {
            x.as_ref().expect("Must be updated before getting value, or there was an attempt to use this value during a map operation.")
        }))
    }
}

fn map<
    T,
    R: SignalDebug,
    P: SignalProvider<Item = T> + ?Sized,
    F: Fn(&P::Item) -> R,
    FEq: Fn(&R, &R) -> bool,
    FDebug: Fn(&R) -> Option<&dyn fmt::Debug>,
>(
    f: F,
    signal: Signal<P>,
    eq: FEq,
    debug: FDebug,
) -> Signal<MapProvider<P, R, F, FEq, FDebug>> {
    let node = new_node::<MapProvider<P, R, F, FEq, FDebug>>(NodeColor::Changed);
    let provider = signal.0;
    add_dependency(provider.get_node(), node.clone());
    Signal(Rc::new(MapProvider {
        provider,
        func: f,
        res: RefCell::new(None),
        node,
        eq,
        debug,
        debug_frame: backtrace::Backtrace::new_unresolved().frames()[6].clone(),
    }))
}

pub struct MapMutProvider<P: SignalProvider + ?Sized, T2, F: Fn(&P::Item, Option<T2>) -> T2> {
    provider: Rc<P>,
    func: F,
    res: RefCell<Option<T2>>,
    node: SignalNodeId,
    #[cfg(any(debug_assertions, feature = "signal-backtrace"))]
    debug_frame: backtrace::BacktraceFrame,
}

impl<P: SignalProvider + ?Sized, T2, F: Fn(&P::Item, Option<T2>) -> T2> fmt::Debug
    for MapMutProvider<P, T2, F>
where
    T2: fmt::Debug,
    P: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dbg = f.debug_struct("MapMutSignal");
        dbg.field("node", &self.node)
            .field("provider", &self.provider)
            .field("f", &format_args!("{}", std::any::type_name::<F>()))
            .field("res", &self.res);
        dbg.finish()
    }
}

impl<P: SignalProvider + ?Sized, T2: SignalDebug, F: Fn(&P::Item, Option<T2>) -> T2> SignalProvider
    for MapMutProvider<P, T2, F>
{
    type Item = T2;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                rewrite_panic!(self.debug_frame, self.provider.update(), self.res.as_ptr());
            }
        }
        let color = self.node.0.borrow().color;
        if color == NodeColor::Changed {
            let init = self.res.borrow_mut().take();

            let res = rewrite_panic!(
                self.debug_frame,
                (self.func)(&self.provider.get_ref(), init),
                self.res.as_ptr()
            );

            *self.res.borrow_mut() = Some(res);
            notify_children_change(&self.node);
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, T2> {
        DynRef::Cell(Ref::map(self.res.borrow(), |x| {
            x.as_ref().expect("Must be updated before getting value, or there was an attempt to use this value during a map operation.")
        }))
    }
}

fn map_mut<
    T,
    R: SignalDebug,
    F: Fn(&P::Item, Option<R>) -> R,
    P: SignalProvider<Item = T> + ?Sized,
>(
    f: F,
    p: Signal<P>,
) -> Signal<MapMutProvider<P, R, F>> {
    let node = new_node::<MapMutProvider<P, R, F>>(NodeColor::Changed);
    let provider = p.0;
    add_dependency(provider.get_node(), node.clone());
    Signal(Rc::new(MapMutProvider {
        provider,
        func: f,
        res: RefCell::new(None),
        node,
        debug_frame: backtrace::Backtrace::new_unresolved().frames()[6].clone(),
    }))
}

fn never_eq<T>(_: &T, _: &T) -> bool {
    false
}

fn never_debug<T>(_: &T) -> Option<&dyn fmt::Debug> {
    None
}

fn always_debug<T: fmt::Debug>(t: &T) -> Option<&dyn fmt::Debug> {
    Some(t)
}

impl<Elem, P: SignalProvider<Item = Elem> + ?Sized> Signal<P> {
    pub fn map<T: PartialEq + SignalDebug, F: Fn(&P::Item) -> T>(
        self,
        f: F,
    ) -> Signal<impl SignalProvider<Item = T>> {
        map(f, self, PartialEq::eq, never_debug)
    }
    pub fn map_pred<T: SignalDebug>(
        self,
        f: impl Fn(&P::Item) -> T,
        eq: impl Fn(&T, &T) -> bool,
    ) -> Signal<impl SignalProvider<Item = T>> {
        map(f, self, eq, never_debug)
    }
    pub fn map_mut<T: SignalDebug>(
        self,
        f: impl Fn(&P::Item, Option<T>) -> T,
    ) -> Signal<impl SignalProvider<Item = T>> {
        map_mut(f, self)
    }
    pub fn map_ex<T: SignalDebug>(
        self,
        f: impl Fn(&P::Item) -> T,
    ) -> Signal<impl SignalProvider<Item = T>> {
        map(f, self, never_eq, never_debug)
    }
    pub fn map_debug<T: fmt::Debug>(
        self,
        f: impl Fn(&P::Item) -> T,
    ) -> Signal<impl SignalProvider<Item = T>> {
        map(f, self, never_eq, always_debug)
    }
}

pub struct JoinProvider<
    T,
    P1: SignalProvider<Item = T> + ?Sized,
    P2: SignalProvider<Item = Signal<P1>> + ?Sized,
> {
    provider: Rc<P2>,
    innerprovider: RefCell<Option<Rc<P1>>>,
    res: MultiCell<UnsafeRef<Rc<P1>>>,
    res2: MultiCell<UnsafeRef<T>>,
    node: SignalNodeId,
    phantom: PhantomData<P1>,
}

impl<T, P1: SignalProvider<Item = T> + ?Sized, P2: SignalProvider<Item = Signal<P1>> + ?Sized>
    fmt::Debug for JoinProvider<T, P1, P2>
where
    P1: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JoinSignal")
            .field("inner", &self.innerprovider)
            .field("node", &self.node)
            .finish()
    }
}

impl<T, P1: SignalProvider<Item = T> + ?Sized, P2: SignalProvider<Item = Signal<P1>> + ?Sized>
    SignalProvider for JoinProvider<T, P1, P2>
{
    type Item = T;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                self.provider.update();
                let color = self.node.0.borrow().color;
                if color == NodeColor::Changed {
                    let innersig = self.provider.get_ref().0.clone();
                    let old_innerprovider = self.innerprovider.borrow().clone();
                    match old_innerprovider {
                        Some(old_innerprovider_actual) => {
                            if old_innerprovider_actual.get_node() == innersig.get_node() {
                                remove_dependency(
                                    old_innerprovider_actual.get_node(),
                                    &self.node,
                                );
                                *self.innerprovider.borrow_mut() = Some(innersig.clone());
                                add_dependency(innersig.get_node(), self.node.clone());
                            }
                        }
                        None => {
                            *self.innerprovider.borrow_mut() = Some(innersig.clone());
                            add_dependency(innersig.get_node(), self.node.clone());
                        }
                    }
                    // TODO: Setting this to Check causes a problem with Res not being updated below. Does leaving
                    // the color as Changed cause a performance issue?
                    //self.node.0.borrow_mut().color = NodeColor::Check;
                }
                self.provider.get_ref().0.update();
                let color = self.node.0.borrow().color;
                if color == NodeColor::Changed {
                    notify_children_change(&self.node);
                }
                self.node.0.borrow_mut().color = NodeColor::Ready;
            }
        }
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, T> {
        let test = self.res.get_or_init(into_unsafe_ref(DynRef::Cell(Ref::map(
            self.innerprovider.borrow(),
            |x| x.as_ref().expect("Must be updated before getting value."),
        ))));

        DynRef::Flatten(self.res2.get_or_init(into_unsafe_ref(test.get_ref())))
    }
}

pub fn join<
    T,
    P1: SignalProvider<Item = T> + ?Sized,
    P2: SignalProvider<Item = Signal<P1>> + ?Sized,
>(
    p: Signal<P2>,
) -> Signal<JoinProvider<T, P1, P2>> {
    Signal(Rc::new(JoinProvider {
        provider: p.0,
        innerprovider: RefCell::new(None),
        res: Default::default(),
        res2: Default::default(),
        node: new_node::<JoinProvider<T, P1, P2>>(NodeColor::Changed),
        phantom: PhantomData,
    }))
}

pub trait SignalProviderMut: SignalProvider {
    fn refcell(&self) -> &RefCell<Self::Item>;
}

pub struct MutableProvider<T, Inputs: Tuple> {
    node: SignalNodeId,
    val: RefCell<T>,
    inputs: Inputs::TupleList,
}

pub trait MutableInputs<T>: TupleList {
    fn add_dependency(&self, node: SignalNodeId);
    fn update_check(&self, val: &mut T, node: SignalNodeId);
}

impl<T> MutableInputs<T> for () {
    fn add_dependency(&self, _: SignalNodeId) {}
    fn update_check(&self, _: &mut T, _: SignalNodeId) {}
}

impl<
    'a,
    T,
    Head: SignalProvider + ?Sized + 'a,
    F: Fn(&mut T, &<Head as SignalProvider>::Item),
    Tail,
> MutableInputs<T> for ((Signal<Head>, F), Tail)
where
    Tail: MutableInputs<T> + 'a,
    Self: TupleList,
{
    fn add_dependency(&self, node: SignalNodeId) {
        add_dependency(self.0.0.0.get_node(), node.clone());
        self.1.add_dependency(node);
    }
    fn update_check(&self, val: &mut T, node: SignalNodeId) {
        let ((signal, handler), tail) = self;
        let color = signal.0.get_node().0.borrow().color;
        if color != NodeColor::Ready {
            handler(val, &*sample(signal));
            notify_change_node(&node);
        }
        tail.update_check(val, node);
    }
}

pub trait DebugMutableInputs<T>: TupleList {
    fn fmt(&self, dbg: &mut fmt::DebugStruct<'_, '_>);
}

impl<T> DebugMutableInputs<T> for () {
    fn fmt(&self, _: &mut fmt::DebugStruct<'_, '_>) {}
}

impl<
    'a,
    T,
    Head: SignalProvider + ?Sized + 'a,
    F: Fn(&mut T, &<Head as SignalProvider>::Item),
    Tail,
> DebugMutableInputs<T> for ((Signal<Head>, F), Tail)
where
    Tail: DebugMutableInputs<T> + 'a,
    Self: TupleList,
    Signal<Head>: fmt::Debug,
{
    fn fmt(&self, dbg: &mut fmt::DebugStruct<'_, '_>) {
        let ((signal, _), tail) = self;
        dbg.field("input", &signal);
        tail.fmt(dbg)
    }
}

impl<T, Inputs: Tuple> fmt::Debug for MutableProvider<T, Inputs>
where
    Inputs::TupleList: DebugMutableInputs<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dbg = f.debug_struct("MutableSignal");
        dbg.field("node", &self.node);
        self.inputs.fmt(&mut dbg);
        dbg.finish()
    }
}

impl<T, Inputs: Tuple> MutableProvider<T, Inputs>
where
    Inputs::TupleList: MutableInputs<T>,
{
    pub fn new(v: T, inputs: Inputs) -> Self {
        let node = new_node::<Self>(NodeColor::Changed);
        let list = inputs.into_tuple_list();
        list.add_dependency(node.clone());
        Self {
            inputs: list,
            node,
            val: RefCell::new(v),
        }
    }
}

impl<T, Inputs: Tuple> SignalProviderMut for MutableProvider<T, Inputs>
where
    Inputs::TupleList: MutableInputs<T>,
{
    fn refcell(&self) -> &RefCell<T> {
        &self.val
    }
}

impl<T, Inputs: Tuple> SignalProvider for MutableProvider<T, Inputs>
where
    Inputs::TupleList: MutableInputs<T>,
{
    type Item = T;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        self.inputs
            .update_check(&mut self.val.borrow_mut(), self.node.clone());
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Changed => {
                notify_children_change(&self.node);
                self.node.0.borrow_mut().color = NodeColor::Ready;
            }
            NodeColor::Check => {
                self.node.0.borrow_mut().color = NodeColor::Ready;
            }
            NodeColor::Ready => {}
        }
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, Self::Item> {
        DynRef::Cell(self.val.borrow())
    }
}

pub struct SignalRefMut<'a, P: SignalProvider + ?Sized>(pub std::cell::RefMut<'a, P::Item>, Rc<P>);

impl<'a, P: SignalProvider + ?Sized> std::ops::Deref for SignalRefMut<'_, P> {
    type Target = P::Item;

    #[inline]
    fn deref(&self) -> &P::Item {
        &self.0
    }
}

impl<'a, P: SignalProvider + ?Sized> std::ops::DerefMut for SignalRefMut<'_, P> {
    #[inline]
    fn deref_mut(&mut self) -> &mut P::Item {
        &mut self.0
    }
}

impl<'a, P: SignalProvider + ?Sized> Drop for SignalRefMut<'a, P> {
    fn drop(&mut self) {
        notify_change_node(self.1.get_node());
    }
}

impl<T: Default> Default for Signal<MutableProvider<T, ()>> {
    fn default() -> Self {
        Self(Rc::new(MutableProvider::new(T::default(), ())))
    }
}

impl<T: Default + 'static> Default for Signal<dyn SignalProvider<Item = T>> {
    fn default() -> Self {
        Self(Rc::new(ConstProvider::new(T::default())))
    }
}

impl<MutProvider: SignalProviderMut + ?Sized> Signal<MutProvider> {
    // Instead of using `borrow` most things should use map() or sample()
    /*pub fn borrow(&self) -> Ref<'_, T> {
        self.0.update();
        self.0.val.borrow()
    }*/
    #[inline]
    pub fn borrow_mut(&self) -> SignalRefMut<'_, MutProvider> {
        self.0.update();
        SignalRefMut(self.0.refcell().borrow_mut(), self.0.clone())
    }
    /*pub fn try_borrow(&self) -> Result<Ref<'_, T>, std::cell::BorrowError> {
        self.0.update();
        self.0.val.try_borrow()
    }*/
    #[inline]
    pub fn try_borrow_mut(
        &self,
    ) -> Result<SignalRefMut<'_, MutProvider>, std::cell::BorrowMutError> {
        self.0.update();
        self.0
            .refcell()
            .try_borrow_mut()
            .map(|x| SignalRefMut(x, self.0.clone()))
    }
    #[inline]
    pub fn replace(&self, x: MutProvider::Item) -> MutProvider::Item {
        let old = self.0.refcell().replace(x);
        notify_change_node(self.0.get_node());
        old
    }
    #[inline]
    pub fn replace_with(
        &self,
        f: impl FnOnce(&mut MutProvider::Item) -> MutProvider::Item,
    ) -> MutProvider::Item {
        let old = self.0.refcell().replace_with(f);
        notify_change_node(self.0.get_node());
        old
    }
    #[inline]
    pub fn set_with(&self, f: impl FnOnce(&mut MutProvider::Item)) {
        f(&mut self.0.refcell().borrow_mut());
        notify_change_node(self.0.get_node());
    }
    #[inline]
    pub fn swap(&self, rhs: &Self) {
        self.0.refcell().swap(rhs.0.refcell());
    }
}

impl<T, Inputs: Tuple> Signal<MutableProvider<T, Inputs>>
where
    Inputs::TupleList: MutableInputs<T>,
{
    pub fn new_inputs(x: T, inputs: Inputs) -> Self {
        Self(Rc::new(MutableProvider::new(x, inputs)))
    }
}

impl<T> Signal<MutableProvider<T, ()>> {
    pub fn new(x: T) -> Self {
        Self(Rc::new(MutableProvider::new(x, ())))
    }
}

impl<P: SignalProvider + ?Sized> Signal<DeferProvider<P>> {
    pub fn set(&self, target: Signal<P>) -> Result<(), Rc<P>> {
        self.0.provider.set(target.0.clone())
    }
}

pub type DynSignal<T> = Signal<dyn SignalProvider<Item = T>>; // Removing the stuff to handle smarter const folding to write less code
pub type MutableSignal<T, Inputs = ()> = Signal<MutableProvider<T, Inputs>>;
pub type DynMutableSignal<T> = Signal<dyn SignalProviderMut<Item = T>>;
pub type DynDeferSignal<T> = Signal<DeferProvider<dyn SignalProvider<Item = T>>>;
pub type ConstSignal<T> = Signal<ConstProvider<T>>;

// A mechanism for declaring that something that isn't a signal cares about checking when a signal changes
pub struct Sampler<Provider: SignalProvider + ?Sized> {
    node: SignalNodeId,
    provider: Rc<Provider>,
}
impl<Provider: SignalProvider + ?Sized> Sampler<Provider> {
    pub fn new(signal: Signal<Provider>) -> Self {
        let node = new_node::<Self>(NodeColor::Changed);
        add_dependency(signal.0.get_node(), node.clone());
        Sampler {
            node,
            provider: signal.0,
        }
    }

    /// If the value didn't change, calls force() and returns the value anyway if force returns true.
    pub fn partial_sample(
        &self,
        force: impl Fn(&Provider::Item) -> bool,
    ) -> Option<DynRef<'_, Provider::Item>> {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {
                if force(&self.provider.get_ref()) {
                    Some(self.provider.get_ref())
                } else {
                    None
                }
            }
            _ => {
                self.provider.update();
                let color = self.node.0.borrow().color;
                match color {
                    NodeColor::Changed => {
                        notify_children_change(&self.node);
                        self.node.0.borrow_mut().color = NodeColor::Ready;
                        Some(self.provider.get_ref())
                    }
                    _ => {
                        self.node.0.borrow_mut().color = NodeColor::Ready;
                        None
                    }
                }
            }
        }
    }

    pub fn sample(&self) -> Option<DynRef<'_, Provider::Item>> {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => None,
            _ => {
                self.provider.update();
                let color = self.node.0.borrow().color;
                match color {
                    NodeColor::Changed => {
                        notify_children_change(&self.node);
                        self.node.0.borrow_mut().color = NodeColor::Ready;
                        Some(self.provider.get_ref())
                    }
                    _ => {
                        self.node.0.borrow_mut().color = NodeColor::Ready;
                        None
                    }
                }
            }
        }
    }

    /// This is an unconditional sample()
    pub fn inspect(&mut self) -> DynRef<'_, Provider::Item> {
        self.provider.update();
        self.provider.get_ref()
    }

    pub fn notify(&mut self, f: impl Fn() + 'static) {
        self.node.0.borrow_mut().callback = Some(Box::new(f));
    }
}

//Scripting language interface
//This API is intended to be used from contexts that don't support an applicative idiom, do notation, etc.
//such as scripting languages interoperating with the rust code.
//Rust code should use combinators for large scale composition and fall back to using the macros for small or awkward expressions.
//Languages with neither macros nor a built in applicative syntax can fake it with this tool.
//However, they should follow the same guideline of preferring combinators for most large scale composition and use this only for specific things.
thread_local! {
static DYNAMIC_DEPS: std::cell::RefCell<Option<SmallSet<4, SignalNodeId>>> = const { std::cell::RefCell::new(None) };
}

pub struct DynamicSignalProvider<T, F: Fn() -> T> {
    node: SignalNodeId,
    lastdeps: RefCell<SmallSet<4, SignalNodeId>>,
    f: F,
    val: RefCell<Option<T>>,
}

impl<T, F: Fn() -> T> std::fmt::Debug for DynamicSignalProvider<T, F>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("DynamicSignal");
        dbg.field("f", &format_args!("{}", std::any::type_name::<F>()))
            .field("node", &self.node)
            .field("value", &self.val)
            .finish()
    }
}

impl<T, F: Fn() -> T> SignalProvider for DynamicSignalProvider<T, F> {
    type Item = T;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        //TODO: Make this smarter so it only runs updates on the minimally required set and avoid unneeded recomputation
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                let push_deps = DYNAMIC_DEPS.replace(Some(SmallSet::new()));

                let res = (self.f)();
                *self.val.borrow_mut() = Some(res);

                let new_deps = DYNAMIC_DEPS
                    .replace(push_deps)
                    .expect("something didn't balance use of the DYNAMIC_DEPS");
                let mut lastdeps = self.lastdeps.borrow_mut();
                for removed in lastdeps.difference(&new_deps) {
                    remove_dependency(removed, &self.node);
                }
                for added in new_deps.difference(&lastdeps) {
                    add_dependency(added, self.node.clone());
                }
                *lastdeps = new_deps;
            }
        }
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, T> {
        DynRef::Cell(Ref::map(self.val.borrow(), |x| {
            x.as_ref().expect("must be updated before getting value")
        }))
    }
}

pub fn new_dynamic_signal<T, F: Fn() -> T>(f: F) -> Signal<DynamicSignalProvider<T, F>> {
    Signal(Rc::new(DynamicSignalProvider {
        node: new_node::<DynamicSignalProvider<T, F>>(NodeColor::Changed),
        lastdeps: RefCell::new(SmallSet::new()),
        f,
        val: RefCell::new(None),
    }))
}

/// This is not a general purpose get trait DO NOT USE IT AS SUCH
trait DynamicSignalGettable<T> {
    /// This is not a general purpose get operation DO NOT USE IT AS SUCH
    /// This is for use in dynamic languages to create Dynamic Signals.
    /// The None return should be mapped to a language appropriate exception or error, and the Some regarded as the normal return.
    /// This should be accessible as conveniently as possible on signals, for example mapped to the call operator or a short method name.
    fn get(self) -> Option<T>;
}

impl<T: Clone, Provider: SignalProvider<Item = T>> DynamicSignalGettable<T> for Signal<Provider> {
    fn get(self) -> Option<T> {
        DYNAMIC_DEPS.with_borrow_mut(|deps| match deps {
            None => None,
            Some(deps) => {
                deps.insert(self.0.get_node().clone());
                self.0.update();
                Some(self.0.get_ref().clone())
            }
        })
    }
}

#[derive(Debug)]
pub(crate) struct NotifySignal {
    node: SignalNodeId,
    track: [SmallSet<4, SignalNodeId>; 2],
    flip: bool,
}

impl Default for NotifySignal {
    fn default() -> Self {
        Self::new(None)
    }
}

impl NotifySignal {
    pub fn new(callback: Option<Box<dyn Fn()>>) -> Self {
        let node = new_node::<Self>(NodeColor::Ready);
        node.0.borrow_mut().callback = callback;

        Self {
            node,
            track: Default::default(),
            flip: false,
        }
    }

    pub fn add_parent(&mut self, parent: &Signal<impl SignalProvider + ?Sized>) {
        let node = parent.0.get_node();
        self.track[self.flip as usize].insert(node.clone());

        if !self.track[(!self.flip) as usize].remove(node) {
            add_dependency(node, self.node.clone());
        }
    }

    pub fn reset(&mut self) {
        // TODO: change to drain if SmallSet adds support for it.
        for id in self.track[(!self.flip) as usize].iter() {
            remove_dependency(id, &self.node);
        }
        self.track[(!self.flip) as usize].clear();
        self.flip = !self.flip;
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }
}

impl Drop for NotifySignal {
    fn drop(&mut self) {
        for s in &self.track {
            for id in s {
                remove_dependency(id, &self.node);
            }
        }
    }
}

//TODO: Signal idiom macro
//TODO: more convenience combinators
//TODO: UNIT TESTS

//FIXME: Lifetimes needed to establish invariant that signals don't outlive the functions needed to compute them (and signals don't outlive their inputs just in case signal contents have limited lifetimes)
pub fn empty_signal() -> Signal<ConstProvider<()>> {
    Signal(Rc::new(ConstProvider::new(())))
}

pub fn zip_pair<
    T1,
    T2,
    R: SignalDebug,
    F: Fn(&P1::Item, &P2::Item) -> R,
    P1: SignalProvider<Item = T1> + ?Sized,
    P2: SignalProvider<Item = T2> + ?Sized,
>(
    p1: Signal<P1>,
    p2: Signal<P2>,
    f: F,
) -> Signal<impl SignalProvider<Item = R>> {
    map(
        move |arg| f(&arg.0, &arg.1),
        zip((p1, p2)),
        never_eq,
        never_debug,
    )
}

pub fn sample<T, P: SignalProvider<Item = T> + ?Sized>(signal: &Signal<P>) -> DynRef<'_, T> {
    signal.0.update();
    signal.0.get_ref()
}

pub fn sample_val<T: Clone, P: SignalProvider<Item = T> + ?Sized>(signal: &Signal<P>) -> T {
    let p = signal;
    p.0.update();
    p.0.get_ref().clone()
}

// Animated signals:
// This API is designed for adding easing to data driven objects, making something transition smoothly from the old value to the new value
// It is not designed for playing "authored" animations with keyframes on a timeline.
// It's designed to make ergonomic reactive content simple without requiring any extra event plumbing or timekeeping
// For keyframed animations, consider using a state machine to trigger them and then just using signal composition to work with animation times
pub enum AnimationOutput<Output> {
    Finish(Output),
    Continue(Output),
}

impl<Output: fmt::Debug> fmt::Debug for AnimationOutput<Output> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Finish(arg0) => f.debug_tuple("Finish").field(arg0).finish(),
            Self::Continue(arg0) => f.debug_tuple("Continue").field(arg0).finish(),
        }
    }
}

pub trait Animator<T> {
    type Output: Clone;
    type State;

    fn init(&self, x: T) -> Self::State;
    fn compute(
        &self,
        x: T,
        momentum: &mut Self::State,
        delta_t: Option<f32>,
    ) -> AnimationOutput<Self::Output>;
}

pub struct AnimProvider<
    T: Clone,
    Anim: Animator<T>,
    Source: SignalProvider<Item = T> + ?Sized,
    Time: SignalProvider<Item = Microseconds>,
> {
    node: SignalNodeId,
    val: RefCell<Option<Anim::Output>>,
    state: RefCell<Option<(bool, Microseconds, Anim::State)>>,
    anim: Anim,
    time: Rc<Time>,
    input: Rc<Source>,
    #[cfg(any(debug_assertions, feature = "signal-backtrace"))]
    debug_frame: backtrace::BacktraceFrame,
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, derive_more::Display)]
pub struct Microseconds(pub u64);

impl Microseconds {
    pub fn diff_to_f32(&self, other: &Microseconds) -> f32 {
        ((self.0 - other.0) as f32) / 1_000_000f32
    }
}

impl<
    T: Clone,
    Anim: Animator<T> + fmt::Debug,
    Source: SignalProvider<Item = T> + ?Sized + fmt::Debug,
    Time: SignalProvider<Item = Microseconds> + fmt::Debug,
> fmt::Debug for AnimProvider<T, Anim, Source, Time>
where
    <Anim as Animator<T>>::Output: fmt::Debug,
    <Anim as Animator<T>>::State: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dbg = f.debug_struct("AnimSignal");
        dbg.field("node", &self.node)
            .field("value", &self.val)
            .field("anim", &self.anim)
            .field("time", &self.time)
            .field("input", &self.input);

        if let Some((a, b, r)) = &*self.state.borrow() {
            dbg.field("state", &format_args!("{}, {}, {:?}", a, b, r));
        }

        dbg.finish()
    }
}

impl<
    T: Clone,
    Anim: Animator<T>,
    Source: SignalProvider<Item = T> + ?Sized,
    Time: SignalProvider<Item = Microseconds>,
> SignalProvider for AnimProvider<T, Anim, Source, Time>
{
    type Item = Anim::Output;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        //eprintln!("animprovider state {:?}", *self.state.borrow());
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                self.input.update();
                match color {
                    NodeColor::Changed => {
                        let inval = self.input.get_ref().clone(); // FIXME: Can this be made a reference?
                        self.time.update();
                        let timeval = *self.time.get_ref();
                        let mut state = self.state.borrow_mut();
                        match &mut *state {
                            Option::None => {
                                let mut animstate = self.anim.init(inval.clone()); // FIXME: is clone ok
                                let animres = self.anim.compute(inval, &mut animstate, None);
                                let mut active = false;
                                //eprintln!("animres is now {:?}", animres);
                                match animres {
                                    AnimationOutput::Finish(val) => {
                                        remove_dependency(self.time.get_node(), &self.node);
                                        active = false;
                                        *self.val.borrow_mut() = Some(val);
                                        notify_children_change(&self.node);
                                    }
                                    AnimationOutput::Continue(val) => {
                                        add_dependency(self.time.get_node(), self.node.clone());
                                        active = true;
                                        *self.val.borrow_mut() = Some(val);
                                        notify_children_change(&self.node);
                                    }
                                }
                                //eprintln!("active is initialized with {:?}", active);
                                *state = Some((active, timeval, animstate));
                            }
                            Some((active, prevtime, animstate)) => {
                                let delta_t = if *active {
                                    Some(timeval.diff_to_f32(prevtime))
                                } else {
                                    None
                                };
                                *prevtime = timeval;
                                let animres = self.anim.compute(inval, animstate, delta_t);
                                //eprintln!("animres is now {:?}", animres);
                                match animres {
                                    AnimationOutput::Finish(val) => {
                                        remove_dependency(self.time.get_node(), &self.node);
                                        *active = false;
                                        *self.val.borrow_mut() = Some(val);
                                        notify_children_change(&self.node);
                                    }
                                    AnimationOutput::Continue(val) => {
                                        add_dependency(self.time.get_node(), self.node.clone());
                                        *active = true;
                                        *self.val.borrow_mut() = Some(val);
                                        notify_children_change(&self.node);
                                    }
                                }
                                eprintln!("active is now {:?}", *active);
                            }
                        }
                    }
                    _ => {
                        let inval = self.input.get_ref().clone();
                        self.time.update();
                        match color {
                            _ => {
                                panic!("what should be here?");
                            }
                            NodeColor::Changed => {
                                let timeval = *self.time.get_ref();
                                let mut state = self.state.borrow_mut();
                                match &mut *state {
                                    Option::None => {
                                        let mut animstate = self.anim.init(inval.clone()); // FIXME: is clone ok
                                        let animres =
                                            self.anim.compute(inval, &mut animstate, None);
                                        let mut active = false;
                                        match animres {
                                            AnimationOutput::Finish(val) => {
                                                remove_dependency(self.time.get_node(), &self.node);
                                                active = false;
                                                *self.val.borrow_mut() = Some(val);
                                                notify_children_change(&self.node);
                                            }
                                            AnimationOutput::Continue(val) => {
                                                add_dependency(
                                                    self.time.get_node(),
                                                    self.node.clone(),
                                                );
                                                active = true;
                                                *self.val.borrow_mut() = Some(val);
                                                notify_children_change(&self.node);
                                            }
                                        }
                                        *state = Some((active, timeval, animstate));
                                    }
                                    Some((active, prevtime, animstate)) => {
                                        let delta_t = if *active {
                                            Some(timeval.diff_to_f32(prevtime))
                                        } else {
                                            None
                                        };
                                        let animres = self.anim.compute(inval, animstate, delta_t);
                                        match animres {
                                            AnimationOutput::Finish(val) => {
                                                remove_dependency(self.time.get_node(), &self.node);
                                                *active = false;
                                                *self.val.borrow_mut() = Some(val);
                                                notify_children_change(&self.node);
                                            }
                                            AnimationOutput::Continue(val) => {
                                                add_dependency(
                                                    self.time.get_node(),
                                                    self.node.clone(),
                                                );
                                                *active = true;
                                                *self.val.borrow_mut() = Some(val);
                                                notify_children_change(&self.node);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                self.node.0.borrow_mut().color = NodeColor::Ready;
            }
        }
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, <Anim as Animator<T>>::Output> {
        DynRef::Cell(Ref::map(self.val.borrow(), |x| {
            x.as_ref().expect("must be updated before getting value")
        }))
    }
}

pub fn animate<
    T: Clone,
    S: SignalProvider<Item = T> + ?Sized,
    Anim: Animator<T>,
    Time: SignalProvider<Item = Microseconds> + Sized,
>(
    p: Signal<S>,
    anim: Anim,
    time: Signal<Time>,
) -> Signal<AnimProvider<T, Anim, S, Time>> {
    let node = new_node::<AnimProvider<T, Anim, S, Time>>(NodeColor::Changed);
    let input = p.0;
    add_dependency(input.get_node(), node.clone());
    Signal(Rc::new(AnimProvider {
        node,
        val: RefCell::new(None),
        state: RefCell::new(None),
        anim,
        time: time.0,
        input,
        debug_frame: backtrace::Backtrace::new_unresolved().frames()[6].clone(),
    }))
}

/// Trait describing any binary operation
pub trait SignalOp<T, Rhs, Output> {
    fn apply(lhs: T, rhs: Rhs) -> Output;
}

// Operation Marker Types
mod marker {
    pub struct AddOp;
    pub struct SubOp;
    pub struct MulOp;
    pub struct DivOp;
    pub struct RemOp;
    pub struct BitAndOp;
    pub struct BitOrOp;
    pub struct BitXorOp;
    pub struct ShlOp;
    pub struct ShrOp;
    pub struct EqOp;
    pub struct OrdOp;
    pub struct LtOp;
    pub struct LeOp;
    pub struct GtOp;
    pub struct GeOp;
    pub struct MinOp;
    pub struct MaxOp;
}

use marker::*;

macro_rules! gen_binop {
    ($t:tt, $marker:path, $op:tt) => {
        impl<T: std::ops::$t<Rhs, Output = Output>, Rhs, Output> SignalOp<T, Rhs, Output>
            for $marker
        {
            fn apply(lhs: T, rhs: Rhs) -> Output {
                lhs $op rhs
            }
        }
    };
}

gen_binop!(Add, AddOp, +);
gen_binop!(Sub, SubOp, -);
gen_binop!(Mul, MulOp, *);
gen_binop!(Div, DivOp, /);
gen_binop!(Rem, RemOp, %);
gen_binop!(BitAnd, BitAndOp, &);
gen_binop!(BitOr, BitOrOp, |);
gen_binop!(BitXor, BitXorOp, ^);
gen_binop!(Shl, ShlOp, <<);
gen_binop!(Shr, ShrOp, >>);

impl<T: PartialEq<Rhs>, Rhs> SignalOp<T, Rhs, bool> for EqOp {
    fn apply(lhs: T, rhs: Rhs) -> bool {
        lhs.eq(&rhs)
    }
}

impl<T: PartialOrd<Rhs>, Rhs> SignalOp<T, Rhs, Option<std::cmp::Ordering>> for OrdOp {
    fn apply(lhs: T, rhs: Rhs) -> Option<std::cmp::Ordering> {
        lhs.partial_cmp(&rhs)
    }
}

macro_rules! gen_cmpop {
    ($marker:path, $op:tt) => {
        impl<T: PartialOrd<Rhs>, Rhs> SignalOp<T, Rhs, bool> for $marker {
            fn apply(lhs: T, rhs: Rhs) -> bool {
                lhs.partial_cmp(&rhs).is_some_and(std::cmp::Ordering::$op)
            }
        }
    };
}

gen_cmpop!(LtOp, is_lt);
gen_cmpop!(LeOp, is_le);
gen_cmpop!(GtOp, is_gt);
gen_cmpop!(GeOp, is_ge);

impl<T: Ord> SignalOp<T, T, T> for MinOp {
    fn apply(lhs: T, rhs: T) -> T {
        std::cmp::min(lhs, rhs)
    }
}

impl<T: Ord> SignalOp<T, T, T> for MaxOp {
    fn apply(lhs: T, rhs: T) -> T {
        std::cmp::max(lhs, rhs)
    }
}

pub struct OpProvider<
    P1: SignalProvider + ?Sized,
    P2: SignalProvider + ?Sized,
    Output,
    OP: SignalOp<P1::Item, P2::Item, Output>,
> {
    provider1: Rc<P1>,
    provider2: Rc<P2>,
    res: RefCell<Option<Output>>,
    node: SignalNodeId,
    phantom: PhantomData<(P1::Item, P2::Item, OP)>,
    debug_frame: backtrace::BacktraceFrame,
}

impl<
    Output: fmt::Debug,
    P1: SignalProvider + ?Sized + fmt::Debug,
    P2: SignalProvider + ?Sized + fmt::Debug,
    OP: SignalOp<P1::Item, P2::Item, Output>,
> fmt::Debug for OpProvider<P1, P2, Output, OP>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dbg = f.debug_struct("OpProvider");
        dbg.field("node", &self.node)
            .field("provider1", &self.provider1)
            .field("provider2", &self.provider2)
            .field("res", &self.res)
            .finish()
    }
}

impl<
    Output: SignalDebug,
    P1: SignalProvider + ?Sized,
    P2: SignalProvider + ?Sized,
    OP: SignalOp<P1::Item, P2::Item, Output>,
> SignalProvider for OpProvider<P1, P2, Output, OP>
where
    P1::Item: Clone,
    P2::Item: Clone,
{
    type Item = Output;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => rewrite_panic!(
                self.debug_frame,
                {
                    self.provider1.update();
                    self.provider2.update();
                },
                self.res.as_ptr()
            ),
        }
        let color = self.node.0.borrow().color;
        if color == NodeColor::Changed {
            let res = rewrite_panic!(
                self.debug_frame,
                OP::apply(
                    self.provider1.get_ref().clone(),
                    self.provider2.get_ref().clone(),
                ),
                self.res.as_ptr()
            );

            *self.res.borrow_mut() = Some(res);
            notify_children_change(&self.node);
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, Output> {
        DynRef::Cell(Ref::map(self.res.borrow(), |x| {
            x.as_ref().expect("must be updated before getting value")
        }))
    }
}

#[inline]
fn operate<
    P1: SignalProvider + ?Sized,
    P2: SignalProvider + ?Sized,
    Output: SignalDebug,
    OP: SignalOp<P1::Item, P2::Item, Output>,
>(
    lhs: Signal<P1>,
    rhs: Signal<P2>,
) -> Signal<OpProvider<P1, P2, Output, OP>>
where
    P1::Item: Clone,
    P2::Item: Clone,
{
    let node = new_node::<OpProvider<P1, P2, Output, OP>>(NodeColor::Changed);
    let provider1 = lhs.0;
    let provider2 = rhs.0;
    add_dependency(provider1.get_node(), node.clone());
    add_dependency(provider2.get_node(), node.clone());
    Signal(Rc::new(OpProvider {
        provider1,
        provider2,
        res: RefCell::new(None),
        node,
        phantom: PhantomData,
        debug_frame: backtrace::Backtrace::new_unresolved().frames()[6].clone(),
    }))
}

macro_rules! gen_binop_impl {
    ($t:tt, $marker:path, $op:tt) => {
        impl<AP: SignalProvider + ?Sized, BP: SignalProvider + ?Sized> std::ops::$t<Signal<BP>>
            for Signal<AP>
        where
            AP::Item: std::ops::$t<BP::Item> + Clone,
            BP::Item: Clone,
            <AP::Item as std::ops::$t<BP::Item>>::Output: Clone + SignalDebug,
        {
            type Output =
                Signal<OpProvider<AP, BP, <AP::Item as std::ops::$t<BP::Item>>::Output, $marker>>;

            fn $op(self, rhs: Signal<BP>) -> Self::Output {
                operate(self, rhs)
            }
        }
    };
}

gen_binop_impl!(Add, AddOp, add);
gen_binop_impl!(Sub, SubOp, sub);
gen_binop_impl!(Mul, MulOp, mul);
gen_binop_impl!(Div, DivOp, div);
gen_binop_impl!(Rem, RemOp, rem);
gen_binop_impl!(BitAnd, BitAndOp, bitand);
gen_binop_impl!(BitOr, BitOrOp, bitor);
gen_binop_impl!(BitXor, BitXorOp, bitxor);
gen_binop_impl!(Shl, ShlOp, shl);
gen_binop_impl!(Shr, ShrOp, shr);

macro_rules! gen_cmpop_impl {
    ($t:ident, $marker:path) => {
        pub fn $t<AP: SignalProvider + ?Sized, BP: SignalProvider + ?Sized>(
            a: Signal<AP>,
            b: Signal<BP>,
        ) -> Signal<impl SignalProvider<Item = bool>>
        where
            AP::Item: Clone + PartialOrd<BP::Item>,
            BP::Item: Clone,
        {
            operate::<AP, BP, bool, $marker>(a, b)
        }
    };
}
pub mod cmp {
    use super::marker::*;
    use super::{Signal, SignalProvider, operate};

    pub fn eq<AP: SignalProvider + ?Sized, BP: SignalProvider + ?Sized>(
        a: Signal<AP>,
        b: Signal<BP>,
    ) -> Signal<impl SignalProvider<Item = bool>>
    where
        AP::Item: Clone + PartialEq<BP::Item>,
        BP::Item: Clone,
    {
        operate::<AP, BP, bool, EqOp>(a, b)
    }

    gen_cmpop_impl!(le, LeOp);
    gen_cmpop_impl!(lt, LtOp);
    gen_cmpop_impl!(ge, GeOp);
    gen_cmpop_impl!(gt, GtOp);

    pub fn min<AP: SignalProvider + ?Sized>(
        a: Signal<AP>,
        b: Signal<AP>,
    ) -> Signal<impl SignalProvider<Item = AP::Item>>
    where
        AP::Item: Clone + Ord + super::SignalDebug,
    {
        operate::<AP, AP, AP::Item, MinOp>(a, b)
    }

    pub fn max<AP: SignalProvider + ?Sized>(
        a: Signal<AP>,
        b: Signal<AP>,
    ) -> Signal<impl SignalProvider<Item = AP::Item>>
    where
        AP::Item: Clone + Ord + super::SignalDebug,
    {
        operate::<AP, AP, AP::Item, MaxOp>(a, b)
    }

    pub fn partial_cmp<AP: SignalProvider + ?Sized, BP: SignalProvider + ?Sized>(
        a: Signal<AP>,
        b: Signal<BP>,
    ) -> Signal<impl SignalProvider<Item = Option<std::cmp::Ordering>>>
    where
        AP::Item: Clone + PartialOrd<BP::Item>,
        BP::Item: Clone,
    {
        operate::<AP, BP, Option<std::cmp::Ordering>, OrdOp>(a, b)
    }
}

impl<P: SignalProvider<Item = bool> + ?Sized> Signal<P> {
    pub fn cond<T: Clone + SignalDebug>(
        self,
        t: DynSignal<T>,
        f: DynSignal<T>,
    ) -> Signal<impl SignalProvider<Item = T>> {
        join(self.map_ex(move |cc| if *cc { t.clone() } else { f.clone() }))
    }
}

#[macro_export]
macro_rules! switch {
    ($e:expr, $($param:expr => $pattern:pat $(if $guard:expr)?),+ $(,)?) => {
        $crate::join($e.map(move |x|
        match x {
            $($pattern $(if $guard)? => $param.clone()),+
        }))
    };
}

pub struct VecMapProvider<
    T1,
    T2,
    Key: Eq + Hash,
    F: Fn(&T1) -> T2,
    Ex: Fn(&T1) -> Key,
    P: SignalProvider<Item = Vector<T1, Ptr, CHUNK_SIZE>> + ?Sized,
    Ptr: imbl::shared_ptr::SharedPointerKind,
    const CHUNK_SIZE: usize,
> {
    provider: Rc<P>,
    map: RefCell<imbl::vector::PersistentMap<T1, T2, Key, Ptr, F, Ex, CHUNK_SIZE>>,
    res: RefCell<Vector<T2, Ptr, CHUNK_SIZE>>,
    node: SignalNodeId,
    #[cfg(any(debug_assertions, feature = "signal-backtrace"))]
    debug_frame: backtrace::BacktraceFrame,
}

impl<
    T1: 'static,
    T2: 'static + fmt::Debug,
    Key: Eq + Hash + 'static,
    F: (Fn(&T1) -> T2) + 'static,
    Ex: (Fn(&T1) -> Key) + 'static,
    P: SignalProvider<Item = Vector<T1, Ptr, CHUNK_SIZE>> + ?Sized + 'static + fmt::Debug,
    Ptr: imbl::shared_ptr::SharedPointerKind + 'static,
    const CHUNK_SIZE: usize,
> fmt::Debug for VecMapProvider<T1, T2, Key, F, Ex, P, Ptr, CHUNK_SIZE>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VecMapProvider")
            .field("node", &self.node)
            .field("provider", &self.provider)
            .field("res", &self.res)
            .finish()
    }
}

/// Ensures that T is 'static
fn assert_static<T: 'static>() {}

impl<
    T1: Clone + 'static,
    T2: Clone + 'static + SignalDebug,
    Key: Eq + Hash + 'static,
    F: (Fn(&T1) -> T2) + 'static,
    Ex: (Fn(&T1) -> Key) + 'static,
    P: SignalProvider<Item = Vector<T1, Ptr, CHUNK_SIZE>> + ?Sized + 'static,
    Ptr: imbl::shared_ptr::SharedPointerKind + 'static,
    const CHUNK_SIZE: usize,
> SignalProvider for VecMapProvider<T1, T2, Key, F, Ex, P, Ptr, CHUNK_SIZE>
{
    type Item = Vector<T2, Ptr, CHUNK_SIZE>;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        assert_static::<Self>();
        &self.node
    }

    fn update(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                rewrite_panic!(self.debug_frame, self.provider.update(), self.res.as_ptr());
            }
        }
        let color = self.node.0.borrow().color;
        if color == NodeColor::Changed {
            let res = rewrite_panic!(
                self.debug_frame,
                self.map.borrow_mut().map(&self.provider.get_ref()),
                self.res.as_ptr()
            );
            let changed = !self.res.borrow().ptr_eq(&res);
            *self.res.borrow_mut() = res;
            if changed {
                notify_children_change(&self.node);
            }
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, Vector<T2, Ptr, CHUNK_SIZE>> {
        DynRef::Cell(self.res.borrow())
    }
}

impl<
    T1: Clone + 'static,
    Ptr: imbl::shared_ptr::SharedPointerKind + 'static,
    const CHUNK_SIZE: usize,
    P: SignalProvider<Item = Vector<T1, Ptr, CHUNK_SIZE>> + ?Sized + 'static,
> Signal<P>
{
    /// Creates a new immutable vector by mapping all elements of this vector into new elements.
    pub fn map_elements<
        T2: Clone + SignalDebug + 'static,
        Key: Eq + Hash + 'static,
        F: (Fn(&T1) -> T2) + 'static,
        Ex: (Fn(&T1) -> Key) + 'static,
    >(
        self,
        f: F,
        ex: Ex,
    ) -> Signal<VecMapProvider<T1, T2, Key, F, Ex, P, Ptr, CHUNK_SIZE>> {
        let node = new_node::<Self>(NodeColor::Changed);
        let provider = self.0;
        add_dependency(provider.get_node(), node.clone());
        Signal(Rc::new(VecMapProvider {
            provider,
            map: RefCell::new(imbl::vector::PersistentMap::new(f, ex)),
            res: Default::default(),
            node,
            debug_frame: backtrace::Backtrace::new_unresolved().frames()[6].clone(),
        }))
    }

    /// This is a special map that allows you to append or remove elements from an immutable vector, without changing
    /// the type of the vector
    pub fn map_modify(
        self,
        f: impl Fn(&P::Item) -> Vector<T1, Ptr, CHUNK_SIZE>,
    ) -> Signal<impl SignalProvider<Item = Vector<T1, Ptr, CHUNK_SIZE>>>
    where
        T1: SignalDebug,
    {
        map(f, self, Vector::ptr_eq, never_debug)
    }
}

pub struct VecFoldProvider<
    T,
    F: FnMut(T, T) -> T,
    P: SignalProvider<Item = Vector<T, Ptr, CHUNK_SIZE>> + ?Sized,
    Ptr: imbl::shared_ptr::SharedPointerKind,
    const CHUNK_SIZE: usize,
> {
    provider: Rc<P>,
    fold: RefCell<imbl::vector::PersistentFold<T, F, Ptr, CHUNK_SIZE>>,
    res: RefCell<T>,
    node: SignalNodeId,
    z: T,
    #[cfg(any(debug_assertions, feature = "signal-backtrace"))]
    debug_frame: backtrace::BacktraceFrame,
}

impl<
    T: Clone + fmt::Debug,
    F: FnMut(T, T) -> T,
    P: SignalProvider<Item = Vector<T, Ptr, CHUNK_SIZE>> + ?Sized + fmt::Debug,
    Ptr: imbl::shared_ptr::SharedPointerKind,
    const CHUNK_SIZE: usize,
> fmt::Debug for VecFoldProvider<T, F, P, Ptr, CHUNK_SIZE>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VecFoldProvider")
            .field("node", &self.node)
            .field("provider", &self.provider)
            .field("result", &self.res)
            .finish()
    }
}

impl<
    T: Clone + SignalDebug,
    F: FnMut(T, T) -> T,
    P: SignalProvider<Item = Vector<T, Ptr, CHUNK_SIZE>> + ?Sized,
    Ptr: imbl::shared_ptr::SharedPointerKind,
    const CHUNK_SIZE: usize,
> SignalProvider for VecFoldProvider<T, F, P, Ptr, CHUNK_SIZE>
{
    type Item = T;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                rewrite_panic!(self.debug_frame, self.provider.update(), self.res.as_ptr());
            }
        }
        let color = self.node.0.borrow().color;
        if color == NodeColor::Changed {
            let res = rewrite_panic!(
                self.debug_frame,
                self.fold.borrow_mut().fold(&self.provider.get_ref()),
                self.res.as_ptr()
            );
            // We use unwrap_or_else here because None is a cold branch which avoids a clone() most of the time.
            *self.res.borrow_mut() = res.unwrap_or_else(|| self.z.clone());
            notify_children_change(&self.node);
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, T> {
        DynRef::Cell(self.res.borrow())
    }
}

pub fn fold_vec<
    T: Clone + SignalDebug,
    F: FnMut(T, T) -> T,
    P: SignalProvider<Item = Vector<T, Ptr, CHUNK_SIZE>> + ?Sized,
    Ptr: imbl::shared_ptr::SharedPointerKind,
    const CHUNK_SIZE: usize,
>(
    f: F,
    p: Signal<P>,
    z: T,
) -> Signal<VecFoldProvider<T, F, P, Ptr, CHUNK_SIZE>> {
    let node = new_node::<VecFoldProvider<T, F, P, Ptr, CHUNK_SIZE>>(NodeColor::Changed);
    let provider = p.0;
    add_dependency(provider.get_node(), node.clone());
    Signal(Rc::new(VecFoldProvider {
        provider,
        fold: RefCell::new(imbl::vector::PersistentFold::new(f)),
        res: RefCell::new(z.clone()),
        node,
        z,
        debug_frame: backtrace::Backtrace::new_unresolved().frames()[6].clone(),
    }))
}

pub struct DeferProvider<P: SignalProvider + ?Sized> {
    provider: std::cell::OnceCell<Rc<P>>,
    node: SignalNodeId,
}

impl<P: SignalProvider + ?Sized + std::fmt::Debug> fmt::Debug for DeferProvider<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeferProvider")
            .field("provider", &self.provider)
            .field("node", &self.node)
            .finish()
    }
}

impl<P: SignalProvider + ?Sized> SignalProvider for DeferProvider<P> {
    type Item = P::Item;

    #[inline]
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update(&self) {
        if let Some(p) = self.provider.get() {
            p.update();
        }
    }

    #[inline]
    fn get_ref(&self) -> DynRef<'_, Self::Item> {
        let p = match self.provider.get() {
            Some(x) => x,
            None => panic!(
                "Tried to get {} value before deferred signal was resolved!",
                std::any::type_name::<Self::Item>()
            ),
        };
        p.get_ref()
    }
}

pub fn defer<T1, P: SignalProvider<Item = T1> + ?Sized>() -> Signal<DeferProvider<P>> {
    let node = new_node::<DeferProvider<P>>(NodeColor::Changed);
    Signal(Rc::new(DeferProvider {
        provider: OnceCell::new(),
        node,
    }))
}

#[test]
fn add() {
    let a = 1.to_signal();
    let b = 2.to_signal();
    let input_signal = a + b;
    let problem = *sample(&input_signal);
    assert_eq!(problem, 3);
}

#[test]
fn reactive_fold() {
    let mut vals = Iterator::map(1..=100, MutableSignal::new).collect::<Vec<_>>();
    let sum = vals.iter().fold(0.to_signal().into_dyn(), |a, b| {
        zip_pair(a, b.clone(), |x, y| *x + *y).into_dyn()
    });
    let modifications = vec![(1, 1, 5049), (1, 2, 5050), (5, -1, 5043)];
    for (idx, val, expectation) in modifications {
        vals[idx].replace(val);
        assert!(*sample(&sum) == expectation);
    }
}

#[test]
fn test_reactive_map_vec() {
    let v = const_new(imbl::vector![1, 2, 3, 4]);

    let result = v.clone().map_elements(|x| *x * *x, |x| *x);

    for i in sample(&result).iter() {
        println!("{i}");
    }

    let rv = const_new(imbl::vector![
        Rc::new(1),
        Rc::new(2),
        Rc::new(3),
        Rc::new(4)
    ]);

    let result = rv
        .clone()
        .map_elements(|x| *x.as_ref() * *x.as_ref(), |x| Identity(x.clone()));

    for i in sample(&result).iter() {
        println!("{i}");
    }
}

fn truncate_str(s: &str, n: usize) -> &str {
    if let Some((idx, _)) = s.char_indices().nth(n) {
        &s[..idx]
    } else {
        s
    }
}

#[cfg(debug_assertions)]
fn trace_deps_node(
    id: &SignalNodeId,
    edges: &mut std::collections::HashSet<(usize, usize)>,
    nodes: &mut std::collections::HashSet<usize>,
) -> String {
    let this = id.0.as_ptr() as usize;

    let mut result: Vec<String> =
        id.0.borrow()
            .children
            .iter()
            .flat_map(|child| {
                if edges.insert((this, child.0.as_ptr() as usize)) {
                    Some(format!(
                        "id{} -> id{}\n{}",
                        this,
                        child.0.as_ptr() as usize,
                        trace_deps_node(child, edges, nodes)
                    ))
                } else {
                    None
                }
            })
            .collect();

    if nodes.insert(this) {
        let color = match id.0.borrow().color {
            NodeColor::Ready => "lime",
            NodeColor::Changed => "yellow",
            NodeColor::Check => "pink",
        };

        let label =
            id.0.borrow()
                .debug
                .replace("feather_ui::", "")
                .replace("reactive::", "")
                .replace("alloc::rc::", "")
                .replace("alloc::string::", "")
                .replace("cosmic_text::buffer::", "")
                .replace("dyn SignalProvider", "DynSignal");
        result.push(format!(
            "id{} [shape=box style=\"filled\" fillcolor=\"{}\" label=\"{}\"];",
            this,
            color,
            truncate_str(&label, 90)
        ));
    }

    result.join("\n")
}

#[cfg(debug_assertions)]
pub fn trace_deps<P: SignalProvider + ?Sized>(sig: Signal<P>) -> String {
    format!(
        "digraph G {{\n{}\n}}",
        trace_deps_node(
            sig.0.get_node(),
            &mut std::collections::HashSet::new(),
            &mut std::collections::HashSet::new()
        )
    )
}

fn verify_subtree(id: &SignalNodeId, mut color: NodeColor) {
    match id.0.borrow().color {
        NodeColor::Ready => assert_eq!(color, NodeColor::Ready),
        NodeColor::Changed => (),
        NodeColor::Check => assert_ne!(color, NodeColor::Ready),
    }
    color = id.0.borrow().color;
    for child in &id.0.borrow().children {
        verify_subtree(child, color);
    }
}
// Verifies the subtree of node dependencies startig at the given node.
pub fn verify_tree(id: &SignalNodeId) {
    let color = id.0.borrow().color;
    #[cfg(debug_assertions)]
    verify_subtree(id, color);
}

#[cfg(debug_assertions)]
// Verifies the subtree of node dependencies
pub fn verify_signal<P: SignalProvider + ?Sized>(id: &Signal<P>) {
    verify_tree(id.0.get_node());
}

#[cfg(any(debug_assertions, feature = "signal-backtrace"))]
pub(super) struct UnwindPayload {
    frame: backtrace::BacktraceFrame,
    inner: Box<dyn std::any::Any + Send + 'static>,
    value: Option<*const dyn std::fmt::Debug>,
}

// We don't actually care about value being Send because we never attempt to actually recover from an UnwindPayload error.
unsafe impl Send for UnwindPayload {}

#[cfg(any(debug_assertions, feature = "signal-backtrace"))]
impl UnwindPayload {
    fn backtrace_fmt(&self, f: &mut backtrace::BacktraceFmt) -> fmt::Result {
        if let Some(inner) = self.inner.downcast_ref::<UnwindPayload>() {
            inner.backtrace_fmt(f)?;
        } else {
            let msg = match self.inner.downcast_ref::<&'static str>() {
                Some(s) => *s,
                None => match self.inner.downcast_ref::<String>() {
                    Some(s) => &s[..],
                    None => "Opaque Inner Payload",
                },
            };
            let thread = std::thread::current();
            let name = thread.name().unwrap_or("<unnamed>");
            writeln!(f.formatter(), "thread '{}' panicked, {}", name, msg)?;
        }

        if let Some(v) = self.value {
            write!(f.formatter(), "  ")?;
            unsafe { v.as_ref().unwrap().fmt(f.formatter())? };
            writeln!(f.formatter())?;
        }

        let mut frame = self.frame.clone();
        frame.resolve();
        f.frame().backtrace_frame(&frame)
    }
}

fn output_filename(
    fmt: &mut fmt::Formatter<'_>,
    bows: BytesOrWideString<'_>,
    print_fmt: backtrace::PrintFmt,
    cwd: Option<&std::path::PathBuf>,
) -> fmt::Result {
    let file: std::borrow::Cow<'_, Path> = match bows {
        #[cfg(unix)]
        BytesOrWideString::Bytes(bytes) => {
            use crate::os::unix::prelude::*;
            Path::new(crate::ffi::OsStr::from_bytes(bytes)).into()
        }
        #[cfg(not(unix))]
        BytesOrWideString::Bytes(bytes) => {
            Path::new(std::str::from_utf8(bytes).unwrap_or("<unknown>")).into()
        }
        #[cfg(windows)]
        BytesOrWideString::Wide(wide) => {
            use std::borrow::Cow;
            use std::os::windows::prelude::*;
            Cow::Owned(std::ffi::OsString::from_wide(wide).into())
        }
        #[cfg(not(windows))]
        BytesOrWideString::Wide(_wide) => Path::new("<unknown>").into(),
    };
    if print_fmt == backtrace::PrintFmt::Short && file.is_absolute()
        && let Some(cwd) = cwd
            && let Ok(stripped) = file.strip_prefix(cwd)
                && let Some(s) = stripped.to_str() {
                    return write!(fmt, ".{}{s}", std::path::MAIN_SEPARATOR);
                }
    fmt::Display::fmt(&file.display(), fmt)
}

#[cfg(any(debug_assertions, feature = "signal-backtrace"))]
impl std::fmt::Display for UnwindPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let style = if f.alternate() {
            backtrace::PrintFmt::Full
        } else {
            backtrace::PrintFmt::Short
        };

        let cwd = std::env::current_dir();
        let mut print_path = move |fmt: &mut fmt::Formatter<'_>, path: BytesOrWideString<'_>| {
            output_filename(fmt, path, style, cwd.as_ref().ok())
        };

        let mut f = backtrace::BacktraceFmt::new(f, style, &mut print_path);
        f.add_context()?;
        self.backtrace_fmt(&mut f)?;
        f.finish()
    }
}
