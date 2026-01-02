use imbl::GenericVector;
use smolset::SmolSet;
use stable_deref_trait::CloneStableDeref;
use std::{
    cell::Ref,
    cmp::{PartialEq, PartialOrd},
    hash::Hash,
    marker::PhantomData,
    ops::Deref,
    rc::Rc,
};
use tuple_list::{Tuple, TupleList};

use std::cell::RefCell;

#[derive(Debug, Clone, Copy)]
enum NodeColor {
    Ready,
    Changed,
    Check,
}

pub enum DynRef<'a, T> {
    Ref(&'a T),
    Cell(Ref<'a, T>),
}

impl<'a, T> Deref for DynRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            DynRef::Ref(v) => *v,
            DynRef::Cell(v) => &*v,
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
    children: SmolSet<[SignalNodeId; 4]>,
    color: NodeColor,
    callback: Option<Box<dyn Fn()>>,
}

impl std::fmt::Debug for SignalNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignalNode")
            .field("children", &self.children)
            .field("color", &self.color)
            .finish()
    }
}

fn notify_check(nodeid: &SignalNodeId) {
    let mut node = nodeid.0.borrow_mut();
    match node.color {
        NodeColor::Ready => {
            node.color = NodeColor::Check;
            node.children.iter().for_each(notify_check);
        }
        _ => {}
    }
    if let Some(f) = &node.callback {
        f();
    }
}

fn notify_change(nodeid: &SignalNodeId) {
    let mut node = nodeid.0.borrow_mut();
    node.color = NodeColor::Changed;
    node.children.iter().for_each(notify_check);
}

fn notify_children_change(nodeid: &SignalNodeId) {
    let node = nodeid.0.borrow();
    for child in node.children.iter() {
        assert!(!Rc::ptr_eq(&nodeid.0, &child.0));
        notify_change(child);
    }
    node.children.iter().for_each(notify_change);
}

fn add_dependency(parent: &SignalNodeId, child: SignalNodeId) {
    parent.0.borrow_mut().children.insert(child);
}

fn remove_dependency(parent: &SignalNodeId, child: &SignalNodeId) {
    parent.0.borrow_mut().children.remove(child);
}

fn new_node(color: NodeColor) -> SignalNodeId {
    Identity(Rc::new(RefCell::new(SignalNode {
        children: SmolSet::new(),
        color: color,
        callback: None,
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

pub trait SignalProvider: std::fmt::Debug {
    type Item;

    fn get_node(&self) -> &SignalNodeId;
    fn get_ref(&self) -> DynRef<'_, Self::Item>;
    // TODO: does this have equivelent performance in the base case? If so it can eliminate storing intermediates in Join and Zip
    // fn with_ref(&self, &mut dyn FnMut(&Self::Item));
    fn update_if_necessary(&self);
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

// This has the same logic as From, but avoids needing manual type annotations in some situations.
impl<Provider: SignalProvider + 'static> Signal<Provider> {
    pub fn into_dyn_signal(self) -> Signal<dyn SignalProvider<Item = Provider::Item>> {
        Signal(self.0.clone())
    }
}

impl<Provider: SignalProvider + 'static> From<Signal<Provider>>
    for Signal<dyn SignalProvider<Item = Provider::Item>>
{
    fn from(value: Signal<Provider>) -> Self {
        Self(value.0.clone())
    }
}

// We use the insane autoref-specialization technique to get an "optional" Debug trait on T
pub trait SignalDebug {
    fn dbg(&self) -> Option<&dyn std::fmt::Debug>;
}

impl<T: std::fmt::Debug> SignalDebug for T {
    fn dbg(&self) -> Option<&dyn std::fmt::Debug> {
        Some(self)
    }
}

pub trait SignalNoDebug {
    fn dbg(&self) -> Option<&dyn std::fmt::Debug> {
        None
    }
}

impl<T> SignalNoDebug for &T {}

macro_rules! opt_debug {
    ($e:expr) => {
        (&$e).dbg()
    };
}

// This is absolutely optimizable to ensure const signals do not need to allocate a node
// A sketch of some of the components needed to do so are included
// But I leave actually implementing the type level optimizations to after prototyping
pub struct ConstProvider<T>(T, SignalNodeId);
impl<T> ConstProvider<T> {
    pub fn new(x: T) -> Self {
        Self(x, new_node(NodeColor::Ready))
    }
}

impl<T> std::fmt::Debug for ConstProvider<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("ConstSignal");
        dbg.field("node", &self.1);

        if let Some(v) = opt_debug!(self.0) {
            dbg.field("value", v);
        } else {
            dbg.field("value", &format_args!("{}", std::any::type_name::<T>()));
        }

        dbg.finish()
    }
}

impl<T> SignalProvider for ConstProvider<T> {
    type Item = T;

    fn update_if_necessary(&self) {}

    fn get_node(&self) -> &SignalNodeId {
        &self.1
    }

    fn get_ref(&self) -> DynRef<'_, T> {
        DynRef::Ref(&self.0)
    }
}

pub trait ToSignal<T> {
    type Provider: SignalProvider<Item = T> + ?Sized;
    fn to_signal(self) -> Signal<Self::Provider>;
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
pub fn const_signal<T>(t: T) -> Signal<ConstProvider<T>> {
    Signal(Rc::new(ConstProvider::new(t)))
}

pub type ConstSignal<T> = Signal<ConstProvider<T>>;

impl<T: Default> Default for Signal<ConstProvider<T>> {
    fn default() -> Self {
        Self(Rc::new(ConstProvider::new(T::default())))
    }
}

//TODO: use macros to generate ZipProviders for multiple tuple lengths
//TODO: constant fusion optimization to eliminate unnecessary intermediate storage
pub struct ZipProvider<P1: SignalProvider + ?Sized, P2: SignalProvider + ?Sized> {
    provider1: Rc<P1>,
    provider2: Rc<P2>,
    node: SignalNodeId,
    result: RefCell<Option<(P1::Item, P2::Item)>>,
}

impl<P1: SignalProvider + ?Sized, P2: SignalProvider + ?Sized> std::fmt::Debug
    for ZipProvider<P1, P2>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZipSignal")
            .field("node", &self.node)
            .field("provider1", &self.provider1)
            .field("provider2", &self.provider2)
            .finish()
    }
}

impl<P1: SignalProvider + ?Sized, P2: SignalProvider + ?Sized> SignalProvider
    for ZipProvider<P1, P2>
where
    P1::Item: Clone,
    P2::Item: Clone,
{
    type Item = (P1::Item, P2::Item);

    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update_if_necessary(&self) {
        let updated = {
            let color = self.node.0.borrow().color;
            match color {
                NodeColor::Ready => false,
                _ => {
                    self.provider1.update_if_necessary();
                    self.provider2.update_if_necessary();
                    true
                }
            }
        };
        if updated {
            if self.result.borrow().is_none() {
                let new_color = self.node.0.borrow().color;
                assert!(
                    matches!(new_color, NodeColor::Changed),
                    "color is incorrect, must be Changed but is {new_color:?}"
                );
            } else {
                // assert!(matches!(
                //     self.node.0.borrow().color,
                //     NodeColor::Changed | NodeColor::Ready
                // ));
            }
        }
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Changed => {
                *self.result.borrow_mut() = Some((
                    self.provider1.get_ref().clone(),
                    self.provider2.get_ref().clone(),
                ));
                notify_children_change(&self.node);
            }
            _ => {}
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    fn get_ref(&self) -> DynRef<'_, (P1::Item, P2::Item)> {
        DynRef::Cell(Ref::map(self.result.borrow(), |x| {
            x.as_ref().expect("must be updated before getting value")
        }))
    }
}

pub fn zip<
    T1: Clone,
    T2: Clone,
    P1: SignalProvider<Item = T1> + ?Sized,
    P2: SignalProvider<Item = T2> + ?Sized,
>(
    p1: Signal<P1>,
    p2: Signal<P2>,
) -> Signal<ZipProvider<P1, P2>> {
    let provider1 = p1.0;
    let provider2 = p2.0;
    let node = new_node(NodeColor::Changed);
    add_dependency(provider1.get_node(), node.clone());
    add_dependency(provider2.get_node(), node.clone());
    Signal(Rc::new(ZipProvider {
        provider1,
        provider2,
        node,
        result: RefCell::new(None),
    }))
}

pub struct SignalMapProvider<P: SignalProvider + ?Sized, T2, F: Fn(&P::Item) -> T2> {
    provider: Rc<P>,
    func: F,
    res: RefCell<Option<T2>>,
    node: SignalNodeId,
    eq: fn(&T2, &T2) -> bool,
}

impl<P: SignalProvider + ?Sized, T2, F: Fn(&P::Item) -> T2> std::fmt::Debug
    for SignalMapProvider<P, T2, F>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("MapSignal");
        dbg.field("node", &self.node)
            .field("provider", &self.provider)
            .field("f", &format_args!("{}", std::any::type_name::<F>()));

        if let Some(r) = &*self.res.borrow()
            && let Some(v) = opt_debug!(*r)
        {
            dbg.field("res", v);
        } else {
            dbg.field("res", &format_args!("{}", std::any::type_name::<T2>()));
        }

        dbg.finish()
    }
}
/*
trait QuickEq {
    fn qeq(&self) -> bool;
}

impl<T: Copy + PartialEq> QuickEq for (&T, &T) {
    fn qeq(&self) -> bool {
        PartialEq::eq(self.0, self.1)
    }
}

trait SlowEq {
    fn qeq(&self) -> bool {
        return false;
    }
}

impl<T> SlowEq for &(&T, &T) {}

macro_rules! opt_eq {
    ($e:expr, $rhs:expr) => {
        (&($e, $rhs)).qeq()
    };
}
*/

impl<P: SignalProvider + ?Sized, T2, F: Fn(&P::Item) -> T2> SignalProvider
    for SignalMapProvider<P, T2, F>
{
    type Item = T2;

    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update_if_necessary(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                self.provider.update_if_necessary();
            }
        }
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Changed => {
                let res = (self.func)(&self.provider.get_ref());

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
            _ => {}
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    fn get_ref(&self) -> DynRef<'_, T2> {
        DynRef::Cell(Ref::map(self.res.borrow(), |x| {
            x.as_ref().expect("must be updated before getting value")
        }))
    }
}

pub fn map<T1, T2, F: Fn(&P::Item) -> T2, P: SignalProvider<Item = T1> + ?Sized>(
    f: F,
    p: Signal<P>,
    eq: fn(&T2, &T2) -> bool,
) -> Signal<SignalMapProvider<P, T2, F>> {
    let node = new_node(NodeColor::Changed);
    let provider = p.0;
    add_dependency(provider.get_node(), node.clone());
    Signal(Rc::new(SignalMapProvider {
        provider,
        func: f,
        res: RefCell::new(None),
        node,
        eq,
    }))
}

pub struct SignalMapMutProvider<P: SignalProvider + ?Sized, T2, F: Fn(&P::Item, Option<T2>) -> T2> {
    provider: Rc<P>,
    func: F,
    res: RefCell<Option<T2>>,
    node: SignalNodeId,
}

impl<P: SignalProvider + ?Sized, T2, F: Fn(&P::Item, Option<T2>) -> T2> std::fmt::Debug
    for SignalMapMutProvider<P, T2, F>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("MapMutSignal");
        dbg.field("node", &self.node)
            .field("provider", &self.provider)
            .field("f", &format_args!("{}", std::any::type_name::<F>()));

        if let Some(r) = &*self.res.borrow()
            && let Some(v) = opt_debug!(*r)
        {
            dbg.field("res", v);
        } else {
            dbg.field("res", &format_args!("{}", std::any::type_name::<T2>()));
        }

        dbg.finish()
    }
}

impl<P: SignalProvider + ?Sized, T2, F: Fn(&P::Item, Option<T2>) -> T2> SignalProvider
    for SignalMapMutProvider<P, T2, F>
{
    type Item = T2;
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update_if_necessary(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                self.provider.update_if_necessary();
            }
        }
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Changed => {
                let init = self.res.borrow_mut().take();
                let res = (self.func)(&self.provider.get_ref(), init);

                *self.res.borrow_mut() = Some(res);
                notify_children_change(&self.node);
            }
            _ => {}
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }
    fn get_ref(&self) -> DynRef<'_, T2> {
        DynRef::Cell(Ref::map(self.res.borrow(), |x| {
            x.as_ref().expect("Must be updated before getting value, or there was an attempt to use this value during a map operation.")
        }))
    }
}

pub fn map_mut<T1, T2, F: Fn(&P::Item, Option<T2>) -> T2, P: SignalProvider<Item = T1> + ?Sized>(
    f: F,
    p: Signal<P>,
) -> Signal<SignalMapMutProvider<P, T2, F>> {
    let node = new_node(NodeColor::Changed);
    let provider = p.0;
    add_dependency(provider.get_node(), node.clone());
    Signal(Rc::new(SignalMapMutProvider {
        provider,
        func: f,
        res: RefCell::new(None),
        node,
    }))
}

pub trait SignalMap<Elem> {
    fn map<T: Copy + PartialEq>(
        self,
        f: impl Fn(&Elem) -> T,
    ) -> Signal<impl SignalProvider<Item = T>>;
    fn map_mut<T>(self, f: impl Fn(&Elem, Option<T>) -> T)
    -> Signal<impl SignalProvider<Item = T>>;
    fn map_ex<T>(self, f: impl Fn(&Elem) -> T) -> Signal<impl SignalProvider<Item = T>>;
}

impl<Elem, P: SignalProvider<Item = Elem> + ?Sized> SignalMap<Elem> for Signal<P> {
    fn map<T: Copy + PartialEq>(
        self,
        f: impl Fn(&Elem) -> T,
    ) -> Signal<impl SignalProvider<Item = T>> {
        map(f, self, PartialEq::eq)
    }
    fn map_mut<T>(
        self,
        f: impl Fn(&Elem, Option<T>) -> T,
    ) -> Signal<impl SignalProvider<Item = T>> {
        map_mut(f, self)
    }
    fn map_ex<T>(self, f: impl Fn(&Elem) -> T) -> Signal<impl SignalProvider<Item = T>> {
        map(f, self, always_fail_eq)
    }
}

fn always_fail_eq<T>(_: &T, _: &T) -> bool {
    false
}

pub trait SignalTupleZip: Tuple {
    fn zip<Output>(self) -> Signal<impl SignalProvider<Item = Output>>
    where
        Output: Tuple + Clone,
        Self::TupleList: SignalTupleListZip<Output::TupleList>,
        Output::TupleList: Clone;
}
pub trait SignalTupleListZip<Output: TupleList + Clone> {
    fn zip(self) -> Signal<impl SignalProvider<Item = Output>>;
}

pub fn empty_signal() -> Signal<ConstProvider<()>> {
    Signal(Rc::new(ConstProvider::new(())))
}

impl SignalTupleListZip<()> for () {
    fn zip(self) -> Signal<impl SignalProvider<Item = ()>> {
        empty_signal()
    }
}

impl<HeadT: Clone, Head: SignalProvider<Item = HeadT> + ?Sized, Tail, TailOut: TupleList + Clone>
    SignalTupleListZip<(HeadT, TailOut)> for (Signal<Head>, Tail)
where
    Tail: SignalTupleListZip<TailOut>,
    (HeadT, TailOut): TupleList,
{
    fn zip(self) -> Signal<impl SignalProvider<Item = (HeadT, TailOut)>> {
        zip(self.0, self.1.zip())
    }
}

impl<Input> SignalTupleZip for Input
where
    Input: Tuple,
{
    fn zip<Output>(self) -> Signal<impl SignalProvider<Item = Output>>
    where
        Output: Tuple + Clone,
        Input::TupleList: SignalTupleListZip<Output::TupleList>,
        Output::TupleList: Clone,
    {
        map(
            |xs: &Output::TupleList| xs.clone().into_tuple(),
            self.into_tuple_list().zip(),
            always_fail_eq,
        )
    }
}

pub struct SignalJoinProvider<
    T: Clone,
    P1: SignalProvider<Item = T> + ?Sized,
    P2: SignalProvider<Item = Signal<P1>> + ?Sized,
> {
    provider: Rc<P2>,
    innerprovider: RefCell<Option<Rc<P1>>>,
    node: SignalNodeId,
    res: RefCell<Option<T>>,
    phantom: PhantomData<P1>,
}

impl<
    T: Clone,
    P1: SignalProvider<Item = T> + ?Sized,
    P2: SignalProvider<Item = Signal<P1>> + ?Sized,
> std::fmt::Debug for SignalJoinProvider<T, P1, P2>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JoinSignal")
            .field("inner", &self.innerprovider)
            .field("node", &self.node)
            .finish()
    }
}

impl<
    T: Clone,
    P1: SignalProvider<Item = T> + ?Sized,
    P2: SignalProvider<Item = Signal<P1>> + ?Sized,
> SignalProvider for SignalJoinProvider<T, P1, P2>
{
    type Item = T;
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update_if_necessary(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                self.provider.update_if_necessary();
                let color = self.node.0.borrow().color;
                match color {
                    NodeColor::Changed => {
                        let innersig = self.provider.get_ref().0.clone();
                        let old_innerprovider = self.innerprovider.borrow().clone();
                        match old_innerprovider {
                            Some(old_innerprovider_actual) => {
                                if old_innerprovider_actual.get_node() == innersig.get_node() {
                                    remove_dependency(
                                        &old_innerprovider_actual.get_node(),
                                        &self.node,
                                    );
                                    *self.innerprovider.borrow_mut() = Some(innersig.clone());
                                    add_dependency(&innersig.get_node(), self.node.clone());
                                }
                            }
                            None => {
                                *self.innerprovider.borrow_mut() = Some(innersig.clone());
                                add_dependency(&innersig.get_node(), self.node.clone());
                            }
                        }
                        // TODO: Setting this to Check causes a problem with Res not being updated below. Does leaving
                        // the color as Changed cause a performance issue?
                        //self.node.0.borrow_mut().color = NodeColor::Check;
                    }
                    _ => {}
                }
                let innersig = self.provider.get_ref().0.clone();
                innersig.update_if_necessary();
                let color = self.node.0.borrow().color;
                match color {
                    NodeColor::Changed => {
                        *self.res.borrow_mut() = Some(innersig.get_ref().clone());
                        notify_children_change(&self.node);
                    }
                    _ => {}
                }
                self.node.0.borrow_mut().color = NodeColor::Ready;
            }
        }
    }

    fn get_ref(&self) -> DynRef<'_, T> {
        // Can't return the underlying reference here because it would require storing the intermediate DynRef
        DynRef::Cell(Ref::map(self.res.borrow(), |x| {
            x.as_ref().expect("Must be updated before getting value, or there was an attempt to use this value during a map operation.")
        }))
    }
}

pub fn join<
    T: Clone,
    P1: SignalProvider<Item = T> + ?Sized,
    P2: SignalProvider<Item = Signal<P1>> + ?Sized,
>(
    p: Signal<P2>,
) -> Signal<SignalJoinProvider<T, P1, P2>> {
    Signal(Rc::new(SignalJoinProvider {
        provider: p.0,
        innerprovider: RefCell::new(None),
        node: new_node(NodeColor::Changed),
        res: RefCell::new(None),
        phantom: PhantomData,
    }))
}

pub struct MutableSignalProvider<T> {
    node: SignalNodeId,
    val: RefCell<T>,
}
impl<T> MutableSignalProvider<T> {
    pub fn new(v: T) -> Self {
        Self {
            node: new_node(NodeColor::Changed),
            val: RefCell::new(v),
        }
    }
}

impl<T> std::fmt::Debug for MutableSignalProvider<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("MutableSignal");
        dbg.field("node", &self.node);

        if let Some(v) = opt_debug!(self.val) {
            dbg.field("value", v);
        } else {
            dbg.field("value", &format_args!("{}", std::any::type_name::<T>()));
        }

        dbg.finish()
    }
}

impl<T> SignalProvider for MutableSignalProvider<T> {
    type Item = T;
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update_if_necessary(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Changed => {
                notify_children_change(&self.node);
                self.node.0.borrow_mut().color = NodeColor::Ready;
            }
            NodeColor::Check => {
                panic!("Should never be unsure about whether or not a mutable signal changed");
            }
            NodeColor::Ready => {}
        }
    }
    fn get_ref(&self) -> DynRef<'_, T> {
        DynRef::Cell(self.val.borrow())
    }
}

pub struct SignalRefMut<'a, T>(std::cell::RefMut<'a, T>, Rc<MutableSignalProvider<T>>);

impl<'a, T> Drop for SignalRefMut<'a, T> {
    fn drop(&mut self) {
        notify_change(&self.1.node);
    }
}

impl<T: Default> Default for Signal<MutableSignalProvider<T>> {
    fn default() -> Self {
        Self(Rc::new(MutableSignalProvider::new(T::default())))
    }
}

impl<Provider: SignalProvider + ?Sized> core::fmt::Debug for Signal<Provider>
where
    Provider::Item: core::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Signal")
            .field("node", self.0.get_node())
            .field("value", &*self.0.get_ref())
            .finish()
    }
}

impl<T> Signal<MutableSignalProvider<T>> {
    pub fn new(x: T) -> Self {
        Self(Rc::new(MutableSignalProvider::new(x)))
    }
    // Instead of using `borrow` most things should use map() or sample()
    /*pub fn borrow(&self) -> Ref<'_, T> {
        self.0.update_if_necessary();
        self.0.val.borrow()
    }*/
    pub fn borrow_mut(&mut self) -> SignalRefMut<'_, T> {
        self.0.update_if_necessary();
        SignalRefMut(self.0.val.borrow_mut(), self.0.clone())
    }
    /*pub fn try_borrow(&self) -> Result<Ref<'_, T>, std::cell::BorrowError> {
        self.0.update_if_necessary();
        self.0.val.try_borrow()
    }*/
    pub fn try_borrow_mut(&mut self) -> Result<SignalRefMut<'_, T>, std::cell::BorrowMutError> {
        self.0.update_if_necessary();
        self.0
            .val
            .try_borrow_mut()
            .map(|x| SignalRefMut(x, self.0.clone()))
    }
    pub fn replace(&mut self, x: T) {
        self.0.val.replace(x);
        notify_change(&self.0.node);
    }
    pub fn replace_with(&mut self, f: impl FnOnce(&mut T) -> T) {
        self.0.val.replace_with(f);
        notify_change(&self.0.node);
    }
    pub fn set_with(&mut self, f: impl FnOnce(&mut T)) {
        f(&mut self.0.val.borrow_mut());
        notify_change(&self.0.node);
    }
    pub fn swap(&mut self, rhs: &mut Self) {
        self.0.val.swap(&rhs.0.val);
    }
}

pub type DynSignal<T> = Signal<dyn SignalProvider<Item = T>>; // Removing the stuff to handle smarter const folding to write less code
pub type MutableSignal<T> = Signal<MutableSignalProvider<T>>;

// A mechanism for declaring that something that isn't a signal cares about checking when a signal changes
pub struct Sampler<Provider: SignalProvider + ?Sized> {
    node: SignalNodeId,
    provider: Rc<Provider>,
}
impl<Provider: SignalProvider + ?Sized> Sampler<Provider> {
    pub fn new(signal: Signal<Provider>) -> Self {
        let node = new_node(NodeColor::Changed);
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
                self.provider.update_if_necessary();
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
                self.provider.update_if_necessary();
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
        self.provider.update_if_necessary();
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
static DYNAMIC_DEPS: std::cell::RefCell<Option<SmolSet<[SignalNodeId; 4]>>> = std::cell::RefCell::new(None);
}
pub struct DynamicSignalProvider<T, F: Fn() -> T> {
    node: SignalNodeId,
    lastdeps: RefCell<SmolSet<[SignalNodeId; 4]>>,
    f: F,
    val: RefCell<Option<T>>,
}

impl<T, F: Fn() -> T> std::fmt::Debug for DynamicSignalProvider<T, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("DynamicSignal");
        dbg.field("f", &format_args!("{}", std::any::type_name::<F>()))
            .field("node", &self.node);

        if let Some(v) = opt_debug!(self.val) {
            dbg.field("value", v);
        } else {
            dbg.field("value", &format_args!("{}", std::any::type_name::<T>()));
        }

        dbg.finish()
    }
}

impl<T, F: Fn() -> T> SignalProvider for DynamicSignalProvider<T, F> {
    type Item = T;
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update_if_necessary(&self) {
        //TODO: Make this smarter so it only runs updates on the minimally required set and avoid unneeded recomputation
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                let push_deps = DYNAMIC_DEPS.replace(Some(SmolSet::new()));

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

    fn get_ref(&self) -> DynRef<'_, T> {
        DynRef::Cell(Ref::map(self.val.borrow(), |x| {
            x.as_ref().expect("must be updated before getting value")
        }))
    }
}

pub fn new_dynamic_signal<T, F: Fn() -> T>(f: F) -> Signal<DynamicSignalProvider<T, F>> {
    Signal(Rc::new(DynamicSignalProvider {
        node: new_node(NodeColor::Changed),
        lastdeps: RefCell::new(SmolSet::new()),
        f: f,
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
                self.0.update_if_necessary();
                Some(self.0.get_ref().clone())
            }
        })
    }
}

//TODO: Signal idiom macro
//TODO: more convenience combinators
//TODO: UNIT TESTS

//FIXME: Lifetimes needed to establish invariant that signals don't outlive the functions needed to compute them (and signals don't outlive their inputs just in case signal contents have limited lifetimes)
pub fn zip_pair<
    T1: Clone,
    T2: Clone,
    R: Clone,
    F: Fn(P1::Item, P2::Item) -> R,
    P1: SignalProvider<Item = T1> + ?Sized,
    P2: SignalProvider<Item = T2> + ?Sized,
>(
    p1: Signal<P1>,
    p2: Signal<P2>,
    f: F,
) -> Signal<impl SignalProvider<Item = R>> {
    map(
        move |arg: &(P1::Item, P2::Item)| f(arg.0.clone(), arg.1.clone()),
        zip(p1, p2),
        always_fail_eq,
    )
}

pub fn sample<T, P: SignalProvider<Item = T> + ?Sized>(signal: &Signal<P>) -> DynRef<'_, T> {
    signal.0.update_if_necessary();
    signal.0.get_ref()
}

pub fn sample_val<T: Clone, P: SignalProvider<Item = T> + ?Sized>(signal: Signal<P>) -> T {
    let p = signal;
    p.0.update_if_necessary();
    p.0.get_ref().clone()
}

#[test]
fn reactive_fold() {
    let mut vals = Iterator::map(1..=100, MutableSignal::new).collect::<Vec<_>>();
    let sum = vals.iter().fold(0.to_signal().into_dyn_signal(), |a, b| {
        zip_pair(a, b.clone(), |x: i32, y: i32| x + y).into_dyn_signal()
    });
    let modifications = vec![(1, 1, 5049), (1, 2, 5050), (5, -1, 5043)];
    for (idx, val, expectation) in modifications {
        vals[idx].replace(val);
        assert!(*sample(&sum) == expectation);
    }
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

impl<Output: std::fmt::Debug> std::fmt::Debug for AnimationOutput<Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    Anim: Animator<T>,
    Source: SignalProvider<Item = T> + ?Sized,
    Time: SignalProvider<Item = Microseconds>,
> std::fmt::Debug for AnimProvider<T, Anim, Source, Time>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("AnimSignal");
        dbg.field("node", &self.node);

        if let Some(r) = &*self.val.borrow()
            && let Some(v) = opt_debug!(*r)
        {
            dbg.field("value", v);
        } else {
            dbg.field(
                "value",
                &format_args!("{}", std::any::type_name::<Anim::Output>()),
            );
        }

        if let Some((a, b, r)) = &*self.state.borrow()
            && let Some(v) = opt_debug!(*r)
        {
            dbg.field("state", &format_args!("{}, {}, {:?}", a, b, v));
        } else if let Some((a, b, _)) = &*self.state.borrow() {
            dbg.field(
                "state",
                &format_args!("{}, {}, {}", a, b, std::any::type_name::<Anim::State>()),
            );
        }

        if let Some(v) = opt_debug!(self.anim) {
            dbg.field("anim", v);
        } else {
            dbg.field("anim", &format_args!("{}", std::any::type_name::<Anim>()));
        }

        dbg.field("time", &self.time)
            .field("input", &self.input)
            .finish()
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
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update_if_necessary(&self) {
        //eprintln!("animprovider state {:?}", *self.state.borrow());
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                self.input.update_if_necessary();
                match color {
                    NodeColor::Changed => {
                        let inval = self.input.get_ref().clone(); // FIXME: Can this be made a reference?
                        self.time.update_if_necessary();
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
                        self.time.update_if_necessary();
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
    let node = new_node(NodeColor::Changed);
    let input = p.0;
    add_dependency(input.get_node(), node.clone());
    Signal(Rc::new(AnimProvider {
        node,
        val: RefCell::new(None),
        state: RefCell::new(None),
        anim: anim,
        time: time.0,
        input,
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

pub struct SignalOpProvider<
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
}

impl<
    Output,
    P1: SignalProvider + ?Sized,
    P2: SignalProvider + ?Sized,
    OP: SignalOp<P1::Item, P2::Item, Output>,
> std::fmt::Debug for SignalOpProvider<P1, P2, Output, OP>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("SignalOpProvider");
        dbg.field("node", &self.node)
            .field("provider1", &self.provider1)
            .field("provider2", &self.provider2);

        if let Some(r) = &*self.res.borrow()
            && let Some(v) = opt_debug!(*r)
        {
            dbg.field("res", v);
        } else {
            dbg.field("res", &format_args!("{}", std::any::type_name::<Output>()));
        }

        dbg.finish()
    }
}

impl<
    Output,
    P1: SignalProvider + ?Sized,
    P2: SignalProvider + ?Sized,
    OP: SignalOp<P1::Item, P2::Item, Output>,
> SignalProvider for SignalOpProvider<P1, P2, Output, OP>
where
    P1::Item: Clone,
    P2::Item: Clone,
{
    type Item = Output;
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update_if_necessary(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                self.provider1.update_if_necessary();
                self.provider2.update_if_necessary();
            }
        }
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Changed => {
                let res = OP::apply(
                    self.provider1.get_ref().clone(),
                    self.provider2.get_ref().clone(),
                );

                *self.res.borrow_mut() = Some(res);
                notify_children_change(&self.node);
            }
            _ => {}
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    fn get_ref(&self) -> DynRef<'_, Output> {
        DynRef::Cell(Ref::map(self.res.borrow(), |x| {
            x.as_ref().expect("must be updated before getting value")
        }))
    }
}

fn operate<
    P1: SignalProvider + ?Sized,
    P2: SignalProvider + ?Sized,
    Output,
    OP: SignalOp<P1::Item, P2::Item, Output>,
>(
    lhs: Signal<P1>,
    rhs: Signal<P2>,
) -> Signal<SignalOpProvider<P1, P2, Output, OP>>
where
    P1::Item: Clone,
    P2::Item: Clone,
{
    let node = new_node(NodeColor::Changed);
    let provider1 = lhs.0;
    let provider2 = rhs.0;
    add_dependency(provider1.get_node(), node.clone());
    add_dependency(provider2.get_node(), node.clone());
    Signal(Rc::new(SignalOpProvider {
        provider1,
        provider2,
        res: RefCell::new(None),
        node,
        phantom: PhantomData,
    }))
}

macro_rules! gen_binop_impl {
    ($t:tt, $marker:path, $op:tt) => {
        impl<AP: SignalProvider + ?Sized, BP: SignalProvider + ?Sized> std::ops::$t<Signal<BP>>
            for Signal<AP>
        where
            AP::Item: std::ops::$t<BP::Item> + Clone,
            BP::Item: Clone,
            <AP::Item as std::ops::$t<BP::Item>>::Output: Clone,
        {
            type Output = Signal<
                SignalOpProvider<AP, BP, <AP::Item as std::ops::$t<BP::Item>>::Output, $marker>,
            >;

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
    use super::{Signal, SignalProvider, marker::*, operate};

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
        AP::Item: Clone + Ord,
    {
        operate::<AP, AP, AP::Item, MinOp>(a, b)
    }

    pub fn max<AP: SignalProvider + ?Sized>(
        a: Signal<AP>,
        b: Signal<AP>,
    ) -> Signal<impl SignalProvider<Item = AP::Item>>
    where
        AP::Item: Clone + Ord,
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

pub fn cond<T: Clone>(
    c: Signal<impl SignalProvider<Item = bool> + ?Sized>,
    a: DynSignal<T>,
    b: DynSignal<T>,
) -> Signal<impl SignalProvider<Item = T>> {
    join(c.map_ex(move |cc| if *cc { a.clone() } else { b.clone() }))
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

#[test]
fn add() {
    let a = 1.to_signal();
    let b = 2.to_signal();
    let input_signal = a + b;
    let problem = *sample(&input_signal);
    assert_eq!(problem, 3);
}

pub struct SignalVecMapProvider<
    T1,
    T2,
    Key: Eq + Hash,
    F: Fn(&T1) -> T2,
    Ex: Fn(&T1) -> Key,
    P: SignalProvider<Item = GenericVector<T1, Ptr, CHUNK_SIZE>> + ?Sized,
    Ptr: imbl::shared_ptr::SharedPointerKind,
    const CHUNK_SIZE: usize,
> {
    provider: Rc<P>,
    map: RefCell<imbl::vector::PersistentMap<T1, T2, Key, Ptr, F, Ex, CHUNK_SIZE>>,
    res: RefCell<GenericVector<T2, Ptr, CHUNK_SIZE>>,
    node: SignalNodeId,
}

impl<
    T1: Clone + 'static,
    T2: Clone + 'static,
    Key: Eq + Hash + 'static,
    F: (Fn(&T1) -> T2) + 'static,
    Ex: (Fn(&T1) -> Key) + 'static,
    P: SignalProvider<Item = GenericVector<T1, Ptr, CHUNK_SIZE>> + ?Sized + 'static,
    Ptr: imbl::shared_ptr::SharedPointerKind + 'static,
    const CHUNK_SIZE: usize,
> std::fmt::Debug for SignalVecMapProvider<T1, T2, Key, F, Ex, P, Ptr, CHUNK_SIZE>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignalVecMapProvider")
            .field("node", &self.node)
            .field("provider", &self.provider)
            .finish()
    }
}

fn assert_static<T: 'static>() {}

impl<
    T1: Clone + 'static,
    T2: Clone + 'static,
    Key: Eq + Hash + 'static,
    F: (Fn(&T1) -> T2) + 'static,
    Ex: (Fn(&T1) -> Key) + 'static,
    P: SignalProvider<Item = GenericVector<T1, Ptr, CHUNK_SIZE>> + ?Sized + 'static,
    Ptr: imbl::shared_ptr::SharedPointerKind + 'static,
    const CHUNK_SIZE: usize,
> SignalProvider for SignalVecMapProvider<T1, T2, Key, F, Ex, P, Ptr, CHUNK_SIZE>
{
    type Item = GenericVector<T2, Ptr, CHUNK_SIZE>;
    fn get_node(&self) -> &SignalNodeId {
        assert_static::<Self>();
        &self.node
    }

    fn update_if_necessary(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                self.provider.update_if_necessary();
            }
        }
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Changed => {
                let res = self.map.borrow_mut().map(&self.provider.get_ref());
                let changed = !self.res.borrow().ptr_eq(&res);
                *self.res.borrow_mut() = res;
                if changed {
                    notify_children_change(&self.node);
                }
            }
            _ => {}
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    fn get_ref(&self) -> DynRef<'_, GenericVector<T2, Ptr, CHUNK_SIZE>> {
        DynRef::Cell(self.res.borrow())
    }
}

pub fn map_vec<
    T1: Clone + 'static,
    T2: Clone + 'static,
    Key: Eq + Hash + 'static,
    F: (Fn(&T1) -> T2) + 'static,
    Ex: (Fn(&T1) -> Key) + 'static,
    P: SignalProvider<Item = GenericVector<T1, Ptr, CHUNK_SIZE>> + ?Sized + 'static,
    Ptr: imbl::shared_ptr::SharedPointerKind + 'static,
    const CHUNK_SIZE: usize,
>(
    f: F,
    ex: Ex,
    p: Signal<P>,
) -> Signal<SignalVecMapProvider<T1, T2, Key, F, Ex, P, Ptr, CHUNK_SIZE>> {
    let node = new_node(NodeColor::Changed);
    let provider = p.0;
    add_dependency(provider.get_node(), node.clone());
    Signal(Rc::new(SignalVecMapProvider {
        provider,
        map: RefCell::new(imbl::vector::PersistentMap::new(f, ex)),
        res: Default::default(),
        node,
    }))
}

#[test]
fn test_reactive_map_vec() {
    let v = const_signal(imbl::vector![1, 2, 3, 4]);

    let result = map_vec(|x| *x * *x, |x| *x, v.clone());

    for i in sample(&result).iter() {
        println!("{i}");
    }

    let rv = const_signal(imbl::vector![
        Rc::new(1),
        Rc::new(2),
        Rc::new(3),
        Rc::new(4)
    ]);

    let result = map_vec(
        |x| *x.as_ref() * *x.as_ref(),
        |x| Identity(x.clone()),
        rv.clone(),
    );

    for i in sample(&result).iter() {
        println!("{i}");
    }
}

pub struct SignalVecFoldProvider<
    T,
    F: FnMut(T, T) -> T,
    P: SignalProvider<Item = GenericVector<T, Ptr, CHUNK_SIZE>> + ?Sized,
    Ptr: imbl::shared_ptr::SharedPointerKind,
    const CHUNK_SIZE: usize,
> {
    provider: Rc<P>,
    fold: RefCell<imbl::vector::PersistentFold<T, F, Ptr, CHUNK_SIZE>>,
    res: RefCell<T>,
    node: SignalNodeId,
}

impl<
    T: Clone,
    F: FnMut(T, T) -> T,
    P: SignalProvider<Item = GenericVector<T, Ptr, CHUNK_SIZE>> + ?Sized,
    Ptr: imbl::shared_ptr::SharedPointerKind,
    const CHUNK_SIZE: usize,
> std::fmt::Debug for SignalVecFoldProvider<T, F, P, Ptr, CHUNK_SIZE>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignalVecFoldProvider")
            .field("node", &self.node)
            .field("provider", &self.provider)
            .finish()
    }
}

impl<
    T: Clone,
    F: FnMut(T, T) -> T,
    P: SignalProvider<Item = GenericVector<T, Ptr, CHUNK_SIZE>> + ?Sized,
    Ptr: imbl::shared_ptr::SharedPointerKind,
    const CHUNK_SIZE: usize,
> SignalProvider for SignalVecFoldProvider<T, F, P, Ptr, CHUNK_SIZE>
{
    type Item = T;
    fn get_node(&self) -> &SignalNodeId {
        &self.node
    }

    fn update_if_necessary(&self) {
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Ready => {}
            _ => {
                self.provider.update_if_necessary();
            }
        }
        let color = self.node.0.borrow().color;
        match color {
            NodeColor::Changed => {
                let res = self.fold.borrow_mut().fold(&self.provider.get_ref());
                *self.res.borrow_mut() = res;
                notify_children_change(&self.node);
            }
            _ => {}
        }
        self.node.0.borrow_mut().color = NodeColor::Ready;
    }

    fn get_ref(&self) -> DynRef<'_, T> {
        DynRef::Cell(self.res.borrow())
    }
}

pub fn fold_vec<
    T: Clone,
    F: FnMut(T, T) -> T,
    P: SignalProvider<Item = GenericVector<T, Ptr, CHUNK_SIZE>> + ?Sized,
    Ptr: imbl::shared_ptr::SharedPointerKind,
    const CHUNK_SIZE: usize,
>(
    f: F,
    p: Signal<P>,
    z: T,
) -> Signal<SignalVecFoldProvider<T, F, P, Ptr, CHUNK_SIZE>> {
    let node = new_node(NodeColor::Changed);
    let provider = p.0;
    add_dependency(provider.get_node(), node.clone());
    Signal(Rc::new(SignalVecFoldProvider {
        provider,
        fold: RefCell::new(imbl::vector::PersistentFold::new(f, z.clone())),
        res: RefCell::new(z),
        node,
    }))
}
