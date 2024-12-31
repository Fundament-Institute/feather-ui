// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2024 Fundament Software SPC <https://fundament.software>

use std::{default::Default, marker::PhantomData};

use derive_where::derive_where;

pub trait FnPersist<Args, Output> {
    type Store: Clone;

    fn init(&self) -> Self::Store;
    fn call(&self, store: Self::Store, args: &Args) -> (Self::Store, Output);
}

impl<Args, Output, T: Fn(&Args) -> Output> FnPersist<Args, Output> for T {
    type Store = ();

    fn init(&self) -> Self::Store {}
    fn call(&self, _: Self::Store, args: &Args) -> (Self::Store, Output) {
        ((), (self)(args))
    }
}

pub trait FnPersist2<Arg1, Arg2, Output> {
    type Store: Clone;

    fn init(&self) -> Self::Store;
    fn call(&self, store: Self::Store, arg1: &Arg1, arg2: &Arg2) -> (Self::Store, Output);
}

impl<Arg1, Arg2, Output, T: Fn(&Arg1, &Arg2) -> Output> FnPersist2<Arg1, Arg2, Output> for T {
    type Store = ();

    fn init(&self) -> Self::Store {}
    fn call(&self, _: Self::Store, arg1: &Arg1, arg2: &Arg2) -> (Self::Store, Output) {
        ((), (self)(arg1, arg2))
    }
}

pub trait MapPersist<T, U> {
    type C<A>;

    fn map<F: FnPersist<T, U>>(f: F) -> impl FnPersist<Self::C<T>, Self::C<U>>;
}

pub trait FoldPersist<T, U> {
    type C<A>;

    fn fold<F: FnPersist2<U, T, U>>(f: F) -> impl FnPersist2<U, Self::C<T>, U>;
}

/*pub fn vector_map<T, U: Clone, F: FnPersist<T, U>>(
    cache: (im::Vector<T>, im::Vector<F::Store>, im::Vector<U>),
    input: im::Vector<T>,
    f: F,
) -> (im::Vector<T>, im::Vector<F::Store>, im::Vector<U>)
where
    F::Store: Clone,
{
    let internal = cache.1.clone();
    let output = cache.2.clone();
    // Get the difference between the items passed in and the cache of what we passed in last
    for item in cache.0.diff(&input) {
        match item {
            DoesNotExist::Insert(i, x) => {
                let (store, result) = f.call(Default::default(), *x);
                internal.insert(i, store);
                output.insert(i, result);
            }
            DoesNotExist::Update(i, old, new) => {
                let (store, result) = f.call(*cache.1.get(i).unwrap(), *new);
                internal.set(i, store);
                output.set(i, result);
            }
            DoesNotExist::Remove(i, _) => {
                internal.remove(i);
                output.remove(i);
            }
        }
    }

    (input, internal, output)
}

// Special case where the applying function has no state and can therefore be a lambda
pub fn vector_map_lamba<T, U: Clone, F: Fn(T) -> U>(
    cache: (im::Vector<T>, im::Vector<U>),
    input: im::Vector<T>,
    f: F,
) -> (im::Vector<T>, im::Vector<U>)
where
    F::Store: Clone,
{
    let output = cache.2.clone();
    // Get the difference between the items passed in and the cache of what we passed in last
    for item in cache.0.diff(&input) {
        match item {
            DoesNotExist::Insert(i, x) => {
                let (store, result) = f.call(Default::default(), *x);
                output.insert(i, result);
            }
            DoesNotExist::Update(i, old, new) => {
                let (store, result) = f.call(*cache.1.get(i).unwrap(), *new);
                output.set(i, result);
            }
            DoesNotExist::Remove(i, _) => {
                output.remove(i);
            }
        }
    }

    (input, internal, output)
}

pub fn vector_fold<T, U: Clone, F: FnPersist<(T, T), U>>(
    cache: (im::Vector<T>, im::Vector<F::Store>, U),
    input: im::Vector<T>,
    f: F,
    init: T,
) -> (im::Vector<T>, im::Vector<F::Store>, U)
where
    F::Store: Clone,
{
    // From our diff, we figure out the start point, and then fold from there
}*/

#[derive_where(Clone, Default)]
pub struct ConcatStore<T: Clone>(im::Vector<T>, im::Vector<T>, im::Vector<T>);
pub struct Concat {}

impl<T: Clone> FnPersist<(im::Vector<T>, im::Vector<T>), im::Vector<T>> for Concat {
    type Store = ConcatStore<T>;

    fn init(&self) -> Self::Store {
        Default::default()
    }
    fn call(
        &self,
        mut store: Self::Store,
        args: &(im::Vector<T>, im::Vector<T>),
    ) -> (Self::Store, im::Vector<T>) {
        // TODO: we need pointer only vector equality checks
        //if store.0 != args.0 || store.1 != args.1 {
        store.0 = args.0.clone();
        store.1 = args.1.clone();
        store.2 = args.0.clone();
        store.2.append(store.1.clone());
        //}
        let r = store.2.clone();
        (store, r)
    }
}

#[derive_where(Clone, Default)]
pub struct OrdSetMapStore<T: Ord + Clone, U: Ord + Clone, F: FnPersist<T, U>> {
    arg: im::OrdSet<T>,
    result: im::OrdSet<U>,
    store: im::OrdMap<T, F::Store>,
}

pub struct OrdSetMap<T, U, F: FnPersist<T, U>> {
    f: F,
    phantom_t: PhantomData<T>,
    phantom_u: PhantomData<U>,
}

impl<T: Ord + Clone, U: Ord + Clone, F: FnPersist<T, U>> OrdSetMap<T, U, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            phantom_t: PhantomData,
            phantom_u: PhantomData,
        }
    }
}

impl<T: Ord + Clone, U: Ord + Clone, F: FnPersist<T, U>> From<F> for OrdSetMap<T, U, F> {
    fn from(f: F) -> Self {
        Self::new(f)
    }
}

impl<T: Ord + Clone, U: Ord + Clone, F: FnPersist<T, U>> FnPersist<im::OrdSet<T>, im::OrdSet<U>>
    for OrdSetMap<T, U, F>
{
    type Store = OrdSetMapStore<T, U, F>;

    fn init(&self) -> Self::Store {
        Default::default()
    }
    fn call(&self, cache: Self::Store, input: &im::OrdSet<T>) -> (Self::Store, im::OrdSet<U>) {
        let mut internal = cache.store.clone();
        let mut output = cache.result.clone();
        // Get the difference between the items passed in and the cache of what we passed in last
        for item in cache.arg.diff(input) {
            match item {
                im::ordset::DiffItem::Add(x) => {
                    let (store, result) = self.f.call(self.f.init(), x);
                    internal.insert(x.clone(), store);
                    output.insert(result);
                }
                im::ordset::DiffItem::Update { old, new } => {
                    let (prevstore, prev) = self.f.call(internal.remove(old).unwrap(), old);
                    output.remove(&prev);
                    let (store, result) = self.f.call(prevstore, new);
                    internal.insert(new.clone(), store);
                    output.insert(result);
                }
                im::ordset::DiffItem::Remove(x) => {
                    let (_, prev) = self.f.call(internal.remove(x).unwrap(), x);
                    output.remove(&prev);
                }
            }
        }

        (
            Self::Store {
                arg: input.clone(),
                store: internal,
                result: output.clone(),
            },
            output,
        )
    }
}

#[allow(refining_impl_trait)]
impl<T: Ord + Clone, U: Ord + Clone> MapPersist<T, U> for im::OrdSet<T> {
    type C<A> = im::OrdSet<A>;
    fn map<F: FnPersist<T, U>>(f: F) -> OrdSetMap<T, U, F> {
        f.into()
    }
}

#[derive_where(Clone, Default)]
pub struct OrdMapMapStore<
    K: Ord + std::cmp::PartialEq + Clone,
    V: std::cmp::PartialEq,
    U: Ord + Clone,
    F: FnPersist<V, U>,
> {
    arg: im::OrdMap<K, V>,
    result: im::OrdMap<K, U>,
    store: im::OrdMap<K, F::Store>,
}

pub struct OrdMapMap<V, U, F: FnPersist<V, U>> {
    f: F,
    phantom_v: PhantomData<V>,
    phantom_u: PhantomData<U>,
}

impl<V, U, F: FnPersist<V, U>> OrdMapMap<V, U, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            phantom_v: PhantomData,
            phantom_u: PhantomData,
        }
    }
}

impl<V, U, F: FnPersist<V, U>> From<F> for OrdMapMap<V, U, F> {
    fn from(f: F) -> Self {
        Self::new(f)
    }
}

impl<
        K: Ord + std::cmp::PartialEq + Clone,
        V: std::cmp::PartialEq,
        U: Ord + Clone,
        F: FnPersist<V, U>,
    > FnPersist<im::OrdMap<K, V>, im::OrdMap<K, U>> for OrdMapMap<V, U, F>
where
    F::Store: Clone,
{
    type Store = OrdMapMapStore<K, V, U, F>;

    fn init(&self) -> Self::Store {
        Default::default()
    }
    fn call(
        &self,
        cache: Self::Store,
        input: &im::OrdMap<K, V>,
    ) -> (Self::Store, im::OrdMap<K, U>) {
        let mut internal = cache.store.clone();
        let mut output = cache.result.clone();
        // Get the difference between the items passed in and the cache of what we passed in last
        for item in cache.arg.diff(input) {
            match item {
                im::ordmap::DiffItem::Add(x, v) => {
                    let (store, result) = self.f.call(self.f.init(), v);
                    internal.insert(x.clone(), store);
                    output.insert(x.clone(), result);
                }
                im::ordmap::DiffItem::Remove(x, _) => {
                    output.remove(x);
                }
                im::ordmap::DiffItem::Update { old, new } => {
                    let (store, result) = self.f.call(internal.get(old.0).unwrap().clone(), new.1);
                    internal.insert(new.0.clone(), store);
                    output.insert(new.0.clone(), result);
                }
            }
        }

        (
            Self::Store {
                arg: input.clone(),
                result: output.clone(),
                store: internal,
            },
            output,
        )
    }
}

#[allow(refining_impl_trait)]
impl<K: Ord + std::cmp::PartialEq + Clone, V: std::cmp::PartialEq, U: Ord + Clone> MapPersist<V, U>
    for im::OrdMap<K, V>
{
    type C<A> = im::OrdMap<K, A>;

    fn map<F: FnPersist<V, U>>(f: F) -> OrdMapMap<V, U, F> {
        f.into()
    }
}

#[allow(dead_code)]
#[derive_where(Clone, Default)]
pub struct VectorMapStore<V: Clone, U: Clone, F: FnPersist<V, U>> {
    arg: im::Vector<V>,
    result: im::Vector<U>,
    store: im::Vector<F::Store>,
}

pub struct VectorMap<V, U, F: FnPersist<V, U>> {
    f: F,
    phantom_v: PhantomData<V>,
    phantom_u: PhantomData<U>,
}

impl<V: Clone, U: Clone, F: FnPersist<V, U>> VectorMap<V, U, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            phantom_v: PhantomData,
            phantom_u: PhantomData,
        }
    }
}

impl<V: Clone, U: Clone, F: FnPersist<V, U>> From<F> for VectorMap<V, U, F> {
    fn from(f: F) -> Self {
        Self::new(f)
    }
}

impl<V: Clone, U: Clone, F: FnPersist<V, U>> FnPersist<im::Vector<V>, im::Vector<U>>
    for VectorMap<V, U, F>
{
    type Store = VectorMapStore<V, U, F>;

    fn init(&self) -> Self::Store {
        Default::default()
    }
    fn call(&self, mut store: Self::Store, args: &im::Vector<V>) -> (Self::Store, im::Vector<U>) {
        // TODO: We can't implement this properly because tracking the storage requires access to im::Vector internals, and we can't even compare the two vectors either
        //if store.arg != *args {
        store.result.clear();
        for v in args.iter() {
            let (_, item) = self.f.call(self.f.init(), v);
            store.result.push_back(item);
        }
        //}
        let result = store.result.clone();
        (store, result)
    }
}

#[allow(refining_impl_trait)]
impl<V: Clone, U: Clone> MapPersist<V, U> for im::Vector<V> {
    type C<A> = im::Vector<A>;

    fn map<F: FnPersist<V, U>>(f: F) -> VectorMap<V, U, F> {
        f.into()
    }
}

#[allow(dead_code)]
#[derive_where(Clone)]
pub struct VectorFoldStore<T: Clone, U: Clone, Store: Clone> {
    arg: im::Vector<T>,
    result: Option<U>,
    store: im::Vector<Store>,
}

pub struct VectorFold<T, U, F> {
    f: F,
    phantom_t: PhantomData<T>,
    phantom_u: PhantomData<U>,
}

impl<T: Clone, U: Clone, F: FnPersist2<U, T, U>> VectorFold<T, U, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            phantom_t: PhantomData,
            phantom_u: PhantomData,
        }
    }
}

impl<T: Clone, U: Clone, F: FnPersist2<U, T, U>> FnPersist2<U, im::Vector<T>, U>
    for VectorFold<T, U, F>
{
    type Store = VectorFoldStore<T, U, F::Store>;

    fn init(&self) -> Self::Store {
        Self::Store {
            arg: Default::default(),
            result: None,
            store: Default::default(),
        }
    }

    fn call(&self, store: Self::Store, arg1: &U, arg2: &im::Vector<T>) -> (Self::Store, U) {
        let mut seed = arg1.clone();

        for item in arg2.iter() {
            let (_, result) = self.f.call(self.f.init(), &seed, item);
            seed = result;
        }

        (store, seed)
    }
}

impl<T: Clone, U: Clone, F: FnPersist2<U, T, U>> From<F> for VectorFold<T, U, F> {
    fn from(f: F) -> Self {
        Self::new(f)
    }
}

#[allow(refining_impl_trait)]
impl<T: Clone, U: Clone> FoldPersist<T, U> for im::Vector<T> {
    type C<A> = im::Vector<A>;

    fn fold<F: FnPersist2<U, T, U>>(f: F) -> VectorFold<T, U, F> {
        f.into()
    }
}
