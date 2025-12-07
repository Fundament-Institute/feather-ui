// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::mem::{ManuallyDrop, MaybeUninit, transmute};
use std::{collections::HashMap, rc::Rc};

struct Node {
    next: Option<Box<Node>>,
    chunk: Box<[u8]>,
}

pub struct Pool {
    root: Box<Node>,
    freelist: *const u8,
    stride: usize,
}

impl Pool {
    pub fn new(stride: usize, initial: usize) -> Self {
        let mut freelist = std::ptr::null();
        Self {
            root: Box::new(Node {
                next: None,
                chunk: Self::init_chunk(&mut freelist, initial * stride, stride),
            }),
            freelist,
            stride,
        }
    }

    fn new_chunk(&mut self, size: usize) {
        let mut node = Box::new(Node {
            next: None,
            chunk: Self::init_chunk(&mut self.freelist, size, self.stride),
        });
        std::mem::swap(&mut self.root, &mut node);
        self.root.next = Some(node);
    }

    fn init_chunk(freelist: &mut *const u8, size: usize, stride: usize) -> Box<[u8]> {
        let mut chunk = Box::new_uninit_slice(size);
        let mut memref = chunk.as_mut_ptr() as *mut u8;
        let memend = unsafe { memref.add(size) };

        while memref < memend {
            unsafe {
                *(memref as *mut *const u8) = *freelist;
                *freelist = memref;
                memref = memref.add(stride);
            }
        }

        unsafe { chunk.assume_init() }
    }

    pub fn alloc<T>(&mut self) -> *mut MaybeUninit<T> {
        assert!(size_of::<T>() == self.stride);

        if self.freelist.is_null() {
            self.new_chunk(self.root.chunk.len() * 2);
            assert!(!self.freelist.is_null());
        }

        let result = self.freelist as *mut MaybeUninit<T>;
        self.freelist = unsafe { *(self.freelist as *const *const u8) };
        result
    }

    pub fn dealloc<T>(&mut self, ptr: *mut T) {
        assert!(size_of::<T>() == self.stride);

        unsafe {
            *(ptr as *mut *const u8) = self.freelist;
        }
        self.freelist = ptr as *const u8;
    }
}

#[derive(Default)]
pub struct ArenaAllocPool {
    arenas: HashMap<usize, Pool>,
}

impl ArenaAllocPool {
    pub fn alloc<T>(&mut self) -> *mut MaybeUninit<T> {
        self.arenas
            .entry(size_of::<T>())
            .or_insert_with_key(|sz| Pool::new(*sz, 32))
            .alloc()
    }

    pub fn dealloc<T>(&mut self, ptr: *mut T) {
        self.arenas
            .get_mut(&size_of::<T>())
            .expect("Tried to deallocate object that had no pool???")
            .dealloc(ptr);
    }
}

pub struct PoolRc<T>(std::rc::Rc<T>);

thread_local! {
    static POOL: std::cell::RefCell<ArenaAllocPool> = Default::default();
}

type UntypedRc = Rc<()>;

pub struct PoolRcK {
    inner: ManuallyDrop<UntypedRc>,
}

impl PoolRcK {
    #[inline(always)]
    fn new_from_inner<T>(rc: Rc<T>) -> PoolRcK {
        PoolRcK {
            inner: ManuallyDrop::new(unsafe { transmute::<Rc<T>, UntypedRc>(rc) }),
        }
    }

    #[inline(always)]
    unsafe fn take_inner<T>(self) -> Rc<T> {
        unsafe {
            let rc: UntypedRc = ManuallyDrop::into_inner(self.inner);

            transmute(rc)
        }
    }

    #[inline(always)]
    unsafe fn as_inner_ref<T>(&self) -> &Rc<T> {
        use std::ops::Deref;
        unsafe {
            let rc_t: *const Rc<T> =
                std::ptr::from_ref::<UntypedRc>(self.inner.deref()).cast::<Rc<T>>();

            // Static check to make sure we are not messing up the sizes.
            // This could happen if we allowed for `T` to be unsized, because it would need to be
            // represented as a wide pointer inside `Rc`.
            // TODO Use static_assertion when https://github.com/nvzqz/static-assertions-rs/issues/21
            //      gets fixed
            let _ = transmute::<UntypedRc, Rc<T>>;

            &*rc_t
        }
    }

    #[inline(always)]
    unsafe fn as_inner_mut<T>(&mut self) -> &mut Rc<T> {
        use std::ops::DerefMut;
        unsafe {
            let rc_t: *mut Rc<T> =
                std::ptr::from_mut::<UntypedRc>(self.inner.deref_mut()).cast::<Rc<T>>();

            &mut *rc_t
        }
    }
}

unsafe impl imbl::archery::SharedPointerKind for PoolRcK {
    #[inline(always)]
    fn new<T>(v: T) -> PoolRcK {
        let p: *mut MaybeUninit<T> = POOL.with_borrow_mut(|x| x.alloc());
        unsafe {
            (*p).write(v);
            PoolRcK::new_from_inner(Rc::from_raw(p))
        }
    }

    #[inline(always)]
    fn from_box<T>(_: Box<T>) -> PoolRcK {
        unimplemented!("Can't use from_box with arena allocated memory");
    }

    #[inline(always)]
    unsafe fn as_ptr<T>(&self) -> *const T {
        unsafe { Rc::as_ptr(self.as_inner_ref()) }
    }

    #[inline(always)]
    unsafe fn deref<T>(&self) -> &T {
        unsafe { self.as_inner_ref::<T>().as_ref() }
    }

    #[inline(always)]
    unsafe fn try_unwrap<T>(self) -> Result<T, PoolRcK> {
        unsafe { Rc::try_unwrap(self.take_inner()).map_err(PoolRcK::new_from_inner) }
    }

    #[inline(always)]
    unsafe fn get_mut<T>(&mut self) -> Option<&mut T> {
        unsafe { Rc::get_mut(self.as_inner_mut()) }
    }

    #[inline(always)]
    unsafe fn make_mut<T: Clone>(&mut self) -> &mut T {
        unsafe { Rc::make_mut(self.as_inner_mut()) }
    }

    #[inline(always)]
    unsafe fn strong_count<T>(&self) -> usize {
        unsafe { Rc::strong_count(self.as_inner_ref::<T>()) }
    }

    #[inline(always)]
    unsafe fn clone<T>(&self) -> PoolRcK {
        unsafe {
            PoolRcK {
                inner: ManuallyDrop::new(Rc::clone(self.as_inner_ref())),
            }
        }
    }

    #[inline(always)]
    unsafe fn drop<T>(&mut self) {
        unsafe {
            // This is safe, because the Rc lives inside a ManualDrop, so it will never try to deallocate the memory.
            POOL.with_borrow_mut(|x| x.dealloc(Rc::as_ptr(self.as_inner_ref::<T>()) as *mut T))
        }
    }
}

impl std::fmt::Debug for PoolRcK {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.write_str("ArenaRcK")
    }
}

pub struct ArenaBox<T>(ManuallyDrop<Box<T>>);

impl<T> ArenaBox<T> {
    pub fn new(v: T) -> Self {
        let p: *mut MaybeUninit<T> = POOL.with_borrow_mut(|x| x.alloc());
        unsafe {
            (*p).write(v);
            Self(ManuallyDrop::new(Box::from_raw(p).assume_init()))
        }
    }
}
impl<T> std::borrow::Borrow<T> for ArenaBox<T> {
    fn borrow(&self) -> &T {
        &self.0
    }
}

impl<T> std::borrow::BorrowMut<T> for ArenaBox<T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Clone> Clone for ArenaBox<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for ArenaBox<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ArenaBox").field(&self.0).finish()
    }
}

impl<T> std::ops::Deref for ArenaBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for ArenaBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> std::ops::Drop for ArenaBox<T> {
    fn drop(&mut self) {
        // This is safe, because the Box lives inside a ManualDrop, so it will never try to deallocate the memory.
        POOL.with_borrow_mut(|x| x.dealloc(self.0.as_mut() as *mut T))
    }
}

/*
#[cfg(test)]
fn shared_pointer_fuzzer<
    P: imbl::shared_ptr::SharedPointerKind,
    const TRIALS: usize,
    const MAXSIZE: usize,
>() {
    let plist: Vec<(P, usize)> = Vec::new();
    assert!(MAXSIZE < 16);
    const id: u8 = 0;

    for k in 0..TRIALS {
        if fastrand::bool() || plist.len() < 3 {
            let sz = fastrand::usize(1..=MAXSIZE);

            let test = match sz {
                1 => P::new::<[u8; 1]>(),
                2 => P::new::<[u8; 2]>(),
                3 => P::new::<[u8; 3]>(),
                4 => P::new::<[u8; 4]>(),
                5 => P::new::<[u8; 5]>(),
                6 => P::new::<[u8; 6]>(),
                7 => P::new::<[u8; 7]>(),
                8 => P::new::<[u8; 8]>(),
                9 => P::new::<[u8; 9]>(),
                10 => P::new::<[u8; 10]>(),
            };

            test.iter_mut()
                .enumerate()
                .map(|(i, x)| x = (uint8_t)((i + id) & 0xFF));
            plist.push((test, sz));
        } else {
            let (p, s) = &plist[fastrand::usize(0..plist.len())];
            let slice = unsafe { std::slice::from_raw_parts(*p, s) };

            for (i, b) in slice.iter().enumerate() {
                assert_eq!(b, (i + id) & 0xFF);
            }

            slice.fill(0xfd);

            plist.swap(index, plist.len() - 1);
            plist.pop();
        }
    }

    while !plist.is_empty() {}

    TEST(pass);
}*/

#[test]
fn test_arena_alloc() {}
