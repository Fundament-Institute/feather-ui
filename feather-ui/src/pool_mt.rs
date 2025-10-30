use static_assertions::const_assert_eq;
use std::mem::MaybeUninit;
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicPtr};

#[cfg(target_pointer_width = "64")]
type AtomicDoublePtr = std::sync::atomic::AtomicU128;

#[cfg(target_pointer_width = "32")]
type AtomicDoublePtr = std::sync::atomic::AtomicU64;

struct Node {
    next: Option<Box<Node>>,
    chunk: Box<[u8]>,
}

#[repr(C, align(16))]
#[derive(Default, Copy, Clone, PartialEq, Eq)]
struct PtrTag(*const u8, usize);

impl PtrTag {
    fn zero() -> Self {
        Default::default()
    }

    #[cfg(target_pointer_width = "64")]
    fn pack(&self) -> u128 {
        let ptr_val = self.0.addr() as u64;
        (ptr_val as u128) | ((self.1 as u128) << 64)
    }

    #[cfg(target_pointer_width = "32")]
    fn pack(&self) -> u64 {
        let ptr_val = self.0.addr() as u32;
        (ptr_val as u64) | ((self.1 as u64) << 32)
    }

    #[cfg(target_pointer_width = "64")]
    fn new(value: u128) -> Self {
        Self((value as u64) as *mut u8, (value >> 64) as usize)
    }

    #[cfg(target_pointer_width = "32")]
    fn new(value: u64) -> Self {
        Self((value as u32) as *mut u8, (value >> 32) as usize)
    }
}

pub struct PoolAlloc<const N: usize, const INIT: usize = 64> {
    freelist: AtomicDoublePtr,
    root: AtomicPtr<Node>,
    flag: AtomicBool,
}

impl<const N: usize, const INIT: usize> Default for PoolAlloc<N, INIT> {
    fn default() -> Self {
        Self {
            freelist: Default::default(),
            root: Default::default(),
            flag: Default::default(),
        }
    }
}

impl<const N: usize, const INIT: usize> PoolAlloc<N, INIT> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn clear(&mut self) {
        if !self.root.load(Ordering::Acquire).is_null() {
            while self.flag.swap(true, Ordering::Acquire) {}
            unsafe {
                std::mem::drop(Box::from_raw(
                    self.root.swap(std::ptr::null_mut(), Ordering::AcqRel),
                ));
            }
            self.freelist = Default::default();
            self.flag.store(false, Ordering::Release);
        }
    }

    pub fn alloc<T>(&self) -> &mut MaybeUninit<T> {
        const_assert_eq!(size_of::<T>(), N);
        let mut ret = PtrTag::zero().pack();
        Self::compare_and_swap(&self.freelist, ret, ret, &mut ret);

        loop {
            let cur = PtrTag::new(ret);
            if cur.0.is_null() {
                if !self.flag.swap(true, Ordering::Acquire) {
                    // Check this due to race condition where someone finishes a new allocation and unlocks while we were testing that flag.
                    if PtrTag::new(self.freelist.load(Ordering::Acquire))
                        .0
                        .is_null()
                    {
                        let root = unsafe { self.root.load(Ordering::Acquire).as_mut() };

                        self.new_chunk(root.map(|x| x.chunk.len() * 2).unwrap_or(INIT * N));
                    }
                    self.flag.store(false, Ordering::Release);
                }

                Self::compare_and_swap(&self.freelist, ret, ret, &mut ret);
                continue;
            }

            let cur = PtrTag::new(ret);
            let nval = PtrTag(unsafe { *(cur.0 as *const *const u8) }, cur.1 + 1);

            if Self::compare_and_swap(&self.freelist, nval.pack(), ret, &mut ret) {
                break;
            }
        }

        unsafe {
            (PtrTag::new(ret).0 as *mut MaybeUninit<T>)
                .as_mut()
                .expect("freelist contained null value!")
        }
    }

    pub fn dealloc<T>(&self, ptr: &mut T) {
        self.set_freelist(ptr as *mut T as *const u8, ptr as *mut T as *mut u8);
    }

    fn new_chunk(&self, size: usize) {
        let root = self.root.load(Ordering::Acquire);
        let next = unsafe { (!root.is_null()).then(move || Box::from_raw(root)) };

        let node = Box::new(Node {
            next,
            chunk: self.init_chunk(size),
        });
        self.root
            .store(Box::leak(node) as *mut Node, Ordering::Release);
    }

    fn init_chunk(&self, size: usize) -> Box<[u8]> {
        let mut hold = std::ptr::null_mut();
        let result = unsafe {
            let mut chunk = Box::new_uninit_slice(size);
            let mut memref = chunk.as_mut_ptr() as *mut u8;
            let memend = memref.add(chunk.len());

            while memref < memend {
                *(memref as *mut *const u8) = hold;
                hold = memref;
                memref = memref.add(N);
            }
            chunk.assume_init()
        };
        self.set_freelist(hold, result.as_ptr()); // The target here is different because normally, the first block (at chunk+1) would point to whatever _freelist used to be. However, since we are lockless, _freelist could not be 0 at the time we insert this, so we have to essentially go backwards and set the first one to whatever freelist is NOW, before setting freelist to the one on the end.
        result
    }

    fn set_freelist(&self, p: *const u8, target: *const u8) {
        let mut prev = PtrTag::zero().pack();
        let mut nval = PtrTag(p, 0);
        Self::compare_and_swap(&self.freelist, prev, prev, &mut prev);

        loop {
            let prevptr = PtrTag::new(prev);
            nval.1 = prevptr.1 + 1;
            unsafe {
                *(target as *mut *const u8) = prevptr.0;
            }

            if Self::compare_and_swap(&self.freelist, nval.pack(), prev, &mut prev) {
                break;
            }
        }
    }

    #[cfg(target_pointer_width = "64")]
    fn compare_and_swap(src: &AtomicDoublePtr, new: u128, old: u128, result: &mut u128) -> bool {
        match src.compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Acquire) {
            Ok(x) => {
                *result = x;
                true
            }
            Err(x) => {
                *result = x;
                false
            }
        }
    }

    #[cfg(target_pointer_width = "32")]
    fn compare_and_swap(src: &AtomicDoublePtr, new: u64, old: u64, result: &mut u64) -> bool {
        match src.compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Acquire) {
            Ok(x) => {
                *result = x;
                true
            }
            Err(x) => {
                *result = x;
                false
            }
        }
    }
}

impl<const N: usize, const INIT: usize> Drop for PoolAlloc<N, INIT> {
    fn drop(&mut self) {
        unsafe {
            std::mem::drop(Box::from_raw(
                self.root.swap(std::ptr::null_mut(), Ordering::AcqRel),
            ));
        }
    }
}
