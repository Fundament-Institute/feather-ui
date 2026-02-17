use std::ops::{Add, Bound, Mul, RangeBounds};

use num_traits::{One, Unsigned, int::PrimInt};

/// A compact sequence represents an infinite sequence of values that compacts homogeneous regions
/// of the same value. Therefore, it only takes up 2*N storage for N unique values. To avoid being
/// undefined, it can't be empty (the infinite sequence must always have a value at all indices).
#[repr(transparent)]
pub struct CompactSequence<T, CT = u32>(Vec<(T, CT)>);

impl<T: Clone, CT: Clone> Clone for CompactSequence<T, CT> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: std::fmt::Debug, CT: std::fmt::Debug> std::fmt::Debug for CompactSequence<T, CT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CompactSequence").field(&self.0).finish()
    }
}

impl<T: num_traits::Zero, CT: num_traits::Zero> std::default::Default for CompactSequence<T, CT> {
    fn default() -> Self {
        Self(vec![(T::zero(), CT::zero())])
    }
}

use superslice::*;

impl<T: PartialEq, CT: PrimInt + Unsigned> CompactSequence<T, CT> {
    pub fn new(init: T) -> Self {
        Self(vec![(init, CT::zero())])
    }

    // Inserts a new entry. Won't take up any storage if the entry is the same as one of its neighbors.
    pub fn insert(&mut self, idx: CT, v: T)
    where
        T: Clone,
    {
        // Special handling for inserting at the start
        let mut pos = 1;
        if idx.is_zero() {
            let right = self.get(idx);

            if *right != v {
                self.0.insert(0, (v, idx));
            }
        } else {
            let left = self.get(idx - CT::one());
            pos = self.upper_bound(idx) - 1;
            let right = &self.0[pos].0;
            match (left.eq(&v), right.eq(&v)) {
                // Case: 0 1 2 and 0 1 1
                (false, false) if left.eq(right) => {
                    self.0.insert(pos, (v, idx));
                    pos += 1;
                }
                // Case: 0 1 0
                (false, false) => {
                    // The split here inserts the new split pair and then the value before it. We don't set idx + 1
                    // because idx will be incremented by the increment loop afterwards anyway
                    self.0.insert(pos, (right.clone(), idx));
                    self.0.insert(pos, (v, idx));

                    pos += 1;
                }
                // Case: 0 1 1
                (false, true) => {
                    self.0.insert(pos, (v, idx));
                    pos += 1;
                }
                // Case: 1 1 0
                (true, false) => {}
                // Case: 0 0 0
                (true, true) => {
                    pos += 1;
                }
            }
        }

        for x in &mut self.0[pos..] {
            x.1 = x.1 + CT::one();
        }
    }

    /// Sets the value of an entry. Increases storage size by up to 2 if it splits a previously homogeneous region,
    /// or it can reduce storage size by up to 2 if this change allow merging two homogeneous regions together.
    pub fn set(&mut self, idx: CT, x: T) {
        todo!()
    }

    /// Removes an entry. Fails if you attempt to remove the last entry. Can reduce storage size by up to 2 if this
    /// allows mergingn two homogeous regions together.
    pub fn remove(&mut self, idx: CT) -> Option<T> {
        if self.0.len().is_one() {
            return None;
        }

        todo!()
    }

    #[inline]
    #[must_use]
    fn lower_bound(&self, idx: CT) -> usize {
        self.0.as_slice().lower_bound_by(|x| x.1.cmp(&idx))
    }

    #[inline]
    #[must_use]
    fn upper_bound(&self, idx: CT) -> usize {
        self.0.as_slice().upper_bound_by(|x| x.1.cmp(&idx))
    }

    #[inline]
    pub fn get(&self, idx: CT) -> &T {
        &self.0[self.upper_bound(idx) - 1].0
    }

    // Using the Mul and Add operators, efficiently sums all elements, implied or explicit, over a given range.
    #[must_use]
    pub fn sum(&self, range: impl RangeBounds<CT>) -> T
    where
        T: num_traits::Zero + Mul<CT, Output = T> + Copy,
    {
        let mut start = match range.start_bound() {
            Bound::Included(v) => *v,
            Bound::Excluded(v) => *v + CT::one(),
            Bound::Unbounded => CT::zero(),
        };

        let first = self.upper_bound(start) - 1;

        let end = match range.end_bound() {
            Bound::Included(v) => *v,
            Bound::Excluded(v) => *v - CT::one(),
            Bound::Unbounded => CT::max_value(),
        };

        let last = self.lower_bound(end);

        let mut result = T::zero();
        for i in (first + 1)..last {
            let mid = self.0[i].1;
            result = result + self.0[i - 1].0 * (mid - start);
            start = mid;
        }

        result + self.0[last].0 * (end - start)
    }

    #[must_use]
    pub fn iter(&self) -> Iter<'_, T, CT> {
        Iter {
            it: &self,
            idx: CT::zero(),
        }
    }
}

impl<T: Add + PartialEq, CT: PrimInt + Unsigned> std::ops::Index<CT> for CompactSequence<T, CT> {
    type Output = T;

    fn index(&self, index: CT) -> &Self::Output {
        self.get(index)
    }
}

/// An iterator over an infinite compacted sequence.
pub struct Iter<'a, T, CT> {
    it: &'a CompactSequence<T, CT>,
    idx: CT,
}

impl<'a, T, CT: Clone> Clone for Iter<'a, T, CT> {
    fn clone(&self) -> Self {
        Iter {
            it: self.it,
            idx: self.idx.clone(),
        }
    }
}

impl<'a, T: PartialEq, CT: PrimInt + Unsigned> Iterator for Iter<'a, T, CT> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let element = self.it.get(self.idx);
        self.idx = self.idx + CT::one();
        Some(element)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.it.0.len(), None)
    }
}

impl<'a, T: PartialEq, CT: PrimInt + Unsigned> std::iter::FusedIterator for Iter<'a, T, CT> {}

impl<'a, T: PartialEq, CT: PrimInt + Unsigned> IntoIterator for &'a CompactSequence<T, CT> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, CT>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: PartialEq + Clone, CT: PrimInt + Unsigned> FromIterator<T> for CompactSequence<T, CT> {
    fn from_iter<A: IntoIterator<Item = T>>(it: A) -> Self {
        let mut iter = IntoIterator::into_iter(it);
        let mut seq = Self::new(
            iter.next()
                .expect("Cannot create empty CompactSequence, iterator must be non-empty!"),
        );
        let mut idx = CT::one();

        while let Some(v) = iter.next() {
            seq.insert(idx, v);
            idx = idx + CT::one();
        }

        seq
    }
}

#[test]
pub fn basic_seq_test() {
    let mut seq = CompactSequence::<i32>::new(1);

    assert_eq!(*seq.get(0), 1);
    assert_eq!(*seq.get(1), 1);
    assert_eq!(*seq.get(2), 1);

    assert_eq!(seq[0], 1);
    assert_eq!(seq[1], 1);
    assert_eq!(seq[2], 1);

    assert_eq!(seq.remove(0), None);
}
