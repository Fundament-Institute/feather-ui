// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use smallvec::SmallVec;
use unicode_segmentation::GraphemeCursor;

#[derive(Default, Debug)]
pub struct EditObj {
    pub(crate) text: RefCell<String>,
    count: AtomicUsize,
    cursor: AtomicUsize,
    select: AtomicUsize, // If there's a selection, this is different from cursor and points at the end. Can be less than cursor.
}

impl Clone for EditObj {
    fn clone(&self) -> Self {
        Self {
            text: self.text.clone(),
            count: self.count.load(Ordering::Relaxed).into(),
            cursor: self.cursor.load(Ordering::Relaxed).into(),
            select: self.select.load(Ordering::Relaxed).into(),
        }
    }
}

impl EditObj {
    pub fn new(text: String, cursor: (usize, usize)) -> Self {
        Self {
            text: text.into(),
            count: 0.into(),
            cursor: cursor.0.into(),
            select: cursor.1.into(),
        }
    }
    pub fn get_content(&self) -> std::cell::Ref<'_, String> {
        self.text.borrow()
    }
    pub fn set_content(&self, content: &str) {
        *self.text.borrow_mut() = content.into();
        self.count.fetch_add(1, Ordering::Release);
    }
    pub fn edit(
        &self,
        multisplice: &[(std::ops::Range<usize>, String)],
    ) -> SmallVec<[(std::ops::Range<usize>, String); 1]> {
        let old: SmallVec<[(std::ops::Range<usize>, String); 1]> = if multisplice.len() == 1 {
            let (range, replace) = &multisplice[0];
            let old = self.text.borrow()[range.clone()].to_string();
            self.text.borrow_mut().replace_range(range.clone(), replace);
            [(range.start..replace.len(), old)].into()
        } else {
            // To preserve the validity of the ranges, we have to assemble the string piecewise
            let mut undo = SmallVec::new();
            let mut last = 0;
            let s = {
                let mut pieces: Vec<&str> = Vec::new();
                let txt = self.text.borrow();
                for (range, replace) in multisplice {
                    pieces.push(&txt[last..range.start]);
                    pieces.push(replace);
                    undo.push((range.start..replace.len(), txt[range.clone()].to_string()));
                    last = range.end;
                }

                pieces.push(&txt[last..]);
                pieces.join("")
            };
            self.text.replace(s);
            undo
        };
        self.count.fetch_add(1, Ordering::Release);
        old
    }

    pub fn get_cursor(&self) -> (usize, usize) {
        (
            self.cursor.load(Ordering::Relaxed),
            self.select.load(Ordering::Relaxed),
        )
    }

    pub fn set_cursor(&self, cursor: usize) {
        self.cursor.store(cursor, Ordering::Release);
        self.select.store(cursor, Ordering::Release);
        self.count.fetch_add(1, Ordering::Release);
    }
    pub fn set_selection(&self, cursor: (usize, usize)) {
        self.cursor.store(cursor.0, Ordering::Release);
        self.select.store(cursor.1, Ordering::Release);
        self.count.fetch_add(1, Ordering::Release);
    }

    pub(crate) fn next_grapheme(&self, byte_offset: usize) -> usize {
        let txt = self.text.borrow();
        let mut cursor = GraphemeCursor::new(byte_offset, txt.len(), true);
        assert!(cursor.is_boundary(&txt, 0).is_ok_and(|v| v));
        match cursor.next_boundary(&txt, 0) {
            Ok(Some(x)) => x,
            Ok(None) => txt.len(),
            Err(_) => byte_offset,
        }
    }

    pub(crate) fn prev_grapheme(&self, byte_offset: usize) -> usize {
        let txt = self.text.borrow();
        let mut cursor = GraphemeCursor::new(byte_offset, txt.len(), true);
        assert!(cursor.is_boundary(&txt, 0).is_ok_and(|v| v));
        match cursor.prev_boundary(&txt, 0) {
            Ok(Some(x)) => x,
            Ok(None) => 0,
            Err(_) => byte_offset,
        }
    }
}

#[derive(Default, Debug)]
pub struct Snapshot {
    pub(crate) obj: Rc<EditObj>,
    count: usize,
}

impl Snapshot {
    pub fn get(&self) -> &EditObj {
        &self.obj
    }
}

// Ensures each clone gets a fresh snapshot to capture changes
impl Clone for Snapshot {
    fn clone(&self) -> Self {
        Self {
            obj: self.obj.clone(),
            count: self.obj.count.load(Ordering::Acquire),
        }
    }
}

impl Eq for Snapshot {}
impl PartialEq for Snapshot {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && Rc::ptr_eq(&self.obj, &other.obj)
    }
}

impl PartialOrd for Snapshot {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if Rc::ptr_eq(&self.obj, &other.obj) {
            self.count.partial_cmp(&other.count)
        } else {
            None
        }
    }
}

impl From<Rc<EditObj>> for Snapshot {
    fn from(value: Rc<EditObj>) -> Self {
        Self {
            obj: value.clone(),
            count: value.count.load(std::sync::atomic::Ordering::Acquire),
        }
    }
}

impl From<EditObj> for Snapshot {
    fn from(value: EditObj) -> Self {
        let value = Rc::new(value);
        Self {
            obj: value.clone(),
            count: value.count.load(std::sync::atomic::Ordering::Acquire),
        }
    }
}
