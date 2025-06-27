// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::cell::RefCell;
use std::ops;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use cosmic_text::{Affinity, AttrsList, Cursor, Metrics};
use smallvec::SmallVec;

use crate::graphics::point_to_pixel;

/// Represents a single change, recording the (`start`,`end`) range of the new string, and the old string
/// that used to be contained in that range. `start` and `end` might be equal, which represents a deletion.
/// Likewise, old might be empty, which represents an insertion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub start: Cursor,
    pub end: Cursor,
    pub old: SmallVec<[u8; 4]>,
    pub attrs: Option<AttrsList>,
}

#[derive(Debug)]
pub struct EditBuffer {
    pub(crate) buffer: Rc<RefCell<cosmic_text::Buffer>>,
    pub(crate) count: AtomicUsize,
    pub(crate) reflow: AtomicBool,
    cursor: AtomicUsize,
    select: AtomicUsize, // If there's a selection, this is different from cursor and points at the end. Can be less than cursor.
}

impl Default for EditBuffer {
    fn default() -> Self {
        Self {
            buffer: Rc::new(RefCell::new(cosmic_text::Buffer::new_empty(Metrics::new(
                1.0, 1.0,
            )))),
            count: Default::default(),
            reflow: Default::default(),
            cursor: Default::default(),
            select: Default::default(),
        }
    }
}
impl Clone for EditBuffer {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            count: self.count.load(Ordering::Relaxed).into(),
            reflow: self.reflow.load(Ordering::Relaxed).into(),
            cursor: self.cursor.load(Ordering::Relaxed).into(),
            select: self.select.load(Ordering::Relaxed).into(),
        }
    }
}

impl EditBuffer {
    pub fn new(text: &str, cursor: (usize, usize)) -> Self {
        let this = Self {
            buffer: Rc::new(RefCell::new(cosmic_text::Buffer::new_empty(Metrics {
                font_size: 1.0,
                line_height: 1.0,
            }))),
            count: 0.into(),
            reflow: true.into(),
            cursor: cursor.0.into(),
            select: cursor.1.into(),
        };
        this.set_content(text);
        this
    }
    pub fn get_content(&self) -> String {
        let mut s = String::new();
        s.reserve(
            self.buffer
                .borrow()
                .lines
                .iter()
                .fold(0, |c, l| c + l.text().len() + l.ending().as_str().len()),
        );
        for line in &self.buffer.borrow().lines {
            s.push_str(line.text());
            s.push_str(line.ending().as_str());
        }
        s
    }

    pub fn set_content(&self, content: &str) {
        let mut buffer = self.buffer.borrow_mut();
        buffer.lines.clear();
        for (range, ending) in cosmic_text::LineIter::new(content) {
            buffer.lines.push(cosmic_text::BufferLine::new(
                &content[range],
                ending,
                AttrsList::new(&cosmic_text::Attrs::new()),
                cosmic_text::Shaping::Advanced,
            ));
        }
        if buffer.lines.is_empty() {
            buffer.lines.push(cosmic_text::BufferLine::new(
                "",
                cosmic_text::LineEnding::default(),
                AttrsList::new(&cosmic_text::Attrs::new()),
                cosmic_text::Shaping::Advanced,
            ));
        }
        self.reflow.store(true, Ordering::Release);
        self.count.fetch_add(1, Ordering::Release);
    }

    pub fn edit(
        &self,
        multisplice: &[(ops::Range<usize>, String)],
    ) -> SmallVec<[(ops::Range<usize>, String); 1]> {
        let mut text = self.get_content();
        if multisplice.len() == 1 {
            let (range, replace) = &multisplice[0];
            let old = text[range.clone()].to_string();
            text.replace_range(range.clone(), replace);
            self.set_content(&text);
            [(range.start..replace.len(), old)].into()
        } else {
            // To preserve the validity of the ranges, we have to assemble the string piecewise
            let mut undo = SmallVec::new();
            let mut last = 0;
            let mut s = String::new();
            {
                for (range, replace) in multisplice {
                    s.push_str(&text[last..range.start]);
                    s.push_str(replace);
                    undo.push((range.start..replace.len(), text[range.clone()].to_string()));
                    last = range.end;
                }

                s.push_str(&text[last..]);
            };
            self.set_content(&s);
            undo
        }
    }

    fn compact(mut idx: usize, affinity: Affinity) -> usize {
        const FLAG: usize = 1 << (usize::BITS - 1);
        idx = idx & (!FLAG);
        if affinity == Affinity::After {
            idx = idx | FLAG;
        }
        idx
    }
    fn expand(cursor: usize) -> (usize, Affinity) {
        const FLAG: usize = 1 << (usize::BITS - 1);
        (
            cursor & (!FLAG),
            if (cursor & FLAG) != 0 {
                Affinity::After
            } else {
                Affinity::Before
            },
        )
    }

    pub fn get_cursor(&self) -> (usize, Affinity) {
        Self::expand(self.cursor.load(Ordering::Relaxed))
    }

    pub fn get_selection(&self) -> (usize, Affinity) {
        Self::expand(self.select.load(Ordering::Relaxed))
    }

    pub fn set_cursor(&self, cursor: usize, affinity: Affinity) {
        let start = Self::compact(cursor, affinity);
        self.cursor.store(start, Ordering::Release);
        self.select.store(start, Ordering::Release);
        self.count.fetch_add(1, Ordering::Release);
    }
    pub fn set_selection(&self, start: (usize, Affinity), end: (usize, Affinity)) {
        let cursor = Self::compact(start.0, start.1);
        let select = Self::compact(end.0, end.1);
        self.cursor.store(cursor, Ordering::Release);
        self.select.store(select, Ordering::Release);
        self.count.fetch_add(1, Ordering::Release);
    }

    pub fn to_cursor(&self, cursor: (usize, Affinity)) -> Cursor {
        let mut lines = 0;
        let (mut idx, mut affinity) = cursor;
        for line in &self.buffer.borrow().lines {
            let len = line.text().len();
            if len >= idx {
                break;
            }
            idx -= len;
            lines += 1;
            if idx < line.ending().as_str().len() {
                affinity = Affinity::Before;
                idx = 0;
                break;
            }
            idx -= line.ending().as_str().len();
        }
        Cursor {
            line: lines,
            index: idx,
            affinity,
        }
    }

    pub fn from_cursor(&self, cursor: Cursor) -> (usize, Affinity) {
        let mut idx = 0;
        for line in self.buffer.borrow().lines.iter().take(cursor.line) {
            idx += line.text().len() + line.ending().as_str().len();
        }
        (idx + cursor.index, cursor.affinity)
    }

    pub fn flowtext(
        &self,
        font_system: &mut crate::cosmic_text::FontSystem,
        font_size: f32,
        line_height: f32,
        wrap: cosmic_text::Wrap,
        dpi: ultraviolet::Vec2,
        attrs: cosmic_text::Attrs<'_>,
    ) {
        let mut text_buffer = self.buffer.borrow_mut();

        let metrics = cosmic_text::Metrics::new(
            point_to_pixel(font_size, dpi.x),
            point_to_pixel(line_height, dpi.y),
        );

        if text_buffer.metrics() != metrics {
            text_buffer.set_metrics(font_system, metrics);
        }
        if text_buffer.wrap() != wrap {
            text_buffer.set_wrap(font_system, wrap);
        }
        for line in &mut text_buffer.lines {
            line.set_attrs_list(AttrsList::new(&attrs));
        }
        text_buffer.shape_until_scroll(font_system, false);
        self.reflow.store(false, Ordering::Release);
    }
}

#[derive(Default, Debug)]
pub struct EditView {
    pub(crate) obj: Rc<EditBuffer>,
    count: usize,
    reflow: bool,
}

impl EditView {
    pub fn get(&self) -> &EditBuffer {
        &self.obj
    }
}

// Ensures each clone gets a fresh snapshot to capture changes
impl Clone for EditView {
    fn clone(&self) -> Self {
        Self {
            obj: self.obj.clone(),
            count: self.obj.count.load(Ordering::Acquire),
            reflow: self.obj.reflow.load(Ordering::Acquire),
        }
    }
}

impl Eq for EditView {}
impl PartialEq for EditView {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count
            && self.reflow == other.reflow
            && Rc::ptr_eq(&self.obj, &other.obj)
    }
}

impl From<Rc<EditBuffer>> for EditView {
    fn from(value: Rc<EditBuffer>) -> Self {
        Self {
            obj: value.clone(),
            count: value.count.load(Ordering::Acquire),
            reflow: value.reflow.load(Ordering::Acquire),
        }
    }
}

impl From<EditBuffer> for EditView {
    fn from(value: EditBuffer) -> Self {
        let value = Rc::new(value);
        Self {
            obj: value.clone(),
            count: value.count.load(Ordering::Acquire),
            reflow: value.reflow.load(Ordering::Acquire),
        }
    }
}
