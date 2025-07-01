// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>
// This file is a modified version of editor.rs from cosmic-text and falls under their license terms.

use core::iter::once;
use core::panic;
use smallvec::SmallVec;
use unicode_segmentation::UnicodeSegmentation;

use cosmic_text::{
    Action, Attrs, AttrsList, Buffer, BufferLine, Cursor, FontSystem, LayoutRun, Selection, Shaping,
};

use crate::text::Change;

/// A wrapper of [`Buffer`] for easy editing. Modified to use an external Rc<RefCell<Buffer>> reference
/// to work better with our immutable framework
#[derive(Debug, Clone)]
pub struct Editor {
    cursor: Cursor,
    cursor_x_opt: Option<i32>,
    selection: Selection,
    cursor_moved: bool,
    auto_indent: bool,
}

impl PartialEq for Editor {
    fn eq(&self, other: &Self) -> bool {
        self.cursor == other.cursor
            && self.cursor_x_opt == other.cursor_x_opt
            && self.selection == other.selection
            && self.cursor_moved == other.cursor_moved
            && self.auto_indent == other.auto_indent
    }
}

fn cursor_glyph_opt(cursor: &Cursor, run: &LayoutRun) -> Option<(usize, f32)> {
    if cursor.line == run.line_i {
        for (glyph_i, glyph) in run.glyphs.iter().enumerate() {
            if cursor.index == glyph.start {
                return Some((glyph_i, 0.0));
            } else if cursor.index > glyph.start && cursor.index < glyph.end {
                // Guess x offset based on characters
                let mut before = 0;
                let mut total = 0;

                let cluster = &run.text[glyph.start..glyph.end];
                for (i, _) in cluster.grapheme_indices(true) {
                    if glyph.start + i < cursor.index {
                        before += 1;
                    }
                    total += 1;
                }

                let offset = glyph.w * (before as f32) / (total as f32);
                return Some((glyph_i, offset));
            }
        }
        match run.glyphs.last() {
            Some(glyph) => {
                if cursor.index == glyph.end {
                    return Some((run.glyphs.len(), 0.0));
                }
            }
            None => {
                return Some((0, 0.0));
            }
        }
    }
    None
}

pub fn cursor_position(cursor: &Cursor, run: &LayoutRun) -> Option<(i32, i32)> {
    let (cursor_glyph, cursor_glyph_offset) = cursor_glyph_opt(cursor, run)?;
    let x = match run.glyphs.get(cursor_glyph) {
        Some(glyph) => {
            // Start of detected glyph
            if glyph.level.is_rtl() {
                (glyph.x + glyph.w - cursor_glyph_offset) as i32
            } else {
                (glyph.x + cursor_glyph_offset) as i32
            }
        }
        None => match run.glyphs.last() {
            Some(glyph) => {
                // End of last glyph
                if glyph.level.is_rtl() {
                    glyph.x as i32
                } else {
                    (glyph.x + glyph.w) as i32
                }
            }
            None => {
                // Start of empty line
                0
            }
        },
    };

    Some((x, run.line_top as i32))
}

#[allow(dead_code)]
impl Editor {
    /// Create a new [`Editor`]
    pub fn new() -> Self {
        Self {
            cursor: Cursor::default(),
            cursor_x_opt: None,
            selection: Selection::None,
            cursor_moved: false,
            auto_indent: false,
        }
    }

    pub fn cursor(&self) -> Cursor {
        self.cursor
    }

    pub fn set_cursor(&mut self, buffer: &mut Buffer, cursor: Cursor) {
        if self.cursor != cursor {
            self.cursor = cursor;
            self.cursor_moved = true;
            buffer.set_redraw(true);
        }
    }

    pub fn selection(&self) -> Selection {
        self.selection
    }

    pub fn selection_or_cursor(&self) -> Cursor {
        match self.selection {
            cosmic_text::Selection::None => self.cursor,
            cosmic_text::Selection::Normal(cursor)
            | cosmic_text::Selection::Line(cursor)
            | cosmic_text::Selection::Word(cursor) => cursor,
        }
    }

    pub fn set_selection(&mut self, buffer: &mut Buffer, selection: Selection) {
        if self.selection != selection {
            self.selection = selection;
            buffer.set_redraw(true);
        }
    }

    pub fn auto_indent(&self) -> bool {
        self.auto_indent
    }

    pub fn set_auto_indent(&mut self, auto_indent: bool) {
        self.auto_indent = auto_indent;
    }

    pub fn shape_as_needed(
        &mut self,
        font_system: &mut FontSystem,
        buffer: &mut Buffer,
        prune: bool,
    ) {
        if self.cursor_moved {
            let cursor = self.cursor;
            buffer.shape_until_cursor(font_system, cursor, prune);
            self.cursor_moved = false;
        } else {
            buffer.shape_until_scroll(font_system, prune);
        }
    }

    fn replace_range(
        &mut self,
        font_system: &mut FontSystem,
        buffer: &mut Buffer,
        start: Cursor,
        end: Cursor,
        data: &str,
        attrs: Option<AttrsList>,
    ) -> (Cursor, Change) {
        let mut first = std::cmp::min(start, end);
        let mut last = std::cmp::max(start, end);
        let change = if start != end {
            self.delete_range(font_system, buffer, start, end, false)
        } else {
            None
        };
        self.selection = Selection::None;

        let mut change = change.unwrap_or(Change {
            start,
            end: start,
            old: Default::default(),
            attrs: None,
        });

        let mut remaining_split_len = data.len();
        if remaining_split_len == 0 {
            self.layout_range(font_system, buffer, first, last);
            return (start, change);
        }

        let mut cursor = start;

        // Ensure there are enough lines in the buffer to handle this cursor
        while cursor.line >= buffer.lines.len() {
            let ending = buffer
                .lines
                .last()
                .map(|line| line.ending())
                .unwrap_or_default();
            let line = BufferLine::new(
                String::new(),
                ending,
                AttrsList::new(&attrs.as_ref().map_or_else(
                    || {
                        buffer
                            .lines
                            .last()
                            .map_or(Attrs::new(), |line| line.attrs_list().defaults())
                    },
                    |x| x.defaults(),
                )),
                Shaping::Advanced,
            );
            buffer.lines.push(line);
        }

        let line: &mut BufferLine = &mut buffer.lines[cursor.line];
        let insert_line = cursor.line + 1;
        let ending = line.ending();

        // Collect text after insertion as a line
        let after: BufferLine = line.split_off(cursor.index);
        let after_len = after.text().len();

        // Collect attributes
        let mut final_attrs = attrs.unwrap_or_else(|| {
            AttrsList::new(&line.attrs_list().get_span(cursor.index.saturating_sub(1)))
        });

        // Append the inserted text, line by line
        // we want to see a blank entry if the string ends with a newline
        //TODO: adjust this to get line ending from data?
        let addendum = once("").filter(|_| data.ends_with('\n'));
        let mut lines_iter = data.split_inclusive('\n').chain(addendum);
        if let Some(data_line) = lines_iter.next() {
            let mut these_attrs = final_attrs.split_off(data_line.len());
            remaining_split_len -= data_line.len();
            core::mem::swap(&mut these_attrs, &mut final_attrs);
            line.append(BufferLine::new(
                data_line
                    .strip_suffix(char::is_control)
                    .unwrap_or(data_line),
                ending,
                these_attrs,
                Shaping::Advanced,
            ));
        } else {
            panic!("str::lines() did not yield any elements");
        }
        if let Some(data_line) = lines_iter.next_back() {
            remaining_split_len -= data_line.len();
            let mut tmp = BufferLine::new(
                data_line
                    .strip_suffix(char::is_control)
                    .unwrap_or(data_line),
                ending,
                final_attrs.split_off(remaining_split_len),
                Shaping::Advanced,
            );
            tmp.append(after);
            buffer.lines.insert(insert_line, tmp);
            cursor.line += 1;
        } else {
            line.append(after);
        }
        for data_line in lines_iter.rev() {
            remaining_split_len -= data_line.len();
            let tmp = BufferLine::new(
                data_line
                    .strip_suffix(char::is_control)
                    .unwrap_or(data_line),
                ending,
                final_attrs.split_off(remaining_split_len),
                Shaping::Advanced,
            );
            buffer.lines.insert(insert_line, tmp);
            cursor.line += 1;
        }

        assert_eq!(remaining_split_len, 0);

        // Append the text after insertion
        cursor.index = buffer.lines[cursor.line].text().len() - after_len;

        change.end = cursor;

        first = std::cmp::min(first, change.end);
        last = std::cmp::max(last, change.end);
        self.layout_range(font_system, buffer, first, last);

        (change.end, change)
    }

    #[must_use]
    fn delete_range(
        &mut self,
        font_system: &mut FontSystem,
        buffer: &mut Buffer,
        start: Cursor,
        end: Cursor,
        relayout: bool,
    ) -> Option<Change> {
        if start == end {
            return None;
        }

        let change_item = {
            // Collect removed data for change tracking
            let mut change_lines = Vec::new();

            // Delete the selection from the last line
            let end_line_opt = if end.line > start.line {
                // Get part of line after selection
                let after = buffer.lines[end.line].split_off(end.index);

                // Remove end line
                let removed = buffer.lines.remove(end.line);
                change_lines.insert(0, removed.text().to_string());

                Some(after)
            } else {
                None
            };

            // Delete interior lines (in reverse for safety)
            for line_i in (start.line + 1..end.line).rev() {
                let removed = buffer.lines.remove(line_i);
                change_lines.insert(0, removed.text().to_string());
            }

            // Delete the selection from the first line
            {
                // Get part after selection if start line is also end line
                let after_opt = if start.line == end.line {
                    Some(buffer.lines[start.line].split_off(end.index))
                } else {
                    None
                };

                // Delete selected part of line
                let removed = buffer.lines[start.line].split_off(start.index);
                change_lines.insert(0, removed.text().to_string());

                // Re-add part of line after selection
                if let Some(after) = after_opt {
                    buffer.lines[start.line].append(after);
                }

                // Re-add valid parts of end line
                if let Some(end_line) = end_line_opt {
                    buffer.lines[start.line].append(end_line);
                }
            }

            Change {
                start,
                end: start,
                old: change_lines.join("\n").as_bytes().into(),
                attrs: None,
            }
        };

        if relayout {
            self.layout_range(font_system, buffer, start, end);
        }
        Some(change_item)
    }

    pub fn copy_selection(&self, buffer: &Buffer) -> Option<String> {
        let (start, end) = self.selection_bounds(buffer)?;
        let mut selection = String::new();
        // Take the selection from the first line
        {
            // Add selected part of line to string
            if start.line == end.line {
                selection.push_str(&buffer.lines[start.line].text()[start.index..end.index]);
            } else {
                selection.push_str(&buffer.lines[start.line].text()[start.index..]);
                selection.push('\n');
            }
        }

        // Take the selection from all interior lines (if they exist)
        for line_i in start.line + 1..end.line {
            selection.push_str(buffer.lines[line_i].text());
            selection.push('\n');
        }

        // Take the selection from the last line
        if end.line > start.line {
            // Add selected part of line to string
            selection.push_str(&buffer.lines[end.line].text()[..end.index]);
        }

        Some(selection)
    }

    #[must_use]
    pub fn delete_selection(
        &mut self,
        font_system: &mut FontSystem,
        buffer: &mut Buffer,
    ) -> Option<Change> {
        let (start, end) = self.selection_bounds(buffer)?;

        // Reset cursor to start of selection
        self.set_cursor(buffer, start);

        // Reset selection to None
        self.selection = Selection::None;
        self.cursor_moved = true;

        // Delete from start to end of selection
        self.delete_range(font_system, buffer, start, end, true)
    }

    pub fn apply_change(
        &mut self,
        font_system: &mut FontSystem,
        buffer: &mut Buffer,
        changes: &[Change],
    ) -> SmallVec<[Change; 1]> {
        let mut reverse = SmallVec::new();
        for change in changes {
            let (cursor, undo) = self.replace_range(
                font_system,
                buffer,
                change.start,
                change.end,
                unsafe { str::from_utf8_unchecked(&change.old) },
                None,
            );
            self.set_cursor(buffer, cursor);
            reverse.push(undo);
        }

        self.shape_as_needed(font_system, buffer, false);
        reverse.reverse();
        reverse
    }

    pub fn layout_range(
        &mut self,
        font_system: &mut FontSystem,
        buffer: &mut Buffer,
        start: Cursor,
        end: Cursor,
    ) {
        for i in start.line..(end.line + 1) {
            buffer.line_layout(font_system, i);
        }
    }

    pub fn action(
        &mut self,
        font_system: &mut FontSystem,
        buffer: &mut Buffer,
        action: Action,
    ) -> SmallVec<[Change; 1]> {
        let change = match action {
            Action::TripleClick { .. } => panic!("Not supported"),
            Action::Motion(motion) => {
                let cursor = self.cursor;
                let cursor_x_opt = self.cursor_x_opt;
                if let Some((new_cursor, new_cursor_x_opt)) =
                    buffer.cursor_motion(font_system, cursor, cursor_x_opt, motion)
                {
                    self.set_cursor(buffer, new_cursor);
                    self.cursor_x_opt = new_cursor_x_opt;
                }
                SmallVec::new()
            }
            Action::Escape => {
                match self.selection {
                    Selection::None => {}
                    _ => buffer.set_redraw(true),
                }
                self.selection = Selection::None;
                self.cursor_moved = true;
                SmallVec::new()
            }
            Action::Insert(character) => {
                if character.is_control() && !['\t', '\n', '\u{92}'].contains(&character) {
                    // Filter out special chars (except for tab), use Action instead
                    panic!("use Action instead for special characters")
                } else if character == '\n' {
                    self.action(font_system, buffer, Action::Enter)
                } else {
                    let mut str_buf = [0u8; 8];
                    let str_ref = character.encode_utf8(&mut str_buf);
                    SmallVec::from_elem(self.insert_string(font_system, buffer, str_ref, None), 1)
                }
            }
            Action::Enter => {
                //TODO: what about indenting more after opening brackets or parentheses?
                let change = if self.auto_indent {
                    let mut string = String::from("\n");
                    let line = &buffer.lines[self.cursor.line];
                    let text = line.text();
                    for c in text.chars() {
                        if c.is_whitespace() {
                            string.push(c);
                        } else {
                            break;
                        }
                    }
                    self.insert_string(font_system, buffer, &string, None)
                } else {
                    self.insert_string(font_system, buffer, "\n", None)
                };

                SmallVec::from_elem(change, 1)
            }
            Action::Backspace => {
                if let Some(c) = self.delete_selection(font_system, buffer) {
                    SmallVec::from_elem(c, 1) // Deleted selection
                } else {
                    // Save current cursor as end
                    let end = self.cursor;

                    if self.cursor.index > 0 {
                        // Move cursor to previous character index
                        self.cursor.index = buffer.lines[self.cursor.line].text()
                            [..self.cursor.index]
                            .char_indices()
                            .next_back()
                            .map_or(0, |(i, _)| i);
                    } else if self.cursor.line > 0 {
                        // Move cursor to previous line
                        self.cursor.line -= 1;
                        self.cursor.index = buffer.lines[self.cursor.line].text().len();
                    }

                    self.delete_range(font_system, buffer, self.cursor, end, true)
                        .map(|c| SmallVec::from_elem(c, 1))
                        .unwrap_or_default()
                }
            }
            Action::Delete => {
                if let Some(c) = self.delete_selection(font_system, buffer) {
                    SmallVec::from_elem(c, 1) // Deleted selection
                } else {
                    // Save current cursor as start and end
                    let mut start = self.cursor;
                    let mut end = self.cursor;

                    if start.index < buffer.lines[start.line].text().len() {
                        let line = &buffer.lines[start.line];

                        let range_opt = line
                            .text()
                            .grapheme_indices(true)
                            .take_while(|(i, _)| *i <= start.index)
                            .last()
                            .map(|(i, c)| i..(i + c.len()));

                        if let Some(range) = range_opt {
                            start.index = range.start;
                            end.index = range.end;
                        }
                    } else if start.line + 1 < buffer.lines.len() {
                        end.line += 1;
                        end.index = 0;
                    }

                    self.set_cursor(buffer, start);
                    self.delete_range(font_system, buffer, start, end, true)
                        .map(|c| SmallVec::from_elem(c, 1))
                        .unwrap_or_default()
                }
            }
            Action::Indent => {
                // Get start and end of selection
                let (start, end) = match self.selection_bounds(buffer) {
                    Some(some) => some,
                    None => (self.cursor, self.cursor),
                };

                // For every line in selection
                let tab_width: usize = buffer.tab_width().into();
                let mut changes = SmallVec::new();
                for line_i in start.line..=end.line {
                    // Determine indexes of last indent and first character after whitespace
                    let mut after_whitespace;
                    let mut required_indent = 0;
                    let line = &buffer.lines[line_i];
                    let text = line.text();
                    // Default to end of line if no non-whitespace found
                    after_whitespace = text.len();
                    for (count, (index, c)) in text.char_indices().enumerate() {
                        if !c.is_whitespace() {
                            after_whitespace = index;
                            required_indent = tab_width - (count % tab_width);
                            break;
                        }
                    }

                    // No indent required (not possible?)
                    if required_indent == 0 {
                        required_indent = tab_width;
                    }

                    let location = Cursor::new(line_i, after_whitespace);
                    let (cursor, change) = self.replace_range(
                        font_system,
                        buffer,
                        location,
                        location,
                        &" ".repeat(required_indent),
                        None,
                    );
                    changes.push(change);
                    self.set_cursor(buffer, cursor);

                    // Adjust cursor
                    if self.cursor.line == line_i {
                        //TODO: should we be forcing cursor index to current indent location?
                        if self.cursor.index < after_whitespace {
                            self.cursor.index = after_whitespace;
                        }
                        self.cursor.index += required_indent;
                    }

                    // Adjust selection
                    match self.selection {
                        Selection::None => {}
                        Selection::Normal(ref mut select)
                        | Selection::Line(ref mut select)
                        | Selection::Word(ref mut select) => {
                            if select.line == line_i && select.index >= after_whitespace {
                                select.index += required_indent;
                            }
                        }
                    }

                    // Request redraw
                    buffer.set_redraw(true);
                }
                changes
            }
            Action::Unindent => {
                // Get start and end of selection
                let (start, end) = match self.selection_bounds(buffer) {
                    Some(some) => some,
                    None => (self.cursor, self.cursor),
                };

                // For every line in selection
                let tab_width: usize = buffer.tab_width().into();
                let mut changes = SmallVec::new();
                for line_i in start.line..=end.line {
                    // Determine indexes of last indent and first character after whitespace
                    let mut last_indent = 0;
                    let mut after_whitespace;
                    let line = &buffer.lines[line_i];
                    let text = line.text();
                    // Default to end of line if no non-whitespace found
                    after_whitespace = text.len();
                    for (count, (index, c)) in text.char_indices().enumerate() {
                        if !c.is_whitespace() {
                            after_whitespace = index;
                            break;
                        }
                        if count % tab_width == 0 {
                            last_indent = index;
                        }
                    }

                    // No de-indent required
                    if last_indent == after_whitespace {
                        continue;
                    }

                    // Delete one indent
                    if let Some(c) = self.delete_range(
                        font_system,
                        buffer,
                        Cursor::new(line_i, last_indent),
                        Cursor::new(line_i, after_whitespace),
                        true,
                    ) {
                        changes.push(c);
                    }

                    // Adjust cursor
                    if self.cursor.line == line_i && self.cursor.index > last_indent {
                        self.cursor.index -= after_whitespace - last_indent;
                    }

                    // Adjust selection
                    match self.selection {
                        Selection::None => {}
                        Selection::Normal(ref mut select)
                        | Selection::Line(ref mut select)
                        | Selection::Word(ref mut select) => {
                            if select.line == line_i && select.index > last_indent {
                                select.index -= after_whitespace - last_indent;
                            }
                        }
                    }

                    // Request redraw
                    buffer.set_redraw(true);
                }
                changes
            }
            Action::Click { x, y } => {
                self.set_selection(buffer, Selection::None);

                if let Some(new_cursor) = buffer.hit(x as f32, y as f32) {
                    if new_cursor != self.cursor {
                        self.set_cursor(buffer, new_cursor);
                        buffer.set_redraw(true);
                    }
                }
                SmallVec::new()
            }
            Action::DoubleClick { x, y } => {
                self.set_selection(buffer, Selection::None);

                if let Some(new_cursor) = buffer.hit(x as f32, y as f32) {
                    if new_cursor != self.cursor {
                        self.set_cursor(buffer, new_cursor);
                        buffer.set_redraw(true);
                    }
                    self.selection = Selection::Word(self.cursor);
                    buffer.set_redraw(true);
                }
                SmallVec::new()
            }
            Action::Drag { x, y } => {
                if let Some(new_cursor) = buffer.hit(x as f32, y as f32) {
                    // We do not trigger a selection if only the affinity changes, because it's not visible and ends up being confusing.
                    if new_cursor.index != self.cursor.index || new_cursor.line != self.cursor.line
                    {
                        if self.selection == Selection::None {
                            self.selection = Selection::Normal(self.cursor);
                            buffer.set_redraw(true);
                        }
                        self.set_cursor(buffer, new_cursor);
                        buffer.set_redraw(true);
                    }
                }
                SmallVec::new()
            }
            Action::Scroll { lines } => {
                let mut scroll = buffer.scroll();
                //TODO: align to layout lines
                scroll.vertical += lines as f32 * buffer.metrics().line_height;
                buffer.set_scroll(scroll);
                SmallVec::new()
            }
        };
        self.shape_as_needed(font_system, buffer, false);
        change
    }

    pub fn cursor_position(&self, buffer: &Buffer) -> Option<(i32, i32)> {
        buffer
            .layout_runs()
            .find_map(|run| cursor_position(&self.cursor, &run))
    }

    pub fn selection_bounds(&self, buffer: &Buffer) -> Option<(Cursor, Cursor)> {
        let cursor = self.cursor();
        match self.selection() {
            Selection::None => None,
            Selection::Normal(select) => match select.line.cmp(&cursor.line) {
                std::cmp::Ordering::Greater => Some((cursor, select)),
                std::cmp::Ordering::Less => Some((select, cursor)),
                std::cmp::Ordering::Equal => {
                    /* select.line == cursor.line */
                    if select.index < cursor.index {
                        Some((select, cursor))
                    } else {
                        /* select.index >= cursor.index */
                        Some((cursor, select))
                    }
                }
            },
            Selection::Line(select) => {
                let start_line = std::cmp::min(select.line, cursor.line);
                let end_line = std::cmp::max(select.line, cursor.line);
                let end_index = buffer.lines[end_line].text().len();
                Some((Cursor::new(start_line, 0), Cursor::new(end_line, end_index)))
            }
            Selection::Word(select) => {
                let (mut start, mut end) = match select.line.cmp(&cursor.line) {
                    std::cmp::Ordering::Greater => (cursor, select),
                    std::cmp::Ordering::Less => (select, cursor),
                    std::cmp::Ordering::Equal => {
                        /* select.line == cursor.line */
                        if select.index < cursor.index {
                            (select, cursor)
                        } else {
                            /* select.index >= cursor.index */
                            (cursor, select)
                        }
                    }
                };

                // Move start to beginning of word
                {
                    let line = &buffer.lines[start.line];
                    start.index = line
                        .text()
                        .unicode_word_indices()
                        .rev()
                        .map(|(i, _)| i)
                        .find(|&i| i < start.index)
                        .unwrap_or(0);
                }

                // Move end to end of word
                {
                    let line = &buffer.lines[end.line];
                    end.index = line
                        .text()
                        .unicode_word_indices()
                        .map(|(i, word)| i + word.len())
                        .find(|&i| i > end.index)
                        .unwrap_or(line.text().len());
                }

                Some((start, end))
            }
        }
    }

    /// Insert a string at the current cursor or replacing the current selection with the given
    /// attributes, or with the previous character's attributes if None is given.
    #[must_use]
    pub fn insert_string(
        &mut self,
        font_system: &mut FontSystem,
        buffer: &mut Buffer,
        data: &str,
        attrs_list: Option<AttrsList>,
    ) -> Change {
        let (cursor, change) = match self.selection_bounds(buffer) {
            Some((start, end)) => {
                self.replace_range(font_system, buffer, start, end, data, attrs_list)
            }
            None => self.replace_range(
                font_system,
                buffer,
                self.cursor(),
                self.cursor(),
                data,
                attrs_list,
            ),
        };
        self.set_cursor(buffer, cursor);
        change
    }
}
