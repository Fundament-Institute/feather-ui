// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::StateMachine;
use crate::input::{ModifierKeys, MouseButton, MouseState, RawEvent, RawEventKind};
use crate::layout::{Layout, base, leaf};
use crate::text::EditObj;
use crate::{SourceID, WindowStateMachine, layout, point_to_pixel, render};
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use glyphon::Cursor;
use smallvec::SmallVec;
use std::ops::Range;
use std::rc::Rc;
use ultraviolet::Vec2;
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};

#[derive(Debug, Dispatch, EnumVariantType, Clone, PartialEq, Eq)]
#[evt(derive(Clone), module = "mouse_area_event")]
pub enum TextBoxEvent {
    Edit(SmallVec<[(Range<usize>, String); 1]>), // Returns the range of the new string and what the old string used to be.
    Cursor(usize, usize),
}

#[derive(Clone, Default)]
struct TextBoxState {
    last_x_offset: f32, // Last cursor x offset when something other than up or down navigation happened
    history: Vec<TextBoxEvent>,
    undo_index: usize,
    insert_mode: bool,
    count: usize,
    focused: bool,
    buffer: Rc<crate::RefCell<Option<glyphon::Buffer>>>,
}

impl PartialEq for TextBoxState {
    fn eq(&self, other: &Self) -> bool {
        self.last_x_offset == other.last_x_offset
            && self.history == other.history
            && self.undo_index == other.undo_index
            && self.insert_mode == other.insert_mode
            && self.count == other.count
            && Rc::ptr_eq(&self.buffer, &other.buffer)
    }
}

impl TextBoxState {
    fn redo(&mut self, textedit: &Rc<EditObj>) -> usize {
        let mut cur = self.undo_index;

        // Redo the current Edit event (or execute cursor events until we find one) then run all Cursor events after it until the next Edit event
        for (i, evt) in self.history[cur..].iter_mut().enumerate() {
            match evt {
                TextBoxEvent::Edit(multisplice) => {
                    // Swap the edit event here with the inverse
                    *multisplice = textedit.edit(multisplice);
                    cur += i;
                    break;
                }
                TextBoxEvent::Cursor(s, e) => textedit.set_selection((*s, *e)),
            }
        }

        for (i, evt) in self.history[cur..].iter().enumerate() {
            match evt {
                TextBoxEvent::Edit(_) => {
                    cur += i;
                    break;
                }
                TextBoxEvent::Cursor(s, e) => textedit.set_selection((*s, *e)),
            }
        }

        cur
    }

    fn undo(&mut self, textedit: &Rc<EditObj>) -> usize {
        let mut cur = self.undo_index;

        // Undo the current Edit event (or walk backwards until we find one), then run the cursor event immediately preceding it, if any.
        for (i, evt) in self.history[..cur].iter_mut().enumerate().rev() {
            cur = i;
            match evt {
                TextBoxEvent::Edit(multisplice) => {
                    // Swap the edit event here with the inverse
                    *multisplice = textedit.edit(multisplice);
                    break;
                }
                TextBoxEvent::Cursor(s, e) => textedit.set_selection((*s, *e)),
            }
        }

        if cur > 0 {
            if let TextBoxEvent::Cursor(s, e) = &self.history[cur - 1] {
                textedit.set_selection((*s, *e))
            }
        }

        cur
    }

    fn insert_into(mut self, textedit: &Rc<EditObj>, c: String) -> (Self, Vec<TextBoxEvent>) {
        let (mut start, mut end) = textedit.get_cursor();
        if end < start {
            std::mem::swap(&mut start, &mut end);
        }
        let old = textedit.text.borrow()[start..end].to_string();
        let len = c.len();
        textedit.edit(&[(start..end, c)]);
        if start == end {
            end = start + len;
        }
        textedit.set_cursor(end);
        // TODO: This hack may need to be replaced with an explicit method of saying "assume a change happened"
        self.count += 1;
        let evt = self
            .queue_history(&[
                TextBoxEvent::Edit([(start..end, old)].into()),
                TextBoxEvent::Cursor(end, end),
            ])
            .to_vec();
        (self, evt)
    }

    fn delete_selection(
        mut self,
        textedit: &Rc<EditObj>,
        mut start: usize,
        mut end: usize,
    ) -> (Self, Vec<TextBoxEvent>) {
        if end < start {
            std::mem::swap(&mut start, &mut end);
        }
        let old = textedit.text.borrow()[start..end].to_string();
        textedit.edit(&[(start..end, String::new())]);
        textedit.set_cursor(start);
        self.count += 1;
        let evt = self
            .queue_history(&[
                TextBoxEvent::Edit([(start..start, old)].into()),
                TextBoxEvent::Cursor(start, start),
            ])
            .to_vec();

        (self, evt)
    }

    fn modify_cursor(
        mut self,
        textedit: &Rc<EditObj>,
        cursor: usize,
        modifiers: u8,
    ) -> (Self, Vec<TextBoxEvent>) {
        let evt = if (modifiers & ModifierKeys::Shift as u8) != 0 {
            let (start, _) = textedit.get_cursor();
            textedit.set_selection((start, cursor));
            TextBoxEvent::Cursor(start, cursor)
        } else {
            textedit.set_cursor(cursor);
            TextBoxEvent::Cursor(cursor, cursor)
        };
        self.append_history(evt.clone());
        (self, vec![evt])
    }

    fn queue_history<'a>(&mut self, evt: &'a [TextBoxEvent]) -> &'a [TextBoxEvent] {
        for e in evt {
            self.append_history(e.clone());
        }
        evt
    }

    fn append_history(&mut self, evt: TextBoxEvent) {
        self.history.truncate(self.undo_index);
        self.history.push(evt);
        self.undo_index += 1;
        assert_eq!(
            self.undo_index,
            self.history.len(),
            "position out of sync???"
        );
    }
}
pub trait Prop: leaf::Padded + base::TextEdit {}

#[derive_where(Clone)]
pub struct TextBox<T: Prop + 'static> {
    id: Rc<SourceID>,
    props: Rc<T>,
    pub font_size: f32,
    pub line_height: f32,
    pub font: glyphon::FamilyOwned,
    pub color: glyphon::Color,
    pub weight: glyphon::Weight,
    pub style: glyphon::Style,
    pub wrap: glyphon::Wrap,
    pub slots: [Option<crate::Slot>; 3], // TODO: TextBoxEvent::SIZE
}

impl<T: Prop + 'static> TextBox<T> {
    pub fn new(
        id: Rc<SourceID>,
        props: T,
        font_size: f32,
        line_height: f32,
        font: glyphon::FamilyOwned,
        color: glyphon::Color,
        weight: glyphon::Weight,
        style: glyphon::Style,
        wrap: glyphon::Wrap,
    ) -> Self {
        Self {
            id: id.clone(),
            props: props.into(),
            font_size,
            line_height,
            font,
            color,
            weight,
            style,
            wrap,
            slots: [None, None, None],
        }
    }

    fn translate(e: RawEvent) -> RawEvent {
        match e {
            RawEvent::Key {
                device_id,
                physical_key: PhysicalKey::Code(key),
                location,
                down,
                logical_key,
                modifiers,
            } => {
                let k = match (key, down, modifiers & ModifierKeys::Control as u8 != 0) {
                    (KeyCode::KeyA, true, true) => Key::Named(NamedKey::Select),
                    (KeyCode::KeyC, true, true) => Key::Named(NamedKey::Copy),
                    (KeyCode::KeyX, true, true) => Key::Named(NamedKey::Cut),
                    (KeyCode::KeyV, true, true) => Key::Named(NamedKey::Paste),
                    (KeyCode::KeyZ, true, true) => Key::Named(NamedKey::Undo),
                    (KeyCode::KeyY, true, true) => Key::Named(NamedKey::Redo),
                    _ => logical_key,
                };
                RawEvent::Key {
                    device_id,
                    physical_key: PhysicalKey::Code(key),
                    location,
                    down,
                    logical_key: k,
                    modifiers,
                }
            }
            _ => e,
        }
    }
}

impl<T: Prop + 'static> crate::StateMachineChild for TextBox<T> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        let props = self.props.clone();
        let oninput = Box::new(crate::wrap_event::<RawEvent, TextBoxEvent, TextBoxState>(
            move |e, area, dpi, mut data| {
                let obj = &props.textedit().obj;
                match Self::translate(e) {
                    RawEvent::Focus { acquired, window } => {
                        data.focused = acquired;
                        window.set_ime_allowed(acquired);
                        if acquired {
                            window.set_ime_purpose(winit::window::ImePurpose::Normal);
                            //window.set_ime_cursor_area(position, size);
                        }
                    }
                    RawEvent::Key {
                        down,
                        logical_key,
                        modifiers,
                        ..
                    } => match logical_key {
                        Key::Named(named_key) => {
                            match named_key {
                                NamedKey::Enter | NamedKey::Tab | NamedKey::Space => {
                                    if down {
                                        let key = match named_key {
                                            NamedKey::Enter => "\n",
                                            NamedKey::Tab => "\t",
                                            NamedKey::Space => " ",
                                            _ => "",
                                        };
                                        return Ok(data.insert_into(obj, key.to_string()));
                                    }
                                }
                                NamedKey::ArrowDown => todo!(),
                                NamedKey::ArrowUp => todo!(),
                                NamedKey::ArrowLeft | NamedKey::ArrowRight => {
                                    if down {
                                        let (start, end) = obj.get_cursor();
                                        let cursor = if named_key == NamedKey::ArrowLeft {
                                            obj.prev_grapheme(end)
                                        } else {
                                            obj.next_grapheme(end)
                                        };
                                        data.count += 1;
                                        if (modifiers & ModifierKeys::Shift as u8) != 0 {
                                            return Ok(data.modify_cursor(obj, cursor, modifiers));
                                        } else {
                                            // If there's a selection, this goes to the start or end of the selection
                                            return Ok(match (named_key, start == end) {
                                                (NamedKey::ArrowLeft, true)
                                                | (NamedKey::ArrowRight, true) => {
                                                    data.modify_cursor(obj, cursor, 0)
                                                }
                                                (NamedKey::ArrowLeft, false) => {
                                                    data.modify_cursor(obj, start.min(end), 0)
                                                }
                                                (NamedKey::ArrowRight, false) => {
                                                    data.modify_cursor(obj, start.max(end), 0)
                                                }
                                                _ => (data, vec![]),
                                            });
                                        }
                                    }
                                }
                                NamedKey::End | NamedKey::Home => {
                                    if down {
                                        // Home and end operations are always relative to the end of the selection, wherever that is.
                                        let (_, end) = obj.get_cursor();
                                        let txt: &str = &obj.text.borrow();
                                        data.count += 1;

                                        return Ok(data.modify_cursor(
                                            obj,
                                            match (
                                                named_key == NamedKey::End,
                                                (modifiers & ModifierKeys::Control as u8) != 0,
                                            ) {
                                                (true, false) => {
                                                    txt[end..].find('\n').unwrap_or(txt.len())
                                                }
                                                (true, true) => txt.len(),
                                                (false, false) => {
                                                    txt[..end].rfind('\n').unwrap_or(0)
                                                }
                                                (false, true) => 0,
                                            },
                                            modifiers,
                                        ));
                                    }
                                }
                                NamedKey::Select => {
                                    // Represents a Select All operation
                                    if down {
                                        obj.set_selection((0, obj.text.borrow().len()));
                                        data.count += 1;
                                    }
                                }
                                NamedKey::PageDown => todo!(),
                                NamedKey::PageUp => todo!(),
                                NamedKey::Backspace | NamedKey::Delete | NamedKey::Clear => {
                                    if down {
                                        let (start, mut end) = obj.get_cursor();
                                        match (named_key, start == end) {
                                            (NamedKey::Backspace, true) => {
                                                end = obj.prev_grapheme(end)
                                            }
                                            (NamedKey::Delete, true) => {
                                                end = obj.next_grapheme(end)
                                            }
                                            _ => (), // Clear does nothing if there is no selection
                                        }
                                        if start != end {
                                            return Ok(data.delete_selection(obj, start, end));
                                        }
                                    }
                                }
                                NamedKey::EraseEof => {
                                    if down {
                                        let (start, end) = obj.get_cursor();
                                        return Ok(data.delete_selection(
                                            obj,
                                            start.min(end),
                                            obj.text.borrow().len(),
                                        ));
                                    }
                                }
                                NamedKey::Insert => {
                                    if down {
                                        data.insert_mode = !data.insert_mode;
                                    }
                                }
                                NamedKey::Cut | NamedKey::Copy => {
                                    if down && (modifiers & ModifierKeys::Held as u8 == 0) {
                                        let (mut start, mut end) = obj.get_cursor();
                                        if start != end {
                                            if end < start {
                                                std::mem::swap(&mut start, &mut end);
                                            }
                                            if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                                if clipboard
                                                    .set_text(&obj.text.borrow()[start..end])
                                                    .is_ok()
                                                    && named_key == NamedKey::Cut
                                                {
                                                    // Only delete the text for a cut command if the operation succeeds
                                                    return Ok(
                                                        data.delete_selection(obj, start, end)
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                NamedKey::Paste => {
                                    if down {
                                        if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                            if let Ok(s) = clipboard.get_text() {
                                                return Ok(data.insert_into(obj, s));
                                            }
                                        }
                                    }
                                }
                                NamedKey::Redo => {
                                    if down && data.undo_index > 0 {
                                        data.undo_index = data.redo(obj)
                                    }
                                }
                                NamedKey::Undo => {
                                    if down && data.undo_index > 0 {
                                        data.undo_index = data.undo(obj)
                                    }
                                }
                                // Do not capture key events we don't recognize
                                _ => return Err((data, vec![])),
                            }
                            // Always capture the key event if we recognize it even if we don't do anything with it
                            return Ok((data, vec![]));
                        }
                        Key::Character(c) => {
                            if down {
                                return Ok(data.insert_into(obj, c.to_string()));
                            }
                        }
                        _ => (),
                    },
                    RawEvent::MouseMove { driver, .. } => {
                        if let Some(d) = driver.upgrade() {
                            *d.cursor.write() = winit::window::CursorIcon::Text;
                        }
                        return Ok((data, vec![]));
                    }
                    RawEvent::Mouse {
                        pos,
                        state,
                        button,
                        modifiers,
                        ..
                    } => {
                        // TODO: Put in selection instead of just clicks
                        if state == MouseState::Down && button == MouseButton::Left {
                            let index = if let Some(buffer) = data.buffer.borrow().as_ref() {
                                let p = area.topleft() + props.padding().resolve(dpi).topleft();
                                let cursor =
                                    buffer.hit(pos.x - p.x, pos.y - p.y).unwrap_or_default();
                                buffer.lines[..cursor.line]
                                    .iter()
                                    .fold(cursor.index, |count, line| count + line.text().len())
                            } else {
                                return Ok((data, vec![]));
                            };
                            return Ok(data.modify_cursor(obj, index, modifiers));
                        }
                        return Ok((data, vec![]));
                    }
                    _ => (),
                }
                Err((data, vec![]))
            },
        ));

        let statemachine = StateMachine {
            state: Some(Default::default()),
            input: [(
                RawEventKind::Focus as u64
                    | RawEventKind::Mouse as u64
                    | RawEventKind::MouseMove as u64
                    | RawEventKind::Touch as u64
                    | RawEventKind::Key as u64,
                oninput,
            )],
            output: self.slots.clone(),
        };
        Ok(Box::new(statemachine))
    }
}

impl<T: Prop + 'static> super::Component<T> for TextBox<T> {
    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &crate::DriverState,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let winstate: &WindowStateMachine = state.get(window).unwrap();
        let winstate = winstate.state.as_ref().expect("No window state available");
        let dpi = winstate.dpi;
        let mut font_system = driver.font_system.write();

        let textstate: &StateMachine<TextBoxEvent, TextBoxState, 1, 3> =
            state.get(&self.id).unwrap();
        let textstate = textstate.state.as_ref().unwrap();

        if textstate.buffer.borrow().is_none() {
            *textstate.buffer.borrow_mut() = Some(glyphon::Buffer::new(
                &mut font_system,
                glyphon::Metrics::new(
                    point_to_pixel(self.font_size, dpi.x),
                    point_to_pixel(self.line_height, dpi.x),
                ),
            ));
        }
        let mut text_binding = textstate.buffer.borrow_mut();
        let text_buffer = text_binding.as_mut().unwrap();

        text_buffer.set_metrics(
            &mut font_system,
            glyphon::Metrics::new(
                point_to_pixel(self.font_size, dpi.x),
                point_to_pixel(self.line_height, dpi.x),
            ),
        );
        text_buffer.set_wrap(&mut font_system, self.wrap);
        text_buffer.set_text(
            &mut font_system,
            &self.props.textedit().obj.text.borrow(),
            &glyphon::Attrs::new()
                .family(self.font.as_family())
                .color(self.color)
                .weight(self.weight)
                .style(self.style),
            glyphon::Shaping::Advanced,
        );

        let renderer = glyphon::TextRenderer::new(
            &mut winstate.atlas.borrow_mut(),
            &driver.device,
            wgpu::MultisampleState::default(),
            None,
        );

        let textrender = Rc::new_cyclic(|this| render::text::Pipeline {
            this: this.clone(),
            text_buffer: textstate.buffer.clone(),
            renderer: renderer.into(),
            padding: self.props.padding().resolve(dpi).into(),
            atlas: winstate.atlas.clone(),
            viewport: winstate.viewport.clone(),
        });

        let line = super::line::build_pipeline(
            driver,
            config,
            (Vec2::zero(), Vec2::zero()),
            if textstate.focused {
                self.color.as_rgba().map(|x| x as f32).into()
            } else {
                ultraviolet::Vec4::zero()
            },
        );

        let pipeline = Pipeline {
            text: textrender.clone(),
            line,
            cursor: self.props.textedit().obj.get_cursor().1,
        };
        Box::new(layout::text::Node::<T> {
            props: self.props.clone(),
            id: Rc::downgrade(&self.id),
            text_render: textrender.clone(),
            renderable: Rc::new(pipeline),
        })
    }
}

crate::gen_component_wrap!(TextBox, Prop);

pub struct Pipeline {
    pub text: Rc<render::text::Pipeline>,
    pub line: Rc<render::line::Pipeline>,
    pub cursor: usize,
}

impl crate::render::Renderable for Pipeline {
    fn render(
        &self,
        area: crate::AbsRect,
        driver: &crate::DriverState,
    ) -> im::Vector<crate::RenderInstruction> {
        let mut offset = self.cursor;
        let buffer = self.text.text_buffer.borrow();
        let mut pos = self.text.padding.get().topleft();
        let mut cursor_height = 0.0;
        for line in buffer.as_ref().unwrap().layout_runs() {
            let len = line.text.len()
                + buffer.as_ref().unwrap().lines[line.line_i]
                    .ending()
                    .as_str()
                    .len();
            if len < offset {
                offset -= len;
                pos.y += line.line_height;
            } else {
                pos.x += line
                    .highlight(
                        Cursor::new(line.line_i, offset),
                        Cursor::new(line.line_i, offset),
                    )
                    .map(|(x, _)| x)
                    .unwrap_or(0.0);
                //line.line_top
                cursor_height = line.line_height;
                break;
            }
        }

        // TODO: current line pipeline ignores area
        *self.line.pos.borrow_mut() = (
            Vec2::new(pos.x, pos.y) + area.topleft(),
            Vec2::new(pos.x, pos.y + cursor_height) + area.topleft(),
        );
        let chain: [Rc<dyn crate::render::Renderable>; 2] = [self.text.clone(), self.line.clone()];
        chain.iter().flat_map(|x| x.render(area, driver)).collect()
    }
}
