// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::StateMachine;
use crate::color::sRGB;
use crate::editor::Editor;
use crate::input::{ModifierKeys, MouseButton, MouseState, RawEvent, RawEventKind};
use crate::layout::{Layout, base, leaf};
use crate::text::{Change, EditBuffer};
use crate::{Dispatchable, Error, InputResult, PxRect, SourceID, layout};
use cosmic_text::{Action, Buffer, Cursor};
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use smallvec::SmallVec;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};

#[derive(Debug, Dispatch, EnumVariantType, Clone, PartialEq, Eq)]
#[evt(derive(Clone), module = "mouse_area_event")]
pub enum TextBoxEvent {
    Edit(SmallVec<[Change; 1]>),
}

struct TextBoxState {
    last_x_offset: Option<f32>, /* Last cursor x offset when something other than up or down
                                 * navigation happened */
    history: Vec<SmallVec<[Change; 1]>>,
    undo_index: usize,
    insert_mode: bool,
    text_count: AtomicUsize,
    cursor_count: AtomicUsize,
    focused: bool,
    editor: Editor,
    props: Rc<dyn Prop + 'static>,
    align: Option<cosmic_text::Align>,
}

impl TextBoxState {
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

impl super::EventRouter for TextBoxState {
    type Input = RawEvent;
    type Output = TextBoxEvent;

    fn process(
        mut this: crate::AccessCell<Self>,
        input: Self::Input,
        area: PxRect,
        _: PxRect,
        dpi: crate::RelDim,
        driver: &std::sync::Weak<crate::Driver>,
    ) -> InputResult<SmallVec<[Self::Output; 1]>> {
        let obj = this.props.textedit().obj.clone();
        let buffer = &mut obj.buffer.borrow_mut();
        let align = this.align;
        match Self::translate(input) {
            RawEvent::Focus { acquired, window } => {
                this.focused = acquired;
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
                    if down && let Some(driver) = driver.upgrade() {
                        let change = match named_key {
                            NamedKey::Enter => this.editor.action(
                                &mut driver.font_system.write(),
                                buffer,
                                Action::Enter,
                                align,
                            ),
                            NamedKey::Tab => this.editor.action(
                                &mut driver.font_system.write(),
                                buffer,
                                if (modifiers & ModifierKeys::Shift as u8) != 0 {
                                    Action::Unindent
                                } else {
                                    Action::Indent
                                },
                                align,
                            ),
                            NamedKey::Space => this.editor.action(
                                &mut driver.font_system.write(),
                                buffer,
                                Action::Insert(' '),
                                align,
                            ),
                            NamedKey::ArrowLeft
                            | NamedKey::ArrowRight
                            | NamedKey::ArrowDown
                            | NamedKey::ArrowUp
                            | NamedKey::End
                            | NamedKey::Home
                            | NamedKey::PageDown
                            | NamedKey::PageUp => {
                                let ctrl = (modifiers & ModifierKeys::Control as u8) != 0;
                                let shift = (modifiers & ModifierKeys::Shift as u8) != 0;
                                let font_system = &mut driver.font_system.write();
                                if !shift {
                                    let line_height = buffer.metrics().line_height;
                                    match (named_key, ctrl) {
                                        (NamedKey::ArrowUp, true) | (NamedKey::ArrowDown, true) => {
                                            this.editor.action(
                                                font_system,
                                                buffer,
                                                Action::Scroll {
                                                    pixels: if named_key == NamedKey::ArrowUp {
                                                        -line_height
                                                    } else {
                                                        line_height
                                                    },
                                                },
                                                align,
                                            );
                                            return InputResult::Consume(SmallVec::new());
                                        }
                                        _ => (),
                                    }

                                    if let Some((start, end)) = this.editor.selection_bounds(buffer)
                                    {
                                        if named_key == NamedKey::ArrowLeft {
                                            this.editor.set_cursor(buffer, start);
                                        } else if named_key == NamedKey::ArrowRight {
                                            this.editor.set_cursor(buffer, end);
                                        }
                                    }
                                    this.editor
                                        .action(font_system, buffer, Action::Escape, align);
                                } else if this.editor.selection() == cosmic_text::Selection::None {
                                    // if a selection doesn't exist, make one.
                                    let cursor = this.editor.cursor();
                                    this.editor.set_selection(
                                        buffer,
                                        cosmic_text::Selection::Normal(cursor),
                                    );
                                }
                                this.editor.action(
                                    font_system,
                                    buffer,
                                    Action::Motion(match (named_key, ctrl) {
                                        (NamedKey::ArrowLeft, false) => {
                                            cosmic_text::Motion::Previous
                                        }
                                        (NamedKey::ArrowRight, false) => cosmic_text::Motion::Next,
                                        (NamedKey::ArrowUp, false) => cosmic_text::Motion::Up,
                                        (NamedKey::ArrowDown, false) => cosmic_text::Motion::Down,
                                        (NamedKey::Home, false) => cosmic_text::Motion::Home,
                                        (NamedKey::End, false) => cosmic_text::Motion::End,
                                        (NamedKey::PageUp, false) => cosmic_text::Motion::PageUp,
                                        (NamedKey::PageDown, false) => {
                                            cosmic_text::Motion::PageDown
                                        }
                                        (NamedKey::ArrowLeft, true) => {
                                            cosmic_text::Motion::PreviousWord
                                        }
                                        (NamedKey::ArrowRight, true) => {
                                            cosmic_text::Motion::NextWord
                                        }
                                        (NamedKey::Home, true) => cosmic_text::Motion::BufferStart,
                                        (NamedKey::End, true) => cosmic_text::Motion::BufferEnd,
                                        _ => return InputResult::Consume(SmallVec::new()),
                                    }),
                                    align,
                                )
                            }
                            NamedKey::Select => {
                                // Represents a Select All operation
                                this.editor.set_selection(
                                    buffer,
                                    cosmic_text::Selection::Normal(Cursor {
                                        line: 0,
                                        index: 0,
                                        affinity: cosmic_text::Affinity::Before,
                                    }),
                                );
                                this.editor.action(
                                    &mut driver.font_system.write(),
                                    buffer,
                                    Action::Motion(cosmic_text::Motion::BufferEnd),
                                    align,
                                );
                                SmallVec::new()
                            }
                            NamedKey::Backspace => this.editor.action(
                                &mut driver.font_system.write(),
                                buffer,
                                Action::Backspace,
                                align,
                            ),
                            NamedKey::Delete => this.editor.action(
                                &mut driver.font_system.write(),
                                buffer,
                                Action::Delete,
                                align,
                            ),
                            NamedKey::Clear => {
                                let change = this
                                    .editor
                                    .delete_selection(&mut driver.font_system.write(), buffer)
                                    .map(|x| SmallVec::from_buf([x]))
                                    .unwrap_or_default();
                                this.editor.shape_as_needed(
                                    &mut driver.font_system.write(),
                                    buffer,
                                    false,
                                );
                                change
                            }
                            NamedKey::EraseEof => {
                                let cursor = this.editor.cursor();
                                this.editor
                                    .set_selection(buffer, cosmic_text::Selection::Normal(cursor));
                                this.editor.action(
                                    &mut driver.font_system.write(),
                                    buffer,
                                    Action::Motion(cosmic_text::Motion::BufferEnd),
                                    align,
                                );
                                let change = this
                                    .editor
                                    .delete_selection(&mut driver.font_system.write(), buffer)
                                    .map(|x| SmallVec::from_buf([x]))
                                    .unwrap_or_default();
                                this.editor.shape_as_needed(
                                    &mut driver.font_system.write(),
                                    buffer,
                                    false,
                                );
                                change
                            }
                            NamedKey::Insert => {
                                this.insert_mode = !this.insert_mode;
                                SmallVec::new()
                            }
                            NamedKey::Cut | NamedKey::Copy => {
                                if modifiers & ModifierKeys::Held as u8 == 0
                                    && let Some(s) = this.editor.copy_selection(buffer)
                                    && let Ok(mut clipboard) = arboard::Clipboard::new()
                                    && clipboard.set_text(&s).is_ok()
                                    && named_key == NamedKey::Cut
                                {
                                    let font_system = &mut driver.font_system.write();
                                    // Only delete the text for a cut command if the operation
                                    // succeeds
                                    if let Some(c) =
                                        this.editor.delete_selection(font_system, buffer)
                                    {
                                        this.editor.shape_as_needed(font_system, buffer, false);
                                        this.append(SmallVec::from_buf([c]))
                                    }
                                }
                                SmallVec::new()
                            }
                            NamedKey::Paste => {
                                if let Ok(mut clipboard) = arboard::Clipboard::new()
                                    && let Ok(s) = clipboard.get_text()
                                {
                                    let c = this.editor.insert_string(
                                        &mut driver.font_system.write(),
                                        buffer,
                                        &s,
                                        None,
                                        align,
                                    );
                                    this.editor.shape_as_needed(
                                        &mut driver.font_system.write(),
                                        buffer,
                                        false,
                                    );
                                    this.append(SmallVec::from_buf([c]))
                                }
                                SmallVec::new()
                            }
                            NamedKey::Redo => {
                                if this.undo_index > 0 {
                                    this.undo_index =
                                        this.redo(&mut driver.font_system.write(), buffer)
                                }
                                SmallVec::new()
                            }
                            NamedKey::Undo => {
                                if this.undo_index > 0 {
                                    this.undo_index =
                                        this.undo(&mut driver.font_system.write(), buffer)
                                }
                                SmallVec::new()
                            }
                            // Do not capture key events we don't recognize
                            _ => return InputResult::Forward(SmallVec::new()),
                        };

                        this.append(change);
                        obj.set_selection(
                            EditBuffer::from_cursor(buffer, this.editor.selection_or_cursor()),
                            EditBuffer::from_cursor(buffer, this.editor.cursor()),
                        );
                        return InputResult::Consume(SmallVec::new());
                    }
                    // Always capture the key event if we recognize it even if we don't do anything
                    // with it
                    return InputResult::Consume(SmallVec::new());
                }
                Key::Character(c) => {
                    if down {
                        if let Some(driver) = driver.upgrade() {
                            let c = this.editor.insert_string(
                                &mut driver.font_system.write(),
                                buffer,
                                &c,
                                None,
                                align,
                            );
                            this.append(SmallVec::from_buf([c]));

                            this.editor.shape_as_needed(
                                &mut driver.font_system.write(),
                                buffer,
                                false,
                            );
                            obj.set_selection(
                                EditBuffer::from_cursor(buffer, this.editor.selection_or_cursor()),
                                EditBuffer::from_cursor(buffer, this.editor.cursor()),
                            );
                        }
                        return InputResult::Consume(SmallVec::new());
                    }
                }
                _ => (),
            },
            RawEvent::MouseMove {
                pos, all_buttons, ..
            } => {
                if let Some(d) = driver.upgrade() {
                    *d.cursor.write() = winit::window::CursorIcon::Text;
                    let p =
                        area.topleft() + this.props.padding().resolve(dpi).topleft().to_vector();

                    if (all_buttons & MouseButton::Left as u16) != 0 {
                        this.editor.action(
                            &mut d.font_system.write(),
                            buffer,
                            Action::Drag {
                                x: (pos.x - p.x).round() as i32,
                                y: (pos.y - p.y).round() as i32,
                            },
                            align,
                        );
                    }
                }
                obj.set_selection(
                    EditBuffer::from_cursor(buffer, this.editor.selection_or_cursor()),
                    EditBuffer::from_cursor(buffer, this.editor.cursor()),
                );
                return InputResult::Consume(SmallVec::new());
            }
            RawEvent::Mouse {
                pos, state, button, ..
            } => {
                if let Some(d) = driver.upgrade() {
                    let p =
                        area.topleft() + this.props.padding().resolve(dpi).topleft().to_vector();

                    let action = match (state, button) {
                        (MouseState::Down, MouseButton::Left) => Action::Click {
                            x: (pos.x - p.x).round() as i32,
                            y: (pos.y - p.y).round() as i32,
                        },
                        (MouseState::DblClick, MouseButton::Left) => Action::DoubleClick {
                            x: (pos.x - p.x).round() as i32,
                            y: (pos.y - p.y).round() as i32,
                        },
                        _ => return InputResult::Consume(SmallVec::new()),
                    };
                    this.editor
                        .action(&mut d.font_system.write(), buffer, action, align);
                }
                obj.set_selection(
                    EditBuffer::from_cursor(buffer, this.editor.selection_or_cursor()),
                    EditBuffer::from_cursor(buffer, this.editor.cursor()),
                );
                return InputResult::Consume(SmallVec::new());
            }
            RawEvent::MouseScroll { delta, .. } => {
                if let Some(d) = driver.upgrade() {
                    let line_height = buffer.metrics().line_height;
                    match delta {
                        Ok(dist) => {
                            let mut scroll = buffer.scroll();
                            //TODO: align to layout lines
                            scroll.vertical += dist.y;
                            buffer.set_scroll(scroll);
                        }
                        Err(dist) => {
                            this.editor.action(
                                &mut d.font_system.write(),
                                buffer,
                                Action::Scroll {
                                    pixels: -dist.y * line_height,
                                },
                                align,
                            );
                        }
                    }
                }
            }
            _ => (),
        }
        InputResult::Forward(SmallVec::new())
    }
}

impl Clone for TextBoxState {
    fn clone(&self) -> Self {
        Self {
            last_x_offset: self.last_x_offset,
            history: self.history.clone(),
            undo_index: self.undo_index,
            insert_mode: self.insert_mode,
            text_count: self.text_count.load(Ordering::Relaxed).into(),
            cursor_count: self.cursor_count.load(Ordering::Relaxed).into(),
            focused: self.focused,
            editor: self.editor.clone(),
            props: self.props.clone(),
            align: self.align,
        }
    }
}

impl PartialEq for TextBoxState {
    fn eq(&self, other: &Self) -> bool {
        self.last_x_offset == other.last_x_offset
            && self.history == other.history
            && self.undo_index == other.undo_index
            && self.insert_mode == other.insert_mode
            && self.text_count.load(Ordering::Relaxed) == other.text_count.load(Ordering::Relaxed)
            && self.cursor_count.load(Ordering::Relaxed)
                == other.cursor_count.load(Ordering::Relaxed)
            && self.editor == other.editor
            && self.align == other.align
            && Rc::ptr_eq(&self.props, &other.props)
    }
}

pub trait Prop: leaf::Padded + base::TextEdit {}

#[derive_where(Clone)]
pub struct TextBox<T> {
    id: Arc<SourceID>,
    props: Rc<T>,
    pub font_size: f32,
    pub line_height: f32,
    pub font: cosmic_text::FamilyOwned,
    pub color: sRGB,
    pub weight: cosmic_text::Weight,
    pub style: cosmic_text::Style,
    pub wrap: cosmic_text::Wrap,
    pub align: Option<cosmic_text::Align>,
    pub slots: [Option<crate::Slot>; TextBoxEvent::SIZE],
}

impl TextBoxState {
    fn redo(&mut self, font_system: &mut cosmic_text::FontSystem, buffer: &mut Buffer) -> usize {
        // Redo the current Edit event (or execute cursor events until we find one) then
        // run all Cursor events after it until the next Edit event
        if self.undo_index < self.history.len() {
            self.history[self.undo_index] = self.editor.apply_change(
                font_system,
                buffer,
                &self.history[self.undo_index],
                self.align,
            );
            self.undo_index + 1
        } else {
            self.undo_index
        }
    }

    fn undo(&mut self, font_system: &mut cosmic_text::FontSystem, buffer: &mut Buffer) -> usize {
        if self.undo_index > 0 {
            self.history[self.undo_index - 1] = self.editor.apply_change(
                font_system,
                buffer,
                &self.history[self.undo_index - 1],
                self.align,
            );
            self.undo_index - 1
        } else {
            self.undo_index
        }
    }

    fn append(&mut self, change: SmallVec<[Change; 1]>) {
        self.history.truncate(self.undo_index);
        self.undo_index += 1;
        self.history.push(change);
    }
}

impl<T: Prop + 'static> TextBox<T> {
    pub fn new(
        id: Arc<SourceID>,
        props: T,
        font_size: f32,
        line_height: f32,
        font: cosmic_text::FamilyOwned,
        color: sRGB,
        weight: cosmic_text::Weight,
        style: cosmic_text::Style,
        wrap: cosmic_text::Wrap,
        align: Option<cosmic_text::Align>,
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
            align,
            slots: [None],
        }
    }
}

impl<T: Prop + 'static> crate::StateMachineChild for TextBox<T> {
    fn id(&self) -> Arc<SourceID> {
        self.id.clone()
    }

    fn init(
        &self,
        _: &std::sync::Weak<crate::Driver>,
    ) -> Result<Box<dyn super::StateMachineWrapper>, Error> {
        let statemachine = StateMachine {
            state: TextBoxState {
                editor: Editor::new(),
                last_x_offset: Default::default(),
                history: Default::default(),
                undo_index: Default::default(),
                insert_mode: Default::default(),
                text_count: Default::default(),
                cursor_count: Default::default(),
                focused: Default::default(),
                props: self.props.clone(),
                align: self.align,
            },
            input_mask: RawEventKind::Focus as u64
                | RawEventKind::Mouse as u64
                | RawEventKind::MouseMove as u64
                | RawEventKind::MouseScroll as u64
                | RawEventKind::Touch as u64
                | RawEventKind::Key as u64,
            output: self.slots.clone(),
            changed: true,
        };
        Ok(Box::new(statemachine))
    }
}

impl<T: Prop + 'static> super::Component for TextBox<T> {
    type Props = T;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        driver: &crate::graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        let dpi = manager
            .get::<super::window::WindowStateMachine>(window)
            .map(|x| x.state.dpi)
            .unwrap_or(crate::BASE_DPI);
        let mut font_system = driver.font_system.write();

        let textstate: &mut StateMachine<TextBoxState, { TextBoxEvent::SIZE }> =
            manager.get_mut(&self.id).unwrap();
        let textstate = &mut textstate.state;
        textstate.props = self.props.clone();
        textstate.align = self.align;

        if self.props.textedit().obj.reflow.load(Ordering::Acquire)
            || self.props.textedit().dpi != dpi
        {
            let attrs = cosmic_text::Attrs::new()
                .family(self.font.as_family())
                .color(self.color.into())
                .weight(self.weight)
                .style(self.style);
            self.props.textedit().obj.flowtext(
                &mut font_system,
                self.font_size,
                self.line_height,
                self.wrap,
                self.align,
                dpi,
                attrs,
            );
        }

        let instance = crate::render::textbox::Instance {
            text_buffer: self.props.textedit().obj.buffer.clone(),
            padding: self.props.padding().as_perimeter(dpi),
            selection: textstate
                .editor
                .selection_bounds(&self.props.textedit().obj.buffer.borrow()),
            color: self.color,
            cursor_color: if textstate.focused {
                self.color
            } else {
                sRGB::transparent()
            },
            cursor: textstate.editor.cursor(),
            selection_bg: sRGB::new(0.2, 0.2, 0.5, 1.0),
            selection_color: self.color,
            scale: 1.0,
        };

        Box::new(layout::text::Node::<T> {
            props: self.props.clone(),
            id: Arc::downgrade(&self.id),
            renderable: Rc::new(instance),
            buffer: self.props.textedit().obj.buffer.clone(),
            realign: self.align.is_some_and(|x| x != cosmic_text::Align::Left),
        })
    }
}
