// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use super::StateMachine;
use crate::color::sRGB;
use crate::editor::Editor;
use crate::input::{ModifierKeys, MouseButton, MouseState, RawEvent, RawEventKind};
use crate::layout::{Layout, base, leaf};
use crate::text::Change;
use crate::{Error, SourceID, WindowStateMachine, layout};
use cosmic_text::{Action, Cursor};
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use feather_macro::Dispatch;
use smallvec::SmallVec;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};

#[derive(Debug, Dispatch, EnumVariantType, Clone, PartialEq, Eq)]
#[evt(derive(Clone), module = "mouse_area_event")]
pub enum TextBoxEvent {
    Edit(SmallVec<[Change; 1]>),
}

struct TextBoxState {
    last_x_offset: Option<f32>, // Last cursor x offset when something other than up or down navigation happened
    history: Vec<SmallVec<[Change; 1]>>,
    undo_index: usize,
    insert_mode: bool,
    text_count: AtomicUsize,
    cursor_count: AtomicUsize,
    focused: bool,
    editor: Editor,
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
    }
}

impl TextBoxState {}
pub trait Prop: leaf::Padded + base::TextEdit {}

#[derive_where(Clone)]
pub struct TextBox<T: Prop + 'static> {
    id: Rc<SourceID>,
    props: Rc<T>,
    pub font_size: f32,
    pub line_height: f32,
    pub font: cosmic_text::FamilyOwned,
    pub color: sRGB,
    pub weight: cosmic_text::Weight,
    pub style: cosmic_text::Style,
    pub wrap: cosmic_text::Wrap,
    pub slots: [Option<crate::Slot>; 3], // TODO: TextBoxEvent::SIZE
}

impl TextBoxState {
    fn redo(&mut self, font_system: &mut cosmic_text::FontSystem) -> usize {
        // Redo the current Edit event (or execute cursor events until we find one) then run all Cursor events after it until the next Edit event
        if self.undo_index < self.history.len() {
            self.history[self.undo_index] = self
                .editor
                .apply_change(font_system, &self.history[self.undo_index]);
            self.undo_index + 1
        } else {
            self.undo_index
        }
    }

    fn undo(&mut self, font_system: &mut cosmic_text::FontSystem) -> usize {
        if self.undo_index > 0 {
            self.history[self.undo_index - 1] = self
                .editor
                .apply_change(font_system, &self.history[self.undo_index - 1]);
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
        id: Rc<SourceID>,
        props: T,
        font_size: f32,
        line_height: f32,
        font: cosmic_text::FamilyOwned,
        color: sRGB,
        weight: cosmic_text::Weight,
        style: cosmic_text::Style,
        wrap: cosmic_text::Wrap,
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

    fn init(
        &self,
        driver: &std::sync::Weak<crate::Driver>,
    ) -> Result<Box<dyn super::StateMachineWrapper>, Error> {
        let props = self.props.clone();
        let driver = driver.clone();
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
                            if down {
                                if let Some(driver) = driver.upgrade() {
                                    let change = match named_key {
                                        NamedKey::Enter => data
                                            .editor
                                            .action(&mut driver.font_system.write(), Action::Enter),
                                        NamedKey::Tab => data.editor.action(
                                            &mut driver.font_system.write(),
                                            if (modifiers & ModifierKeys::Shift as u8) != 0 {
                                                Action::Unindent
                                            } else {
                                                Action::Indent
                                            },
                                        ),
                                        NamedKey::Space => data.editor.action(
                                            &mut driver.font_system.write(),
                                            Action::Insert(' '),
                                        ),
                                        NamedKey::ArrowLeft
                                        | NamedKey::ArrowRight
                                        | NamedKey::ArrowDown
                                        | NamedKey::ArrowUp
                                        | NamedKey::End
                                        | NamedKey::Home
                                        | NamedKey::PageDown
                                        | NamedKey::PageUp => {
                                            let ctrl =
                                                (modifiers & ModifierKeys::Control as u8) != 0;
                                            let shift =
                                                (modifiers & ModifierKeys::Shift as u8) != 0;
                                            let font_system = &mut driver.font_system.write();
                                            if !shift {
                                                match (named_key, ctrl) {
                                                    (NamedKey::ArrowUp, true)
                                                    | (NamedKey::ArrowDown, true) => {
                                                        data.editor.action(
                                                            font_system,
                                                            Action::Scroll {
                                                                lines: if named_key
                                                                    == NamedKey::ArrowUp
                                                                {
                                                                    -1
                                                                } else {
                                                                    1
                                                                },
                                                            },
                                                        );
                                                        return Ok((data, vec![]));
                                                    }
                                                    _ => (),
                                                }

                                                if let Some((start, end)) =
                                                    data.editor.selection_bounds()
                                                {
                                                    if named_key == NamedKey::ArrowLeft {
                                                        data.editor.set_cursor(start);
                                                    } else if named_key == NamedKey::ArrowRight {
                                                        data.editor.set_cursor(end);
                                                    }
                                                }
                                                data.editor.action(font_system, Action::Escape);
                                            } else if data.editor.selection()
                                                == cosmic_text::Selection::None
                                            {
                                                // if a selection doesn't exist, make one.
                                                data.editor.set_selection(
                                                    cosmic_text::Selection::Normal(
                                                        data.editor.cursor(),
                                                    ),
                                                );
                                            }
                                            data.editor.action(
                                                font_system,
                                                Action::Motion(match (named_key, ctrl) {
                                                    (NamedKey::ArrowLeft, false) => {
                                                        cosmic_text::Motion::Previous
                                                    }
                                                    (NamedKey::ArrowRight, false) => {
                                                        cosmic_text::Motion::Next
                                                    }
                                                    (NamedKey::ArrowUp, false) => {
                                                        cosmic_text::Motion::Up
                                                    }
                                                    (NamedKey::ArrowDown, false) => {
                                                        cosmic_text::Motion::Down
                                                    }
                                                    (NamedKey::Home, false) => {
                                                        cosmic_text::Motion::Home
                                                    }
                                                    (NamedKey::End, false) => {
                                                        cosmic_text::Motion::End
                                                    }
                                                    (NamedKey::PageUp, false) => {
                                                        cosmic_text::Motion::PageUp
                                                    }
                                                    (NamedKey::PageDown, false) => {
                                                        cosmic_text::Motion::PageDown
                                                    }
                                                    (NamedKey::ArrowLeft, true) => {
                                                        cosmic_text::Motion::PreviousWord
                                                    }
                                                    (NamedKey::ArrowRight, true) => {
                                                        cosmic_text::Motion::NextWord
                                                    }
                                                    (NamedKey::Home, true) => {
                                                        cosmic_text::Motion::BufferStart
                                                    }
                                                    (NamedKey::End, true) => {
                                                        cosmic_text::Motion::BufferEnd
                                                    }
                                                    _ => return Ok((data, vec![])),
                                                }),
                                            )
                                        }
                                        NamedKey::Select => {
                                            // Represents a Select All operation
                                            data.editor.set_selection(
                                                cosmic_text::Selection::Normal(Cursor {
                                                    line: 0,
                                                    index: 0,
                                                    affinity: cosmic_text::Affinity::Before,
                                                }),
                                            );
                                            data.editor.action(
                                                &mut driver.font_system.write(),
                                                Action::Motion(cosmic_text::Motion::BufferEnd),
                                            );
                                            SmallVec::new()
                                        }
                                        NamedKey::Backspace => data.editor.action(
                                            &mut driver.font_system.write(),
                                            Action::Backspace,
                                        ),
                                        NamedKey::Delete => data.editor.action(
                                            &mut driver.font_system.write(),
                                            Action::Delete,
                                        ),
                                        NamedKey::Clear => {
                                            let change = data
                                                .editor
                                                .delete_selection(&mut driver.font_system.write())
                                                .map(|x| SmallVec::from_elem(x, 1))
                                                .unwrap_or_default();
                                            data.editor.shape_as_needed(
                                                &mut driver.font_system.write(),
                                                false,
                                            );
                                            change
                                        }
                                        NamedKey::EraseEof => {
                                            data.editor.set_selection(
                                                cosmic_text::Selection::Normal(
                                                    data.editor.cursor(),
                                                ),
                                            );
                                            data.editor.action(
                                                &mut driver.font_system.write(),
                                                Action::Motion(cosmic_text::Motion::BufferEnd),
                                            );
                                            let change = data
                                                .editor
                                                .delete_selection(&mut driver.font_system.write())
                                                .map(|x| SmallVec::from_elem(x, 1))
                                                .unwrap_or_default();
                                            data.editor.shape_as_needed(
                                                &mut driver.font_system.write(),
                                                false,
                                            );
                                            change
                                        }
                                        NamedKey::Insert => {
                                            data.insert_mode = !data.insert_mode;
                                            SmallVec::new()
                                        }
                                        NamedKey::Cut | NamedKey::Copy => {
                                            if modifiers & ModifierKeys::Held as u8 == 0 {
                                                if let Some(s) = data.editor.copy_selection() {
                                                    if let Ok(mut clipboard) =
                                                        arboard::Clipboard::new()
                                                    {
                                                        if clipboard.set_text(&s).is_ok()
                                                            && named_key == NamedKey::Cut
                                                        {
                                                            // Only delete the text for a cut command if the operation succeeds
                                                            if let Some(c) =
                                                                data.editor.delete_selection(
                                                                    &mut driver.font_system.write(),
                                                                )
                                                            {
                                                                data.editor.shape_as_needed(
                                                                    &mut driver.font_system.write(),
                                                                    false,
                                                                );
                                                                data.append(SmallVec::from_elem(
                                                                    c, 1,
                                                                ))
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            SmallVec::new()
                                        }
                                        NamedKey::Paste => {
                                            if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                                if let Ok(s) = clipboard.get_text() {
                                                    let c = data.editor.insert_string(
                                                        &mut driver.font_system.write(),
                                                        &s,
                                                        None,
                                                    );
                                                    data.editor.shape_as_needed(
                                                        &mut driver.font_system.write(),
                                                        false,
                                                    );
                                                    data.append(SmallVec::from_elem(c, 1))
                                                }
                                            }
                                            SmallVec::new()
                                        }
                                        NamedKey::Redo => {
                                            if data.undo_index > 0 {
                                                data.undo_index =
                                                    data.redo(&mut driver.font_system.write())
                                            }
                                            SmallVec::new()
                                        }
                                        NamedKey::Undo => {
                                            if data.undo_index > 0 {
                                                data.undo_index =
                                                    data.undo(&mut driver.font_system.write())
                                            }
                                            SmallVec::new()
                                        }
                                        // Do not capture key events we don't recognize
                                        _ => return Err((data, vec![])),
                                    };

                                    data.append(change);
                                    obj.set_selection(
                                        obj.from_cursor(data.editor.selection_or_cursor()),
                                        obj.from_cursor(data.editor.cursor()),
                                    );
                                    return Ok((data, vec![]));
                                }
                            }
                            // Always capture the key event if we recognize it even if we don't do anything with it
                            return Ok((data, vec![]));
                        }
                        Key::Character(c) => {
                            if down {
                                if let Some(driver) = driver.upgrade() {
                                    let c = data.editor.insert_string(
                                        &mut driver.font_system.write(),
                                        &c,
                                        None,
                                    );
                                    data.append(SmallVec::from_elem(c, 1));

                                    data.editor
                                        .shape_as_needed(&mut driver.font_system.write(), false);
                                    obj.set_selection(
                                        obj.from_cursor(data.editor.selection_or_cursor()),
                                        obj.from_cursor(data.editor.cursor()),
                                    );
                                }
                                return Ok((data, vec![]));
                            }
                        }
                        _ => (),
                    },
                    RawEvent::MouseMove {
                        pos, all_buttons, ..
                    } => {
                        if let Some(d) = driver.upgrade() {
                            *d.cursor.write() = winit::window::CursorIcon::Text;
                            let p = area.topleft() + props.padding().resolve(dpi).topleft();

                            if (all_buttons & MouseButton::Left as u16) != 0 {
                                data.editor.action(
                                    &mut d.font_system.write(),
                                    Action::Drag {
                                        x: (pos.x - p.x).round() as i32,
                                        y: (pos.y - p.y).round() as i32,
                                    },
                                );
                            }
                        }
                        obj.set_selection(
                            obj.from_cursor(data.editor.selection_or_cursor()),
                            obj.from_cursor(data.editor.cursor()),
                        );
                        return Ok((data, vec![]));
                    }
                    RawEvent::Mouse {
                        pos, state, button, ..
                    } => {
                        if let Some(d) = driver.upgrade() {
                            let p = area.topleft() + props.padding().resolve(dpi).topleft();
                            let action = match (state, button) {
                                (MouseState::Down, MouseButton::Left) => Action::Click {
                                    x: (pos.x - p.x).round() as i32,
                                    y: (pos.y - p.y).round() as i32,
                                },
                                (MouseState::DblClick, MouseButton::Left) => Action::DoubleClick {
                                    x: (pos.x - p.x).round() as i32,
                                    y: (pos.y - p.y).round() as i32,
                                },
                                _ => return Ok((data, vec![])),
                            };
                            data.editor.action(&mut d.font_system.write(), action);
                        }
                        obj.set_selection(
                            obj.from_cursor(data.editor.selection_or_cursor()),
                            obj.from_cursor(data.editor.cursor()),
                        );
                        return Ok((data, vec![]));
                    }
                    RawEvent::MouseScroll { delta, pixels, .. } => {
                        if let Some(d) = driver.upgrade() {
                            if pixels {
                                let mut buffer = data.editor.buffer_ref.borrow_mut();
                                let mut scroll = buffer.scroll();
                                //TODO: align to layout lines
                                scroll.vertical += delta.y;
                                buffer.set_scroll(scroll);
                            } else {
                                data.editor.action(
                                    &mut d.font_system.write(),
                                    Action::Scroll {
                                        lines: -(delta.y.round() as i32),
                                    },
                                );
                            }
                        }
                    }
                    _ => (),
                }
                Err((data, vec![]))
            },
        ));

        let statemachine = StateMachine {
            state: Some(TextBoxState {
                editor: Editor::new(self.props.textedit().obj.buffer.clone()),
                last_x_offset: Default::default(),
                history: Default::default(),
                undo_index: Default::default(),
                insert_mode: Default::default(),
                text_count: Default::default(),
                cursor_count: Default::default(),
                focused: Default::default(),
            }),
            input: [(
                RawEventKind::Focus as u64
                    | RawEventKind::Mouse as u64
                    | RawEventKind::MouseMove as u64
                    | RawEventKind::MouseScroll as u64
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
        driver: &crate::graphics::Driver,
        window: &Rc<SourceID>,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let winstate: &WindowStateMachine = state.get(window).unwrap();
        let winstate = winstate.state.as_ref().expect("No window state available");
        let dpi = winstate.dpi;
        let mut font_system = driver.font_system.write();

        let textstate: &StateMachine<TextBoxEvent, TextBoxState, 1, 3> =
            state.get(&self.id).unwrap();
        let textstate = textstate.state.as_ref().unwrap();
        if self.props.textedit().obj.reflow.load(Ordering::Acquire) {
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
                dpi,
                attrs,
            );
        }

        let instance = crate::render::textbox::Instance {
            text_buffer: textstate.editor.buffer_ref.clone(),
            padding: self.props.padding().resolve(dpi),
            selection: textstate.editor.selection_bounds(),
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
            id: Rc::downgrade(&self.id),
            renderable: Rc::new(instance),
            buffer: textstate.editor.buffer_ref.clone(),
        })
    }
}

crate::gen_component_wrap!(TextBox, Prop);
