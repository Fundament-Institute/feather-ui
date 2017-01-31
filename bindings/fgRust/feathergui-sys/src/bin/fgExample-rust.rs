extern crate feathergui_sys;

use feathergui_sys::*;
use feathergui_sys::FGVALUE::*;
use feathergui_sys::FG_MSGTYPE::*;
use std::{mem, ptr, thread};
use std::os::raw::{c_char, c_void};
use std::time::Duration;

static LAYOUT_XML: &'static str = include_str!("../../../../../media/feathertest.xml");

macro_rules! cstr {
    ($s:expr) => {
        concat!($s, "\0").as_ptr() as *const c_char
    }
}

unsafe extern "C" fn state_listener(this: *mut fgElement, _: *const FG_Msg) {
    let progbar = fgRoot_GetID(fgSingleton(), cstr!("#progbar"));
    fgFloatMessage(progbar,
                   FG_SETVALUE as u16,
                   FGVALUE_FLOAT as u16,
                   fgGetFloatMessage(this, FG_GETVALUE as u16, FGVALUE_FLOAT as u16, 0) /
                   fgGetFloatMessage(this, FG_GETVALUE as u16, FGVALUE_FLOAT as u16, 1),
                   0);
    let value = fgIntMessage(this, FG_GETVALUE as u16, 0, 0);
    let text = format!("{}\0", value);
    fgVoidMessage(progbar, FG_SETTEXT as u16, text.as_ptr() as *mut c_void, 0);
}

unsafe extern "C" fn make_pressed(this: *mut fgElement, _: *const FG_Msg) {
    fgVoidMessage(this, FG_SETTEXT as u16, cstr!("Pressed!") as *mut c_void, 0);
}

unsafe fn unsafe_main() {
    fgLoadBackend(cstr!("fgDirect2D.dll"));
    fgRegisterFunction(cstr!("statelistener"), Some(state_listener));
    fgRegisterFunction(cstr!("makepressed"), Some(make_pressed));

    let mut layout = mem::uninitialized();
    fgLayout_Init(&mut layout);
    fgLayout_LoadXML(&mut layout, LAYOUT_XML.as_ptr() as *const c_char, LAYOUT_XML.len() as u32);
    fgVoidMessage(&mut (*fgSingleton()).gui.element,
                  FG_LAYOUTLOAD as u16,
                  (&mut layout) as *mut _ as *mut c_void,
                  0);

    let tab_focus = fgRoot_GetID(fgSingleton(), cstr!("#tabfocus"));
    if tab_focus != ptr::null_mut() {
        let selected = fgIntMessage(tab_focus, FG_GETSELECTEDITEM as u16, 0, 0);
        fgIntMessage(selected as *mut _, FG_ACTION as u16, 0, 0);
    }

    let process = (*fgSingleton()).backend.fgProcessMessages.unwrap();
    while process() != 0 {
        // nothing to do
    }

    fgLayout_Destroy(&mut layout);
    fgUnloadBackend();
}

fn main() {
    unsafe {
        unsafe_main();
    }
}
