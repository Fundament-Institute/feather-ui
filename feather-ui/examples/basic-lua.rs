// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use feather_ui::component::mouse_area;
use feather_ui::lua::LuaApp;
use feather_ui::{WrapEventEx, handlers};
use mlua::{FromLua, Lua, UserData, UserDataFields};

const LAYOUT: &[u8] = include_bytes!("./basic.lua");

#[derive(PartialEq, Clone, Debug)]
struct CounterState {
    count: i32,
}

impl UserData for CounterState {
    fn add_fields<F: UserDataFields<Self>>(f: &mut F) {
        f.add_field_method_get("count", |_, this| Ok(this.count));
    }
}

impl FromLua for CounterState {
    #[inline]
    fn from_lua(value: ::mlua::Value, _: &::mlua::Lua) -> ::mlua::Result<Self> {
        match value {
            ::mlua::Value::UserData(ud) => Ok(ud.borrow::<Self>()?.clone()),
            _ => Err(::mlua::Error::FromLuaConversionError {
                from: value.type_name(),
                to: stringify!(CounterState).to_string(),
                message: None,
            }),
        }
    }
}

fn main() {
    let lua = Lua::new();

    let onclick = |_: mouse_area::MouseAreaEvent,
                   mut appdata: CounterState|
     -> Result<CounterState, CounterState> {
        {
            appdata.count += 1;
            Ok(appdata)
        }
    }
    .wrap();

    let (mut app, event_loop) = LuaApp::<CounterState>::new::<()>(
        &lua,
        CounterState { count: 0 },
        handlers![CounterState, onclick],
        LAYOUT,
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
