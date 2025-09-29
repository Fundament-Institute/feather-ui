// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use feather_ui::component::mouse_area;
use feather_ui::lua::LuaApp;
use feather_ui::mlua::Lua;
use feather_ui::{InputResult, WrapEventEx, handlers};

const LAYOUT: &[u8] = include_bytes!("./basic.lua");

#[derive(PartialEq, Clone, Debug, feather_macro::UserData)]
struct CounterState {
    count: i32,
}

fn main() {
    let lua = Lua::new();

    let onclick = |_: mouse_area::MouseAreaEvent,
                   mut appdata: feather_ui::AccessCell<CounterState>|
     -> InputResult<()> {
        {
            appdata.count += 1;
            InputResult::Forward(())
        }
    }
    .wrap();

    let (mut app, event_loop) = LuaApp::<CounterState, ()>::new(
        &lua,
        CounterState { count: 0 },
        handlers![CounterState, onclick],
        LAYOUT,
    )
    .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
