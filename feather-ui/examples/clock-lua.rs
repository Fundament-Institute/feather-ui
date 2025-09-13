// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use std::sync::atomic::{AtomicUsize, Ordering};

use feather_ui::lua::LuaApp;
use mlua::{FromLua, Lua, UserData, UserDataFields};

const LAYOUT: &[u8] = include_bytes!("./clock.lua");

#[derive(Debug)]
struct TimeState {
    count: AtomicUsize,
}

impl PartialEq for TimeState {
    fn eq(&self, other: &Self) -> bool {
        self.count.load(Ordering::Relaxed) == other.count.load(Ordering::Relaxed)
    }
}

impl Clone for TimeState {
    fn clone(&self) -> Self {
        Self {
            count: self.count.load(Ordering::Relaxed).into(),
        }
    }
}

impl UserData for TimeState {
    fn add_fields<F: UserDataFields<Self>>(f: &mut F) {
        f.add_field_method_get("time_hour", |_, s| {
            s.count.fetch_add(1, Ordering::Relaxed);
            let day = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                % (24 * 60 * 60);
            Ok(day / (60 * 60))
        });
        f.add_field_method_get("time_min", |_, s| {
            s.count.fetch_add(1, Ordering::Relaxed);
            let day = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                % (24 * 60 * 60);
            Ok((day / 60) % 60)
        });
        f.add_field_method_get("time_sec", |_, s| {
            s.count.fetch_add(1, Ordering::Relaxed);
            let day = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                % (24 * 60 * 60);
            Ok(day % 60)
        });
    }
}

impl FromLua for TimeState {
    #[inline]
    fn from_lua(value: ::mlua::Value, _: &::mlua::Lua) -> ::mlua::Result<Self> {
        match value {
            ::mlua::Value::UserData(ud) => Ok(ud.borrow::<Self>()?.clone()),
            _ => Err(::mlua::Error::FromLuaConversionError {
                from: value.type_name(),
                to: stringify!(TimeState).to_string(),
                message: None,
            }),
        }
    }
}

fn main() {
    let lua = Lua::new();

    let (mut app, event_loop) =
        LuaApp::<TimeState>::new(&lua, TimeState { count: 0.into() }, Vec::new(), LAYOUT).unwrap();

    event_loop.run_app(&mut app).unwrap();
}
