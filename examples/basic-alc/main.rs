// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use feather_ui::lua::AppState;
use feather_ui::lua::LuaApp;
use feather_ui::App;
use mlua::prelude::*;
use mlua::Function;

fn wrap_luafunc<'lua>(
    f: Function<'lua>,
) -> impl FnMut(feather_ui::DispatchPair, AppState) -> Result<AppState, AppState> + 'lua {
    // Due to how mlua works, there is no safe way to ensure this lifetime is validated, so we just have
    // to cast the lifetime away and ensure the lua runtime is never closed while we're still using this.
    let f = unsafe { std::mem::transmute::<Function<'lua>, Function<'static>>(f) };
    move |pair, state| Ok(f.call((pair.0, state)).unwrap())
}

fn main() {
    let lua = unsafe { Lua::unsafe_new() };
    feather_ui::lua::init_environment(&lua).unwrap();
    let alicorn = Box::new(alicorn::Alicorn::new(Some(lua)).unwrap());

    // Because of constraints on lifetimes, this needs to technically last forever.
    let alicorn = Box::leak(alicorn);
    {
        let ast = alicorn.parse(include_str!("layout.alc")).unwrap();
        let terms = alicorn.process(ast).unwrap();
        let program = alicorn.evaluate(terms).unwrap();
        let (window, init, onclick): (Function, Function, Function) =
            alicorn.execute(program).unwrap();

        let onclick = Box::new(wrap_luafunc(onclick));
        let outline = LuaApp { window, init };
        let (mut app, event_loop): (App<AppState, LuaApp>, winit::event_loop::EventLoop<()>) =
            App::new(LuaNil, vec![onclick], outline).unwrap();

        event_loop.run_app(&mut app).unwrap();
    }
    //drop(unsafe { Box::from_raw(alicorn) });
}
