use crate::component::window::Window;
use crate::component::ComponentFrom;
use crate::layout::basic::Basic;
use crate::layout::root::Root;
use crate::Component;
use crate::EventHandler;
use mlua::prelude::*;
use winit::window::WindowBuilder;

pub type AppState<'lua> = LuaValue<'lua>;

fn create_window<'lua>(
    _: &'lua Lua,
    args: (String, Box<ComponentFrom<AppState<'lua>, Root>>),
) -> mlua::Result<Box<dyn Component<LuaValue<'lua>, ()>>> {
    Ok(Box::new(Window::new(
        &instance,
        WindowBuilder::new()
            .with_title(args.0)
            .with_inner_size(winit::dpi::PhysicalSize::new(600, 400)),
        &event_loop,
        args.1,
    )))
}

fn create_area<'lua>(
    _: &'lua Lua,
    args: (URect, Vec<Box<ComponentFrom<AppState<'lua>, Root>>>),
) -> mlua::Result<Box<dyn Component<LuaValue<'lua>, ()>>> {
}

fn create_button<'lua>(
    _: &'lua Lua,
    args: (String, URect, Box<ComponentFrom<AppState<'lua>, Basic>>),
) -> mlua::Result<Box<dyn Component<LuaValue<'lua>, ()>>> {
}

fn create_label<'lua>(
    _: &'lua Lua,
    args: (String, URect),
) -> mlua::Result<Box<dyn Component<LuaValue<'lua>, ()>>> {
}

fn create_round_rect<'lua>(
    _: &'lua Lua,
    args: (String, URect),
) -> mlua::Result<Box<dyn Component<LuaValue<'lua>, ()>>> {
}

// Expects function(Event, AbsRect, AppData)
fn create_event_handler<'lua>(
    _: &'lua Lua,
    args: LuaFunction<'lua>,
) -> mlua::Result<EventHandler<AppState<'lua>>> {
}

#[test]
fn lua_test() {
    let lua = Lua::new();
    lua.create_function(|_, x| Ok(()))
}
