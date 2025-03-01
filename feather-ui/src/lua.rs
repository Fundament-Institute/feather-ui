// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::basic::Basic;
use crate::layout::root;
use crate::layout::root::Root;
use crate::layout::simple;
use crate::layout::Desc;
use crate::outline::button::Button;
use crate::outline::region::Region;
use crate::outline::shader_standard::ShaderStandard;
use crate::outline::text::Text;
use crate::outline::window::Window;
use crate::outline::OutlineFrom;
use crate::DataID;
use crate::FnPersist;
use crate::Outline;
use crate::RelPoint;
use crate::Slot;
use crate::SourceID;
use crate::URect;
use mlua::prelude::*;
use mlua::UserData;
use std::rc::Rc;
use ultraviolet::Vec2;
use ultraviolet::Vec4;

pub type AppState = LuaValue;
type LuaSourceID = SourceID;

struct BoxedOutline<T: Desc>(Box<OutlineFrom<T>>);

impl<T: Desc> Clone for BoxedOutline<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

macro_rules! gen_from_lua {
    ($type_name:ident) => {
        impl mlua::FromLua for $type_name {
            #[inline]
            fn from_lua(value: ::mlua::Value, _: &::mlua::Lua) -> ::mlua::Result<Self> {
                match value {
                    ::mlua::Value::UserData(ud) => Ok(ud.borrow::<Self>()?.clone()),
                    _ => Err(::mlua::Error::FromLuaConversionError {
                        from: value.type_name(),
                        to: stringify!($type_name).to_string(),
                        message: None,
                    }),
                }
            }
        }
    };
}

impl UserData for Window {}
gen_from_lua!(Window);

impl UserData for SourceID {}
gen_from_lua!(SourceID);

impl UserData for Slot {}
gen_from_lua!(Slot);

impl UserData for URect {}
gen_from_lua!(URect);

//impl UserData for AppState<'_> {}
//gen_from_lua!(URect);

impl<T: Desc + 'static> UserData for BoxedOutline<T> {}
impl<T: Desc + 'static> mlua::FromLua for BoxedOutline<T> {
    #[inline]
    fn from_lua(value: ::mlua::Value, _: &::mlua::Lua) -> ::mlua::Result<Self> {
        match value {
            ::mlua::Value::UserData(ud) => Ok(ud.borrow::<BoxedOutline<T>>()?.clone()),
            _ => Err(::mlua::Error::FromLuaConversionError {
                from: value.type_name(),
                to: stringify!($type_name).to_string(),
                message: None,
            }),
        }
    }
}

fn create_id(_: &Lua, (id, _): (LuaValue, Option<LuaSourceID>)) -> mlua::Result<LuaSourceID> {
    Ok(crate::SourceID {
        // parent: parent.map(|x| Rc::downgrade(&x)).unwrap_or_default(),
        parent: None,
        id: if let Some(i) = id.as_integer() {
            DataID::Int(i)
        } else if let Some(s) = id.as_string_lossy() {
            DataID::Owned(s)
        } else {
            panic!("Invalid ID")
        },
    })
}

fn get_appdata_id(_: &Lua, (): ()) -> mlua::Result<LuaSourceID> {
    Ok(crate::APP_SOURCE_ID)
}

fn create_slot(_: &Lua, args: (Option<LuaSourceID>, u64)) -> mlua::Result<Slot> {
    if let Some(id) = args.0 {
        Ok(Slot(id.into(), args.1))
    } else {
        Ok(Slot(crate::APP_SOURCE_ID.into(), args.1))
    }
}

fn create_urect(_: &Lua, args: (f32, f32, f32, f32, f32, f32, f32, f32)) -> mlua::Result<URect> {
    Ok(URect {
        topleft: crate::UPoint {
            abs: Vec2::new(args.0, args.1),
            rel: RelPoint {
                x: args.4,
                y: args.5,
            },
        },
        bottomright: crate::UPoint {
            abs: Vec2::new(args.2, args.3),
            rel: RelPoint {
                x: args.6,
                y: args.7,
            },
        },
    })
}

fn create_window(_: &Lua, args: (LuaSourceID, String, BoxedOutline<Root>)) -> mlua::Result<Window> {
    Ok(Window::new(
        args.0.into(),
        winit::window::Window::default_attributes()
            .with_title(args.1)
            .with_resizable(true)
            .with_inner_size(winit::dpi::PhysicalSize::new(600, 400)),
        args.2 .0,
    ))
}

fn create_region(
    _: &Lua,
    args: (LuaSourceID, URect, Option<Vec<BoxedOutline<Basic>>>),
) -> mlua::Result<BoxedOutline<Root>> {
    let mut children = im::Vector::new();
    children.extend(args.2.unwrap().into_iter().map(|x| Some(x.0)));
    Ok(BoxedOutline(Box::new(Region {
        id: args.0.into(),
        props: root::Inherited { area: args.1 },
        basic: Basic {
            padding: Default::default(),
            zindex: 0,
        },
        children,
    })))
}

fn create_button(
    _: &Lua,
    args: (
        LuaSourceID,
        URect,
        String,
        Slot,
        [f32; 4],
        Option<BoxedOutline<simple::Simple>>,
    ),
) -> mlua::Result<BoxedOutline<Basic>> {
    let id = Rc::new(args.0);
    let rect = RoundRect::<()> {
        id: SourceID {
            parent: Some(id.clone()),
            id: DataID::Named("__internal_rect__"),
        }
        .into(),
        fill: args.4.into(),
        corners: Vec4::broadcast(10.0),
        props: (),
        rect: crate::FILL_URECT,
        ..Default::default()
    };

    let text = Text::<()> {
        id: SourceID {
            parent: Some(id.clone()),
            id: DataID::Named("__internal_text__"),
        }
        .into(),
        props: (),
        text: args.2,
        font_size: 30.0,
        line_height: 42.0,
        ..Default::default()
    };

    let mut children: im::Vector<Option<Box<OutlineFrom<simple::Simple>>>> = im::Vector::new();
    children.push_back(Some(Box::new(text)));
    children.push_back(Some(Box::new(rect)));
    if let Some(x) = args.5 {
        children.push_back(Some(x.0));
    }

    Ok(BoxedOutline(Box::new(Button::<()>::new(
        id,
        (),
        simple::Simple {
            area: args.1,
            margin: Default::default(),
            limits: crate::DEFAULT_LIMITS,
            anchor: Default::default(),
            zindex: 0,
        },
        args.3,
        children,
    ))))
}

fn create_label(_: &Lua, args: (LuaSourceID, URect, String)) -> mlua::Result<BoxedOutline<Basic>> {
    Ok(BoxedOutline(Box::new(Text::<()> {
        id: args.0.into(),
        // props: basic::Inherited {
        //     area: args.1,
        //     margin: Default::default(),
        //     limits: crate::DEFAULT_LIMITS,
        //     anchor: Default::default(),
        // },
        props: (),
        text: args.2,
        font_size: 30.0,
        line_height: 42.0,
        ..Default::default()
    })))
}
use crate::outline::round_rect::RoundRect;

#[allow(clippy::type_complexity)]
fn create_shader_standard(
    _: &Lua,
    args: (
        LuaSourceID,
        URect,
        String,
        [f32; 4],
        [f32; 4],
        [f32; 4],
        [f32; 4],
    ),
) -> mlua::Result<BoxedOutline<Basic>> {
    Ok(BoxedOutline(Box::new(ShaderStandard::<()> {
        id: args.0.into(),
        rect: args.1,
        fragment: args.2,
        props: (),
        uniforms: [args.3.into(), args.4.into(), args.5.into(), args.6.into()],
    })))
}

fn create_round_rect(
    _: &Lua,
    args: (LuaSourceID, URect, u32, f32, f32, u32),
) -> mlua::Result<BoxedOutline<Basic>> {
    let fill = args.2.to_be_bytes().map(|x| x as f32);
    let outline = args.5.to_be_bytes().map(|x| x as f32);
    Ok(BoxedOutline(Box::new(RoundRect::<()> {
        id: args.0.into(),
        rect: args.1,
        fill: Vec4::new(fill[0], fill[1], fill[2], fill[3]),
        outline: Vec4::new(outline[0], outline[1], outline[2], outline[3]),
        corners: Vec4::broadcast(args.3),
        border: args.4,
        props: (),
        blur: 0.0,
    })))
}

/// This defines the "lua" app that knows how to handle a lua value that contains the
/// expected rust objects, and hand them off for processing. This is analogous to the
/// pure-rust [App] struct defined in lib.rs
pub struct LuaApp {
    pub window: LuaFunction, // takes a Store and an appstate and returns a Window
    pub init: LuaFunction,
}

impl FnPersist<AppState, im::HashMap<Rc<SourceID>, Option<Window>>> for LuaApp {
    type Store = LuaValue;

    fn init(&self) -> Self::Store {
        let r = self.init.call::<LuaValue>(());
        match r {
            Err(LuaError::RuntimeError(s)) => panic!("{}", s),
            Err(e) => panic!("{:?}", e),
            Ok(v) => v,
        }
    }
    fn call(
        &self,
        store: Self::Store,
        args: &AppState,
    ) -> (Self::Store, im::HashMap<Rc<SourceID>, Option<Window>>) {
        let mut h = im::HashMap::new();
        let (store, w) = self
            .window
            .call::<(LuaValue, crate::outline::window::Window)>((store, args.clone()))
            .unwrap();
        h.insert(w.id().clone(), Some(w));
        (store, h)
    }
}

// These all map lua functions to rust code that creates the necessary rust objects and returns them inside lua userdata.
pub fn init_environment(lua: &Lua, tab: &mut mlua::Table) -> mlua::Result<()> {
    tab.set("create_id", lua.create_function(create_id)?)?;
    tab.set(
        "get_appdata_id",
        lua.create_function(|_: &Lua, (): ()| -> mlua::Result<LuaSourceID> {
            Ok(crate::APP_SOURCE_ID)
        })?,
    )?;
    tab.set("create_slot", lua.create_function(create_slot)?)?;
    tab.set("create_urect", lua.create_function(create_urect)?)?;
    tab.set("create_window", lua.create_function(create_window)?)?;
    tab.set("create_region", lua.create_function(create_region)?)?;
    tab.set("create_button", lua.create_function(create_button)?)?;
    tab.set("create_label", lua.create_function(create_label)?)?;
    tab.set("create_round_rect", lua.create_function(create_round_rect)?)?;
    tab.set(
        "create_shader_standard",
        lua.create_function(create_shader_standard)?,
    )?;

    Ok(())
}
