// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::component::button::Button;
use crate::component::region::Region;
use crate::component::shape::Shape;
use crate::component::text::Text;
use crate::component::window::Window;
use crate::component::{ComponentFrom, ComponentWrap};
use crate::layout::fixed;
use crate::propbag::PropBag;
use crate::{Component, DataID, FnPersist, Slot, SourceID, URect};
use mlua::prelude::*;
use mlua::UserData;
use std::rc::Rc;
use ultraviolet::Vec4;
use wide::f32x4;

pub type AppState = LuaValue;
type LuaSourceID = SourceID;

/*
#[allow(dead_code)]
impl LuaBag {
    pub fn contains(&self, key: &str) -> bool {
        self.props.contains_key(key).unwrap_or(false)
    }
    fn get_value<T: mlua::FromLua>(&self, key: &str) -> T {
        self.props.get(key).unwrap()
    }
    fn set_value<T: mlua::IntoLua>(&mut self, key: &str, v: T) -> bool {
        self.props.set(key, v).is_ok()
    }
}*/
/*
macro_rules! gen_lua_bag {
    ($prop:path, $name:ident, $t:ty) => {
        impl $prop for mlua::Table {
            fn $name(&self) -> &$t {
                &self
                    .get::<$t>(stringify!($name))
                    .expect(concat!("LuaBag didn't have ", stringify!($name)))
            }
        }
    };
}

macro_rules! gen_lua_bag_clone {
    ($prop:path, $name:ident, $t:ty) => {
        impl $prop for mlua::Table {
            fn $name(&self) -> $t {
                self.get::<$t>(stringify!($name))
                    .expect(concat!("PropBag didn't have ", stringify!($name)))
                    .clone()
            }
        }
    };
}

gen_lua_bag_clone!(crate::layout::base::Order, order, i64);
gen_lua_bag_clone!(crate::layout::base::ZIndex, zindex, i32);
gen_lua_bag_clone!(
    crate::layout::domain_write::Prop,
    domain,
    std::rc::Rc<crate::CrossReferenceDomain>
);

gen_lua_bag!(crate::layout::base::Area, area, crate::URect);
gen_lua_bag!(crate::layout::base::Padding, padding, crate::URect);
gen_lua_bag!(crate::layout::base::Margin, margin, crate::URect);
gen_lua_bag!(crate::layout::base::Limits, limits, crate::URect);
gen_lua_bag!(crate::layout::base::Anchor, anchor, crate::DPoint);

impl crate::layout::root::Prop for mlua::Table {
    fn dim(&self) -> &crate::AbsDim {
        let v = self
            .raw_get::<mlua::Value>("dim")
            .expect("LuaBag didn\'t have dim");

        if let ::mlua::Value::UserData(ud) = v {
            let r = ud.borrow::<Rc<crate::AbsDim>>().unwrap().clone();
            r.as_ref()
        } else {
            panic!("custom data isn't userdata???")
        }
    }
}

impl crate::layout::base::Empty for mlua::Table {}
impl crate::layout::leaf::Prop for mlua::Table {}
impl crate::layout::simple::Prop for mlua::Table {}

impl crate::layout::flex::Prop for mlua::Table {
    fn direction(&self) -> crate::layout::flex::FlexDirection {
        self.get("direction").unwrap()
    }

    fn wrap(&self) -> bool {
        self.get::<bool>("wrap").unwrap()
    }

    fn justify(&self) -> crate::layout::flex::FlexJustify {
        self.get::<u8>("justify").unwrap().try_into().unwrap()
    }

    fn align(&self) -> crate::layout::flex::FlexJustify {
        self.get::<u8>("align").unwrap()
    }
}

impl crate::layout::flex::Child for mlua::Table {
    fn grow(&self) -> f32 {
        self.get("grow").unwrap()
    }

    fn shrink(&self) -> f32 {
        self.get("shrink").unwrap()
    }

    fn basis(&self) -> f32 {
        self.get("basis").unwrap()
    }
}

impl crate::layout::base::Obstacles for mlua::Table {
    fn obstacles(&self) -> &[AbsRect] {
        &self.obstacles.get_or_init(|| {
            self.get::<Vec<AbsRect>>("obstacles")
                .expect("PropBag didn't have obstacles")
        })
    }
}
*/

type ComponentBag = Box<dyn crate::component::Component<PropBag>>;

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

impl UserData for ComponentBag {}
impl mlua::FromLua for ComponentBag {
    #[inline]
    fn from_lua(value: ::mlua::Value, _: &::mlua::Lua) -> ::mlua::Result<Self> {
        match value {
            ::mlua::Value::UserData(ud) => Ok(ud.borrow::<ComponentBag>()?.clone()),
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

#[allow(dead_code)]
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
        abs: crate::AbsRect(f32x4::new([args.0, args.1, args.2, args.3])),
        rel: crate::RelRect(f32x4::new([args.4, args.5, args.6, args.7])),
    })
}

fn create_window(_: &Lua, args: (LuaSourceID, String, ComponentBag)) -> mlua::Result<Window> {
    Ok(Window::new(
        args.0.into(),
        winit::window::Window::default_attributes()
            .with_title(args.1)
            .with_resizable(true)
            .with_inner_size(winit::dpi::PhysicalSize::new(600, 400)),
        Box::new(args.2),
    ))
}

fn create_region(
    _: &Lua,
    args: (LuaSourceID, URect, Option<Vec<ComponentBag>>),
) -> mlua::Result<ComponentBag> {
    let mut children = im::Vector::new();
    children.extend(
        args.2
            .unwrap()
            .into_iter()
            .map(|x| -> Option<Box<dyn ComponentWrap<dyn fixed::Child>>> { Some(Box::new(x)) }),
    );

    let mut bag = PropBag::new();
    bag.set_area(args.1.into());
    Ok(Box::new(Region::<PropBag> {
        id: args.0.into(),
        props: bag.into(),
        children,
    }))
}

fn create_button(
    _: &Lua,
    args: (
        LuaSourceID,
        URect,
        String,
        Slot,
        [f32; 4],
        Option<ComponentBag>,
    ),
) -> mlua::Result<ComponentBag> {
    let id = Rc::new(args.0);

    let rect = Shape::round_rect(
        SourceID {
            parent: Some(id.clone()),
            id: DataID::Named("__internal_rect__"),
        }
        .into(),
        crate::FILL_DRECT.into(),
        0.0,
        0.0,
        Vec4::broadcast(10.0),
        args.4.into(),
        Default::default(),
    );

    let text = Text::<crate::DRect> {
        id: SourceID {
            parent: Some(id.clone()),
            id: DataID::Named("__internal_text__"),
        }
        .into(),
        props: crate::FILL_DRECT.into(),
        text: args.2,
        font_size: 40.0,
        line_height: 56.0,
        ..Default::default()
    };

    let mut children: im::Vector<Option<Box<ComponentFrom<dyn fixed::Prop>>>> = im::Vector::new();
    children.push_back(Some(Box::new(text)));
    children.push_back(Some(Box::new(rect)));
    if let Some(x) = args.5 {
        children.push_back(Some(Box::new(x)));
    }

    let mut bag = PropBag::new();
    bag.set_area(args.1.into());
    Ok(Box::new(Button::<PropBag>::new(id, bag, args.3, children)))
}

fn create_label(_: &Lua, args: (LuaSourceID, URect, String)) -> mlua::Result<ComponentBag> {
    let mut bag = PropBag::new();
    bag.set_area(args.1.into());
    Ok(Box::new(Text::<PropBag> {
        id: args.0.into(),
        props: bag.into(),
        text: args.2,
        font_size: 40.0,
        line_height: 56.0,
        ..Default::default()
    }))
}

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
) -> mlua::Result<ComponentBag> {
    let mut bag = PropBag::new();
    bag.set_area(args.1.into());

    Ok(Box::new(Shape::<PropBag> {
        id: args.0.into(),
        fragment: std::borrow::Cow::Owned(args.2),
        props: bag.into(),
        label: "Custom Shader FS",
        uniforms: [args.3.into(), args.4.into(), args.5.into(), args.6.into()],
    }))
}

fn create_round_rect(
    _: &Lua,
    args: (LuaSourceID, URect, u32, f32, f32, u32),
) -> mlua::Result<ComponentBag> {
    let fill = args.2.to_be_bytes().map(|x| x as f32);
    let outline = args.5.to_be_bytes().map(|x| x as f32);
    let mut bag = PropBag::new();
    bag.set_area(args.1.into());
    Ok(Box::new(Shape::round_rect(
        args.0.into(),
        bag.into(),
        args.4,
        0.0,
        Vec4::broadcast(args.3),
        fill.into(),
        outline.into(),
    )))
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
            .call::<(LuaValue, crate::component::window::Window)>((store, args.clone()))
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
