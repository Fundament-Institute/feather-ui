// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::component::ChildOf;
use crate::component::button::Button;
use crate::component::region::Region;
use crate::component::shape::{Shape, ShapeKind};
use crate::component::text::Text;
use crate::component::window::Window;
use crate::layout::fixed;
use crate::propbag::PropBag;
use crate::{
    AbsRect, DAbsPoint, DAbsRect, DPoint, DRect, DValue, DataID, FnPersist, RelPoint, RelRect,
    Slot, SourceID, StateMachineChild, URect, ZERO_RECT,
};
use mlua::UserData;
use mlua::prelude::*;
use std::f32;
use std::sync::Arc;
use wide::f32x4;

pub type AppState = LuaValue;
type LuaSourceID = SourceID;

fn get_key<V: FromLua>(t: &LuaTable, key: &str) -> LuaResult<Option<V>> {
    if t.contains_key(key)? {
        Ok(Some(t.get(key)?))
    } else {
        Ok(None)
    }
}

fn get_default<V: FromLua + Default>(t: &LuaTable, key: &str) -> LuaResult<V> {
    Ok(get_key(t, key)?.unwrap_or_default())
}

fn is_dvalue(t: &LuaTable) -> LuaResult<bool> {
    if let Some(mt) = t.metatable() {
        mt.contains_key("__isvalue")
    } else {
        Ok(false)
    }
}

impl FromLua for DValue {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let t = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        if !is_dvalue(t)? {
            return Err(LuaError::UserDataTypeMismatch);
        }

        Ok(DValue {
            dp: get_default(t, "dp")?,
            px: get_default(t, "px")?,
            rel: get_default(t, "rel")?,
        })
    }
}

impl FromLua for AbsPoint {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        if let Some(v) = value.as_number() {
            return Ok(AbsPoint(Vec2::broadcast(v as f32)));
        }

        let t = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        if is_dvalue(t)? {
            if t.contains_key("dp")? && !t.contains_key("px")? && !t.contains_key("rel")? {
                Ok(AbsPoint(Vec2::broadcast(t.get("dp")?)))
            } else {
                Err(LuaError::UserDataTypeMismatch)
            }
        } else {
            Ok(AbsPoint(Vec2::new(t.get("x")?, t.get("y")?)))
        }
    }
}

impl FromLua for RelPoint {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        if let Some(v) = value.as_number() {
            return Ok(RelPoint(Vec2::broadcast(v as f32)));
        }

        let t = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        if is_dvalue(t)? {
            if !t.contains_key("dp")? && !t.contains_key("px")? && t.contains_key("rel")? {
                Ok(RelPoint(Vec2::broadcast(t.get("rel")?)))
            } else {
                Err(LuaError::UserDataTypeMismatch)
            }
        } else {
            Ok(RelPoint(Vec2::new(t.get("x")?, t.get("y")?)))
        }
    }
}

impl FromLua for AbsRect {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        Ok(AbsRect(f32x4::new(
            if v.contains_key("x")? || v.contains_key("y")? {
                let x = getf32(v, "x")?;
                let y = getf32(v, "y")?;
                [x, y, x, y]
            } else {
                [
                    getf32(v, "left")?,
                    getf32(v, "top")?,
                    getf32(v, "right")?,
                    getf32(v, "bottom")?,
                ]
            },
        )))
    }
}

impl FromLua for RelRect {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        Ok(RelRect(f32x4::new(
            if v.contains_key("x")? || v.contains_key("y")? {
                let x = getf32(v, "x")?;
                let y = getf32(v, "y")?;
                [x, y, x, y]
            } else {
                [
                    getf32(v, "left")?,
                    getf32(v, "top")?,
                    getf32(v, "right")?,
                    getf32(v, "bottom")?,
                ]
            },
        )))
    }
}

fn get_value<T: FromLua + Default>(t: &LuaTable, key: &str) -> LuaResult<T> {
    if t.contains_key(key)? {
        t.get(key)
    } else {
        Ok(Default::default())
    }
}

impl FromLua for DAbsRect {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        let px = get_value::<AbsRect>(v, "px")?;
        let dp = get_value::<AbsRect>(v, "dp")?;
        Ok(DAbsRect { dp, px })
    }
}

impl FromLua for DRect {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        let px = get_value::<AbsRect>(v, "px")?;
        let dp = get_value::<AbsRect>(v, "dp")?;
        let rel = get_value::<RelRect>(v, "rel")?;
        Ok(DRect { dp, px, rel })
    }
}

impl FromLua for DAbsPoint {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        let px = get_value::<AbsPoint>(v, "px")?.0;
        let dp = get_value::<AbsPoint>(v, "dp")?.0;
        Ok(DAbsPoint { dp, px })
    }
}

impl FromLua for DPoint {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        let px = get_value::<AbsPoint>(v, "px")?.0;
        let dp = get_value::<AbsPoint>(v, "dp")?.0;
        let rel = get_value::<RelPoint>(v, "rel")?;
        Ok(DPoint { dp, px, rel })
    }
}

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
        impl $prop for LuaTable {
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
        impl $prop for LuaTable {
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

impl crate::layout::root::Prop for LuaTable {
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

impl crate::layout::base::Empty for LuaTable {}
impl crate::layout::leaf::Prop for LuaTable {}
impl crate::layout::simple::Prop for LuaTable {}

impl crate::layout::flex::Prop for LuaTable {
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

impl crate::layout::flex::Child for LuaTable {
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

impl crate::layout::base::Obstacles for LuaTable {
    fn obstacles(&self) -> &[AbsRect] {
        &self.obstacles.get_or_init(|| {
            self.get::<Vec<AbsRect>>("obstacles")
                .expect("PropBag didn't have obstacles")
        })
    }
}
*/

type ComponentBag = Box<dyn crate::component::Component<Props = PropBag>>;

impl<U: ?Sized> crate::component::ComponentWrap<U> for ComponentBag
where
    for<'a> &'a U: std::convert::From<&'a PropBag>,
{
    fn layout(
        &self,
        manager: &mut crate::StateManager,
        driver: &crate::graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn crate::layout::Layout<U> + 'static> {
        use std::ops::Deref;
        Box::new(Box::deref(self).layout(manager, driver, window))
    }
}

impl StateMachineChild for ComponentBag {
    fn id(&self) -> Arc<SourceID> {
        use std::ops::Deref;
        Box::deref(self).id()
    }

    fn init(
        &self,
        driver: &std::sync::Weak<crate::graphics::Driver>,
    ) -> Result<Box<dyn crate::component::StateMachineWrapper>, crate::Error> {
        use std::ops::Deref;
        Box::deref(self).init(driver)
    }

    fn apply_children(
        &self,
        f: &mut dyn FnMut(&dyn StateMachineChild) -> eyre::Result<()>,
    ) -> eyre::Result<()> {
        use std::ops::Deref;
        Box::deref(self).apply_children(f)
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
        parent: crate::OnceCell::new(),
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
    children.extend(args.2.unwrap().into_iter().map(
        |x| -> Option<Box<dyn crate::component::ComponentWrap<dyn fixed::Child>>> {
            Some(Box::new(x))
        },
    ));

    let mut bag = PropBag::new();
    bag.set_area(args.1.into());
    Ok(Box::new(Region::<PropBag>::new(
        args.0.into(),
        bag.into(),
        children,
    )))
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
    let id = Arc::new(args.0);

    let rect = Shape::<DRect, { ShapeKind::RoundRect as u8 }>::new(
        SourceID {
            parent: id.clone().into(),
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

    let text = Text::<DRect> {
        id: SourceID {
            parent: id.clone().into(),
            id: DataID::Named("__internal_text__"),
        }
        .into(),
        props: crate::FILL_DRECT.into(),
        text: args.2,
        font_size: 40.0,
        line_height: 56.0,
        ..Default::default()
    };

    let mut children: im::Vector<Option<Box<crate::component::ChildOf<dyn fixed::Prop>>>> =
        im::Vector::new();
    children.push_back(Some(Box::new(rect)));
    children.push_back(Some(Box::new(text)));
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

#[allow(unused_variables)]
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
    todo!();
}

fn create_round_rect(
    _: &Lua,
    args: (LuaSourceID, URect, u32, f32, f32, u32),
) -> mlua::Result<ComponentBag> {
    let fill = args.2.to_be_bytes().map(|x| x as f32);
    let outline = args.5.to_be_bytes().map(|x| x as f32);
    let mut bag = PropBag::new();
    bag.set_area(args.1.into());
    Ok(Box::new(
        Shape::<PropBag, { ShapeKind::RoundRect as u8 }>::new(
            args.0.into(),
            bag.into(),
            args.4,
            0.0,
            Vec4::broadcast(args.3),
            fill.into(),
            outline.into(),
        ),
    ))
}

/// This defines the "lua" app that knows how to handle a lua value that contains the
/// expected rust objects, and hand them off for processing. This is analogous to the
/// pure-rust [App] struct defined in lib.rs
pub struct LuaApp {
    pub window: LuaFunction, // takes a Store and an appstate and returns a Window
    pub init: LuaFunction,
}

impl FnPersist<AppState, im::HashMap<Arc<SourceID>, Option<Window>>> for LuaApp {
    type Store = LuaValue;

    fn init(&self) -> Self::Store {
        let r = self.init.call::<LuaValue>(());
        match r {
            Err(LuaError::RuntimeError(s)) => panic!("{}", s),
            Err(e) => panic!("{e:?}"),
            Ok(v) => v,
        }
    }
    fn call(
        &mut self,
        store: Self::Store,
        args: &AppState,
    ) -> (Self::Store, im::HashMap<Arc<SourceID>, Option<Window>>) {
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
pub fn init_environment(lua: &Lua, tab: &mut LuaTable) -> mlua::Result<()> {
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

fn gather_props(props: &LuaTable) -> mlua::Result<PropBag> {
    let mut bag = PropBag::new();

    if props.contains_key("domain")? {
        let area: std::sync::Arc<crate::CrossReferenceDomain> = props.get("domain")?;
    }
    if props.contains_key("direction")? {
        let area: crate::RowDirection = props.get("direction")?;
    }
    if props.contains_key("wrap")? {
        let area: bool = props.get("wrap")?;
    }
    if props.contains_key("justify")? {
        let area: crate::layout::flex::FlexJustify = props.get("justify")?;
    }
    if props.contains_key("align")? {
        let area: crate::layout::flex::FlexJustify = props.get("align")?;
    }
    if props.contains_key("zindex")? {
        let area: i32 = props.get("zindex")?;
    }
    if props.contains_key("obstacles")? {
        let area: Vec<DAbsRect> = props.get("obstacles")?;
    }
    if props.contains_key("order")? {
        let area: i64 = props.get("order")?;
    }
    if props.contains_key("grow")? {
        let area: f32 = props.get("grow")?;
    }
    if props.contains_key("shrink")? {
        let area: f32 = props.get("shrink")?;
    }
    if props.contains_key("basis")? {
        let area: DValue = props.get("area")?;
    }
    if props.contains_key("padding")? {
        let area: DAbsRect = props.get("padding")?;
    }
    if props.contains_key("margin")? {
        let area: DRect = props.get("margin")?;
    }
    if props.contains_key("maxsize")? {
        let area: DPoint = props.get("maxsize")?;
    }
    if props.contains_key("minsize")? {
        let area: DPoint = props.get("minsize")?;
    }
    if props.contains_key("anchor")? {
        let area: DPoint = props.get("anchor")?;
    }
    if props.contains_key("dim")? {
        let area: AbsPoint = props.get("dim")?;
    }

    Ok(bag)
}

fn region(_: &Lua, args: (LuaSourceID, LuaTable)) -> mlua::Result<ComponentBag> {
    let mut children: im::Vector<Option<Box<ChildOf<dyn fixed::Prop>>>> = im::Vector::new();

    for i in 0..args.1.len()? {
        let component: ComponentBag = args.1.get(i)?;
        children.push_back(Some(Box::new(component)));
    }

    let bag = if args.1.contains_key("props")? {
        gather_props(&args.1.get("props")?)?
    } else {
        PropBag::new()
    };

    Ok(Box::new(Region::<PropBag>::new(
        args.0.into(),
        bag.into(),
        children,
    )))
}

pub fn init_dsl(
    lua: &Lua,
    tab: &mut LuaTable,
    handlers: Vec<(String, Function)>,
) -> mlua::Result<()> {
    let handler_table = lua.create_table()?;

    for (name, f) in handlers {
        handler_table.set(name, f)?;
    }

    tab.set("handlers", handler_table)?;
    tab.set("region", lua.create_function(region)?)?;
}
