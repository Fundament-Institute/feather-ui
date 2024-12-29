use crate::layout::basic;
use crate::layout::basic::Basic;
use crate::layout::root;
use crate::layout::root::Root;
use crate::layout::simple;
use crate::layout::Desc;
use crate::outline::button::Button;
use crate::outline::region::Region;
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

pub type AppState<'lua> = LuaValue<'lua>;
type LuaSourceID = SourceID;

struct BoxedOutline<T: Desc>(Box<OutlineFrom<T>>);

impl<T: Desc> Clone for BoxedOutline<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

macro_rules! gen_from_lua {
    ($type_name:ident) => {
        impl mlua::FromLua<'_> for $type_name {
            #[inline]
            fn from_lua(value: ::mlua::Value, _: &::mlua::Lua) -> ::mlua::Result<Self> {
                match value {
                    ::mlua::Value::UserData(ud) => Ok(ud.borrow::<Self>()?.clone()),
                    _ => Err(::mlua::Error::FromLuaConversionError {
                        from: value.type_name(),
                        to: stringify!($type_name),
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
impl<T: Desc + 'static> mlua::FromLua<'_> for BoxedOutline<T> {
    #[inline]
    fn from_lua(value: ::mlua::Value, _: &::mlua::Lua) -> ::mlua::Result<Self> {
        match value {
            ::mlua::Value::UserData(ud) => {
                let test = std::cell::Ref::<'_, Self>::clone(&ud.borrow()?).clone();
                Ok(test)
            }
            _ => Err(::mlua::Error::FromLuaConversionError {
                from: value.type_name(),
                to: stringify!($type_name),
                message: None,
            }),
        }
    }
}

fn create_id<'lua>(
    _: &'lua Lua,
    (id, parent): (LuaValue<'lua>, Option<LuaSourceID>),
) -> mlua::Result<LuaSourceID> {
    Ok(crate::SourceID {
        // parent: parent.map(|x| Rc::downgrade(&x)).unwrap_or_default(),
        parent: std::rc::Weak::new(),
        id: if let Some(i) = id.as_integer() {
            DataID::Int(i)
        } else if let Some(s) = id.as_string_lossy() {
            DataID::Owned(s.into_owned())
        } else {
            panic!("Invalid ID")
        },
    }
    .into())
}

fn get_appdata_id(_: &Lua, (): ()) -> mlua::Result<LuaSourceID> {
    Ok(crate::APP_SOURCE_ID.into())
}

fn create_slot<'lua>(_: &'lua Lua, args: (LuaSourceID, u64)) -> mlua::Result<Slot> {
    Ok(Slot(args.0.into(), args.1))
}

fn create_urect<'lua>(
    _: &'lua Lua,
    args: (f32, f32, f32, f32, f32, f32, f32, f32),
) -> mlua::Result<URect> {
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

fn create_window<'lua>(
    _: &'lua Lua,
    args: (LuaSourceID, String, BoxedOutline<Root>),
) -> mlua::Result<Window> {
    Ok(Window::new(
        args.0.into(),
        winit::window::Window::default_attributes()
            .with_title(args.1)
            .with_resizable(true)
            .with_inner_size(winit::dpi::PhysicalSize::new(600, 400)),
        args.2 .0,
    ))
}

fn create_region<'lua>(
    _: &'lua Lua,
    args: (LuaSourceID, URect, Vec<BoxedOutline<Basic>>),
) -> mlua::Result<BoxedOutline<Root>> {
    let mut children = im::Vector::new();
    children.extend(args.2.into_iter().map(|x| Some(x.0)));
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

fn create_button<'lua>(
    _: &'lua Lua,
    args: (
        LuaSourceID,
        URect,
        String,
        Slot,
        BoxedOutline<simple::Simple>,
    ),
) -> mlua::Result<BoxedOutline<simple::Simple>> {
    let id = Rc::new(args.0);
    let text = Text::<()> {
        id: SourceID {
            parent: Rc::downgrade(&id),
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
    children.push_back(Some(args.4 .0));

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

fn create_label<'lua>(
    _: &'lua Lua,
    args: (LuaSourceID, URect, String),
) -> mlua::Result<BoxedOutline<simple::Simple>> {
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

fn create_round_rect<'lua>(
    _: &'lua Lua,
    args: (LuaSourceID, URect, u32, f32, f32, u32),
) -> mlua::Result<BoxedOutline<simple::Simple>> {
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

pub struct LuaApp<'lua> {
    pub window: LuaFunction<'lua>, // takes a Store and an appstate and returns a Window
    pub init: LuaFunction<'lua>,
}

impl<'lua> FnPersist<AppState<'lua>, im::HashMap<Rc<SourceID>, Option<Window>>> for LuaApp<'lua> {
    type Store = LuaValue<'lua>;

    fn init(&self) -> Self::Store {
        self.init.call::<_, LuaValue<'lua>>(()).unwrap()
    }
    fn call(
        &self,
        store: Self::Store,
        args: &AppState<'lua>,
    ) -> (Self::Store, im::HashMap<Rc<SourceID>, Option<Window>>) {
        let mut h = im::HashMap::new();
        let (store, w) = self
            .window
            .call::<(mlua::Value<'lua>, AppState<'lua>), (LuaValue, crate::outline::window::Window)>((
                store, args.clone(),
            ))
            .unwrap();
        h.insert(w.id().clone(), Some(w));
        (store, h)
    }
}

pub fn init_environment(lua: &Lua) -> mlua::Result<()> {
    lua.globals()
        .set("create_id", lua.create_function(create_id)?)?;
    lua.globals()
        .set("get_appdata_id", lua.create_function(get_appdata_id)?)?;
    lua.globals()
        .set("create_slot", lua.create_function(create_slot)?)?;
    lua.globals()
        .set("create_urect", lua.create_function(create_urect)?)?;
    lua.globals()
        .set("create_window", lua.create_function(create_window)?)?;
    lua.globals()
        .set("create_region", lua.create_function(create_region)?)?;
    lua.globals()
        .set("create_button", lua.create_function(create_button)?)?;
    lua.globals()
        .set("create_label", lua.create_function(create_label)?)?;
    lua.globals()
        .set("create_round_rect", lua.create_function(create_round_rect)?)?;

    Ok(())
}
