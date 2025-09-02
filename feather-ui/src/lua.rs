// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::{sRGB, sRGB32};
use crate::component::button::Button;
use crate::component::domain_line::DomainLine;
use crate::component::domain_point::DomainPoint;
use crate::component::flexbox::FlexBox;
use crate::component::gridbox::GridBox;
use crate::component::image::Image;
use crate::component::line::Line;
use crate::component::listbox::ListBox;
use crate::component::mouse_area::MouseArea;
use crate::component::region::Region;
use crate::component::scroll_area::ScrollArea;
use crate::component::text::Text;
use crate::component::textbox::TextBox;
use crate::component::window::Window;
use crate::component::{ChildOf, shape};
use crate::layout::{fixed, flex, grid, list};
use crate::persist::FnPersistStore;
use crate::propbag::PropBag;
use crate::{
    APP_SOURCE_ID, AccessCell, DAbsPoint, DAbsRect, DPoint, DRect, DValue, DataID, FnPersist2,
    InputResult, Logical, Pixel, Rect, Relative, ScopeID, Slot, SourceID, StateMachineChild,
    UNSIZED_AXIS,
};
use guillotiere::euclid::Point2D;
use mlua::UserData;
use mlua::prelude::*;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use wide::f32x4;
use winit::application::ApplicationHandler;

use winit::dpi;

const SANDBOX: &[u8] = include_bytes!("./sandbox.lua");
const FEATHER: &[u8] = include_bytes!("./feather.lua");

struct NamedChunk<'a>(&'a [u8], &'a str);

impl mlua::AsChunk for NamedChunk<'static> {
    fn name(&self) -> Option<String> {
        Some(self.1.into())
    }

    fn source<'a>(&self) -> std::io::Result<std::borrow::Cow<'a, [u8]>> {
        Ok(std::borrow::Cow::Borrowed(self.0))
    }
}

fn point_to_rect<T>(pt: Point2D<f32, T>) -> Rect<T> {
    Rect::new(pt.x, pt.y, pt.x, pt.y)
}

#[derive(Clone, Debug)]
struct LuaSourceID(Arc<SourceID>);

impl std::fmt::Display for LuaSourceID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

struct LuaEnum<T>(T);

impl<T: TryFrom<u8>> FromLua for LuaEnum<T>
where
    <T as TryFrom<u8>>::Error: std::error::Error + 'static + Sync + Send,
{
    fn from_lua(value: LuaValue, _: &mlua::Lua) -> LuaResult<Self> {
        T::try_from(value.as_i32().ok_or(LuaError::RuntimeError(format!(
            "Can't convert {} to enum",
            value.type_name()
        )))? as u8)
        .map_err(|e| LuaError::ExternalError(std::sync::Arc::new(e)))
        .map(|x| LuaEnum(x))
    }
}

impl<T: Into<u8>> IntoLua for LuaEnum<T> {
    fn into_lua(self, _: &Lua) -> LuaResult<LuaValue> {
        Ok(LuaValue::Integer(self.0.into().into()))
    }
}

fn get_key<V: FromLua>(t: &LuaTable, key: &str) -> LuaResult<Option<V>> {
    if t.contains_key(key)? {
        Ok(Some(t.get(key)?))
    } else {
        Ok(None)
    }
}

fn get_or_default<V: FromLua + Default>(t: &LuaTable, key: &str) -> LuaResult<V> {
    Ok(get_key(t, key)?.unwrap_or_default())
}

fn get_or<V: FromLua>(t: &LuaTable, key: &str, v: V) -> LuaResult<V> {
    Ok(get_key(t, key)?.unwrap_or(v))
}

fn get_required<V: FromLua>(t: &LuaTable, key: &str) -> LuaResult<V> {
    if !t.contains_key(key)? {
        Err(LuaError::RuntimeError(format!(
            "Missing required property: {key}"
        )))
    } else {
        t.get(key)
    }
}

fn get_array_or<T: num_traits::FromPrimitive + FromLua + Clone + Copy, const N: usize>(
    lua: &Lua,
    t: &LuaTable,
    key: &str,
    d: [T; N],
) -> LuaResult<[T; N]> {
    Ok(if t.contains_key(key)? {
        let v = t.get::<LuaValue>(key)?;
        if let Some(n) = v.as_number() {
            let num = T::from_f64(n).ok_or(LuaError::RuntimeError(format!(
                "Failed to convert {n}, is it out of range?"
            )))?;
            let test: [T; N] = [num; N];
            test
        } else if let Some(n) = v.as_integer() {
            let num = T::from_i64(n).ok_or(LuaError::RuntimeError(format!(
                "Failed to convert {n}, is it out of range?"
            )))?;
            let test: [T; N] = [num; N];
            test
        } else {
            <[T; N] as FromLua>::from_lua(v, lua)?
        }
    } else {
        d
    })
}

fn get_arg_default<V: FromLua + Default>(
    args: &mut HashSet<&'static str>,
    t: &LuaTable,
    key: &'static str,
) -> LuaResult<V> {
    args.insert(key);
    get_or_default(t, key)
}

fn get_arg_or<V: FromLua>(
    args: &mut HashSet<&'static str>,
    t: &LuaTable,
    key: &'static str,
    v: V,
) -> LuaResult<V> {
    args.insert(key);
    get_or(t, key, v)
}

fn get_arg_required<V: FromLua>(
    args: &mut HashSet<&'static str>,
    t: &LuaTable,
    key: &'static str,
) -> LuaResult<V> {
    args.insert(key);
    get_required(t, key)
}

fn get_array_arg<T: num_traits::FromPrimitive + FromLua + Clone + Copy, const N: usize>(
    args: &mut HashSet<&'static str>,
    lua: &Lua,
    t: &LuaTable,
    key: &'static str,
    d: [T; N],
) -> LuaResult<[T; N]> {
    args.insert(key);
    get_array_or(lua, t, key, d)
}

fn is_dvalue(t: &LuaTable) -> LuaResult<bool> {
    if let Some(mt) = t.metatable() {
        Ok(mt.get::<String>("name")? == "value_mt")
    } else {
        Ok(false)
    }
}

#[allow(dead_code)]
fn dump_value(k: LuaValue, v: LuaValue, i: usize) -> LuaResult<()> {
    if let Some(t) = v.as_table() {
        println!("{:i$}{k:?} -", "");
        if let Some(mt) = t.metatable() {
            println!("{:i$}  mt - {}", "", mt.get::<String>("name")?);
            mt.for_each(|k, v| dump_value(k, v, i + 4))?;
        }

        t.for_each(|k, v| dump_value(k, v, i + 2))?;
    } else {
        println!("{:i$}{k:?} : {v:?}", "");
    }
    Ok(())
}

fn get_kind(t: &LuaTable) -> LuaResult<String> {
    if let Some(mt) = t.metatable() {
        if !mt.contains_key("kind")? {
            Err(LuaError::RuntimeError("Unknown metatable".into()))
        } else {
            mt.get("kind")
        }
    } else {
        Err(LuaError::RuntimeError(
            "Expected type to have metatable, but no metatable was found.".into(),
        ))
    }
}

fn get_name(t: &LuaTable) -> LuaResult<String> {
    if let Some(mt) = t.metatable() {
        if !mt.contains_key("name")? {
            Err(LuaError::RuntimeError("nameless metatable".into()))
        } else {
            mt.get("name")
        }
    } else {
        Err(LuaError::RuntimeError(
            "Expected type to have metatable, but no metatable was found.".into(),
        ))
    }
}

fn expect_name(t: &LuaTable, expect: &str) -> LuaResult<()> {
    let name = match get_name(t) {
        Ok(s) => s,
        Err(_) => "[unknown table type]".into(),
    };

    if name != expect {
        Err(LuaError::RuntimeError(format!(
            "Expected {expect} type, but found {name}",
        )))
    } else {
        Ok(())
    }
}

impl FromLua for DValue {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let t = value.as_table().ok_or(LuaError::RuntimeError(format!(
            "Expected DValue, found {}",
            value.type_name(),
        )))?;
        if !is_dvalue(t)? {
            return Err(LuaError::RuntimeError(format!(
                "Expected DValue, found {}",
                get_name(t)?,
            )))?;
        }

        Ok(DValue {
            dp: get_or_default(t, "dp")?,
            px: get_or_default(t, "px")?,
            rel: get_or_default(t, "rel")?,
        })
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct LuaPoint<U>(Point2D<f32, U>);

impl<U> LuaPoint<U> {
    pub const fn nan() -> Self {
        Self(Point2D::new(f32::NAN, f32::NAN))
    }
}

impl<U> From<LuaPoint<U>> for Point2D<f32, U> {
    fn from(val: LuaPoint<U>) -> Self {
        val.0
    }
}

trait LuaKind {
    const KIND: &str;
}
impl LuaKind for crate::Pixel {
    const KIND: &str = "px";
}
impl LuaKind for crate::Logical {
    const KIND: &str = "dp";
}
impl LuaKind for crate::Relative {
    const KIND: &str = "rel";
}

impl<U: LuaKind> FromLua for LuaPoint<U> {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        if let Some(v) = value.as_number() {
            return Ok(LuaPoint(Point2D::<f32, U>::splat(v as f32)));
        }
        if let Some(v) = value.as_integer() {
            return Ok(LuaPoint(Point2D::<f32, U>::splat(v as f32)));
        }

        let t = value.as_table().ok_or(LuaError::RuntimeError(format!(
            "Expected a number or an object, but found {}",
            value.type_name()
        )))?;
        if is_dvalue(t)? {
            const TYPES: [&str; 3] = ["dp", "px", "rel"];

            for ty in TYPES {
                if t.contains_key(ty)? != (U::KIND == ty) {
                    return Err(LuaError::RuntimeError(format!(
                        "Tried to build a {}point but found a value of kind {ty}",
                        U::KIND
                    )));
                }
            }
            Ok(LuaPoint(Point2D::<f32, U>::splat(t.get(U::KIND)?)))
        } else if let Some(mt) = t.metatable() {
            expect_name(&mt, "point_mt")?;

            if get_kind(t)? == U::KIND {
                Ok(LuaPoint(Point2D::<f32, U>::new(t.get("x")?, t.get("y")?)))
            } else {
                Err(LuaError::RuntimeError(format!(
                    "Tried to build a {}point but found kind {}",
                    U::KIND,
                    get_kind(t)?
                )))
            }
        } else {
            Err(LuaError::RuntimeError(format!(
                "Found unknown table kind {} while looking for a {}point",
                get_kind(t)?,
                U::KIND,
            )))
        }
    }
}

impl<U: LuaKind> FromLua for Rect<U> {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::RuntimeError(format!(
            "Expected a rect object, but found {}",
            value.type_name()
        )))?;

        if get_kind(v)? != U::KIND {
            return Err(LuaError::RuntimeError(format!(
                "Trying to build a {}rect but found kind {}",
                U::KIND,
                get_kind(v)?
            )));
        }

        Ok(Rect::<U> {
            v: f32x4::new(if v.contains_key("x")? || v.contains_key("y")? {
                let x = get_or_default(v, "x")?;
                let y = get_or_default(v, "y")?;
                [x, y, x, y]
            } else {
                [
                    get_or_default(v, "left")?,
                    get_or_default(v, "top")?,
                    get_or_default(v, "right")?,
                    get_or_default(v, "bottom")?,
                ]
            }),
            _unit: PhantomData,
        })
    }
}

impl FromLua for DAbsRect {
    fn from_lua(value: LuaValue, lua: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::RuntimeError(format!(
            "Expected a rect, but found {}",
            value.type_name()
        )))?;
        let name = get_name(v)?;
        if name == "value_mt" {
            if v.contains_key("rel")? {
                return Err(LuaError::RuntimeError(
                    "DAbsRect cannot have a relative component!".to_string(),
                ));
            }
            let v = DValue::from_lua(value, lua)?;
            let px = Rect::splat(v.px);
            let dp = Rect::splat(v.dp);
            Ok(DAbsRect { dp, px })
        } else if name == "pxrect_mt" {
            Ok(Rect::<Pixel>::from_lua(value, lua)?.into())
        } else if name == "dprect_mt" {
            Ok(Rect::<Logical>::from_lua(value, lua)?.into())
        } else if name == "pxpoint_mt" {
            Ok(point_to_rect(LuaPoint::<Pixel>::from_lua(value, lua)?.0).into())
        } else if name == "abspoint_mt" {
            Ok(point_to_rect(LuaPoint::<Logical>::from_lua(value, lua)?.0).into())
        } else if name == "area_mt" {
            let px = get_or_default(v, "px")?;
            let dp = get_or_default(v, "dp")?;
            if v.contains_key("rel")? {
                return Err(LuaError::RuntimeError(
                    "DAbsRect cannot have a relative component!".to_string(),
                ));
            }
            Ok(DAbsRect { dp, px })
        } else {
            Err(LuaError::RuntimeError(format!(
                "Expected an AbsRect or PxRect, but found {name}",
            )))
        }
    }
}

impl FromLua for DRect {
    fn from_lua(value: LuaValue, lua: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::RuntimeError(format!(
            "Expected a rect, but found {}",
            value.type_name()
        )))?;
        let name = get_name(v)?;
        if name == "value_mt" {
            let v = DValue::from_lua(value, lua)?;
            let px = Rect::splat(v.px);
            let dp = Rect::splat(v.dp);
            let rel = Rect::splat(v.rel);
            Ok(DRect { dp, px, rel })
        } else if name == "pxrect_mt" {
            Ok(Rect::<Pixel>::from_lua(value, lua)?.into())
        } else if name == "dprect_mt" {
            Ok(Rect::<Logical>::from_lua(value, lua)?.into())
        } else if name == "relrect_mt" {
            Ok(Rect::<Relative>::from_lua(value, lua)?.into())
        } else if name == "pxpoint_mt" {
            Ok(point_to_rect(LuaPoint::<Pixel>::from_lua(value, lua)?.0).into())
        } else if name == "abspoint_mt" {
            Ok(point_to_rect(LuaPoint::<Logical>::from_lua(value, lua)?.0).into())
        } else if name == "relpoint_mt" {
            Ok(point_to_rect(LuaPoint::<Relative>::from_lua(value, lua)?.0).into())
        } else if name == "area_mt" {
            let px = get_or_default(v, "px")?;
            let dp = get_or_default(v, "dp")?;
            let rel = get_or_default(v, "rel")?;
            Ok(DRect { dp, px, rel })
        } else {
            Err(LuaError::RuntimeError(format!(
                "Expected a rect, but found {name}",
            )))
        }
    }
}

impl FromLua for DAbsPoint {
    fn from_lua(value: LuaValue, lua: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::RuntimeError(format!(
            "Expected a point, but found {}",
            value.type_name()
        )))?;
        let name = get_name(v)?;
        if name == "value_mt" {
            if v.contains_key("rel")? {
                return Err(LuaError::RuntimeError(
                    "DAbsPoint cannot have a relative component!".to_string(),
                ));
            }
            let v = DValue::from_lua(value, lua)?;
            let px = Point2D::splat(v.px);
            let dp = Point2D::splat(v.dp);
            Ok(DAbsPoint { dp, px })
        } else if name == "pxpoint_mt" {
            Ok(LuaPoint::<Pixel>::from_lua(value, lua)?.0.into())
        } else if name == "abspoint_mt" {
            Ok(LuaPoint::<Logical>::from_lua(value, lua)?.0.into())
        } else if name == "coord_mt" {
            let px = get_or_default::<LuaPoint<Pixel>>(v, "px")?.0;
            let dp = get_or_default::<LuaPoint<Logical>>(v, "dp")?.0;
            if v.contains_key("rel")? {
                return Err(LuaError::RuntimeError(
                    "DAbsPoint cannot have a relative component!".to_string(),
                ));
            }
            Ok(DAbsPoint { dp, px })
        } else {
            Err(LuaError::RuntimeError(format!(
                "Expected either an abs or a px point, but found {name}",
            )))
        }
    }
}

impl FromLua for DPoint {
    fn from_lua(value: LuaValue, lua: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::RuntimeError(format!(
            "Expected a point, but found {}",
            value.type_name()
        )))?;
        let name = get_name(v)?;
        if name == "value_mt" {
            let v = DValue::from_lua(value, lua)?;
            let px = Point2D::splat(v.px);
            let dp = Point2D::splat(v.dp);
            let rel = Point2D::splat(v.rel);
            Ok(DPoint { dp, px, rel })
        } else if name == "pxpoint_mt" {
            Ok(LuaPoint::<Pixel>::from_lua(value, lua)?.0.into())
        } else if name == "abspoint_mt" {
            Ok(LuaPoint::<Logical>::from_lua(value, lua)?.0.into())
        } else if name == "relpoint_mt" {
            Ok(LuaPoint::<Relative>::from_lua(value, lua)?.0.into())
        } else if name == "coord_mt" {
            let px = get_or_default::<LuaPoint<Pixel>>(v, "px")?.0;
            let dp = get_or_default::<LuaPoint<Logical>>(v, "dp")?.0;
            let rel = get_or_default::<LuaPoint<Relative>>(v, "rel")?.0;
            Ok(DPoint { dp, px, rel })
        } else {
            Err(LuaError::RuntimeError(format!(
                "Expected a point, but found {name}",
            )))
        }
    }
}

struct LimitPoint(DPoint);

impl FromLua for LimitPoint {
    fn from_lua(value: LuaValue, lua: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::RuntimeError(format!(
            "Expected a point, but found {}",
            value.type_name()
        )))?;

        // We can't use the default .into() method here because all unassigned values must be NaN
        let name = get_name(v)?;
        if name == "pxpoint_mt" {
            Ok(LimitPoint(DPoint {
                rel: LuaPoint::nan().0,
                dp: LuaPoint::nan().0,
                px: LuaPoint::<Pixel>::from_lua(value, lua)?.0,
            }))
        } else if name == "abspoint_mt" {
            Ok(LimitPoint(DPoint {
                rel: LuaPoint::nan().0,
                dp: LuaPoint::<Logical>::from_lua(value, lua)?.0,
                px: LuaPoint::nan().0,
            }))
        } else if name == "relpoint_mt" {
            Ok(LimitPoint(DPoint {
                rel: LuaPoint::<Relative>::from_lua(value, lua)?.0,
                dp: LuaPoint::nan().0,
                px: LuaPoint::nan().0,
            }))
        } else if name == "coord_mt" {
            let px = get_or::<LuaPoint<Pixel>>(v, "px", LuaPoint::nan())?.0;
            let dp = get_or::<LuaPoint<Logical>>(v, "dp", LuaPoint::nan())?.0;
            let rel = get_or::<LuaPoint<Relative>>(v, "rel", LuaPoint::nan())?.0;
            Ok(LimitPoint(DPoint { dp, px, rel }))
        } else {
            Err(LuaError::RuntimeError(format!(
                "Expected a point, but found {name}",
            )))
        }
    }
}

impl FromLua for sRGB {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        if let Some(i) = value.as_integer() {
            Ok(sRGB32 { rgba: i as u32 }.as_f32())
        } else if let Some(v) = value.as_table() {
            if v.len()? == 4 {
                Ok(sRGB::new(v.get(1)?, v.get(2)?, v.get(3)?, v.get(4)?))
            } else {
                Err(LuaError::RuntimeError(format!(
                    "A color must be an array of exactly 4 numbers, declared like {{ R, G, B, A }}, but only found {}",
                    v.len()?
                )))
            }
        } else {
            Err(LuaError::RuntimeError(format!(
                "Expected a color but found {}",
                value.type_name()
            )))
        }
    }
}

struct LuaFontFamily(cosmic_text::FamilyOwned);

impl Default for LuaFontFamily {
    fn default() -> Self {
        Self(cosmic_text::FamilyOwned::SansSerif)
    }
}

impl FromLua for LuaFontFamily {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let name =
            value
                .as_string()
                .and_then(|s| s.to_str().ok())
                .ok_or(LuaError::RuntimeError(format!(
                    "Expected a string, but found {}",
                    value.type_name()
                )))?;

        Ok(LuaFontFamily(if name.eq_ignore_ascii_case("serif") {
            cosmic_text::FamilyOwned::Serif
        } else if name.eq_ignore_ascii_case("cursive") {
            cosmic_text::FamilyOwned::Cursive
        } else if name.eq_ignore_ascii_case("fantasy") {
            cosmic_text::FamilyOwned::Fantasy
        } else if name.eq_ignore_ascii_case("monospace") {
            cosmic_text::FamilyOwned::Monospace
        } else if name.eq_ignore_ascii_case("sansserif") || name.eq_ignore_ascii_case("sans-serif")
        {
            cosmic_text::FamilyOwned::SansSerif
        } else {
            cosmic_text::FamilyOwned::Name((*name).into())
        }))
    }
}

pub type ComponentBag = Box<dyn crate::component::Component<Props = PropBag>>;

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

#[derive(Clone)]
struct LuaDomain(std::sync::Arc<crate::CrossReferenceDomain>);

impl UserData for LuaDomain {}
gen_from_lua!(LuaDomain);

impl UserData for Window {}
gen_from_lua!(Window);

impl UserData for LuaSourceID {}
gen_from_lua!(LuaSourceID);

impl UserData for Slot {
    fn add_methods<M: mlua::UserDataMethods<Self>>(methods: &mut M) {
        methods.add_meta_method(mlua::MetaMethod::ToString, |_, this, ()| {
            Ok(format!("Slot #{}: {}", this.1, this.0))
        });
    }
}
gen_from_lua!(Slot);

impl UserData for ComponentBag {}
gen_from_lua!(ComponentBag);

/// This defines the "lua" app that knows how to handle a lua value that contains the
/// expected rust objects, and hand them off for processing. This is analogous to the
/// pure-rust [App] struct defined in lib.rs
pub struct LuaPersist<AppData> {
    pub window: LuaFunction, // takes a Store and an appstate and returns a Window
    pub id_enter: LuaFunction,
    pub init: Result<LuaFunction, AppData>,
    phantom: PhantomData<AppData>,
}

impl<AppData: Clone> FnPersistStore for LuaPersist<AppData> {
    type Store = AppData;
}

impl<AppData: Clone + FromLua + IntoLua>
    FnPersist2<AppData, ScopeID<'static>, im::HashMap<Arc<SourceID>, Option<Window>>>
    for LuaPersist<AppData>
{
    fn init(&self) -> Self::Store {
        let r = match &self.init {
            Ok(init) => init.call::<AppData>(()),
            Err(data) => Ok(data.clone()),
        };

        match r {
            Err(LuaError::RuntimeError(s)) => panic!("{}", s),
            Err(e) => panic!("{e:?}"),
            Ok(v) => v,
        }
    }
    fn call(
        &mut self,
        _: Self::Store,
        appdata: AppData,
        mut id: ScopeID<'static>,
    ) -> (Self::Store, im::HashMap<Arc<SourceID>, Option<Window>>) {
        let mut h = im::HashMap::new();

        let (store, w) = self
            .id_enter
            .call::<(AppData, crate::component::window::Window)>((
                LuaSourceID(id.id().clone()),
                self.window.clone(),
                appdata.clone(),
            ))
            .unwrap();
        h.insert(w.id().clone(), Some(w));
        (store, h)
    }
}

enum LuaDualPoint {
    Px(Point2D<f32, Pixel>),
    Dp(Point2D<f32, Logical>),
}

impl FromLua for LuaDualPoint {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let t = value.as_table().ok_or(LuaError::RuntimeError(format!(
            "Expected a 2D point, but found {}",
            value.type_name()
        )))?;

        if is_dvalue(t)? {
            if t.contains_key("dp")? && !t.contains_key("px")? {
                Ok(LuaDualPoint::Dp(Point2D::<f32, Logical>::splat(
                    t.get("dp")?,
                )))
            } else if t.contains_key("px")? && !t.contains_key("dp")? {
                Ok(LuaDualPoint::Px(Point2D::<f32, Pixel>::splat(t.get("px")?)))
            } else {
                Err(LuaError::RuntimeError("This property only accepts a point that is either abs or px, not relative or a combination.".to_string()))
            }
        } else if t.contains_key("dp")? && !t.contains_key("px")? {
            Ok(LuaDualPoint::Dp(t.get::<LuaPoint<Logical>>("dp")?.0))
        } else if t.contains_key("px")? && !t.contains_key("dp")? {
            Ok(LuaDualPoint::Px(t.get::<LuaPoint<Pixel>>("px")?.0))
        } else if get_kind(t)? == "px" {
            Ok(LuaDualPoint::Px(Point2D::<f32, Pixel>::new(
                t.get("x")?,
                t.get("y")?,
            )))
        } else if get_kind(t)? == "dp" {
            Ok(LuaDualPoint::Dp(Point2D::<f32, Logical>::new(
                t.get("x")?,
                t.get("y")?,
            )))
        } else {
            Err(LuaError::RuntimeError("This property only accepts a point that is either abs or px, not relative or a combination.".to_string()))
        }
    }
}

fn load_prop<T: mlua::FromLua>(
    f: fn(&mut PropBag, T) -> Option<T>,
    bag: &mut PropBag,
    props: &LuaTable,
    name: &str,
) -> LuaResult<()> {
    if props.contains_key(name)? {
        f(bag, props.get(name)?);
    }
    Ok(())
}

#[inline]
fn replace_unsized<U>(rect: &mut Rect<U>) {
    rect.v = f32x4::new(
        rect.v
            .to_array()
            .map(|x| if x.is_nan() { UNSIZED_AXIS } else { x }),
    );
}

#[inline]
fn replace_unsized_drect(mut rect: DRect) -> DRect {
    replace_unsized(&mut rect.dp);
    replace_unsized(&mut rect.px);
    replace_unsized(&mut rect.rel);
    rect
}

#[inline]
fn replace_limit<U>(p: &mut Point2D<f32, U>, bound: f32) {
    if p.x.is_nan() {
        p.x = bound
    }
    if p.y.is_nan() {
        p.y = bound
    }
}

#[inline]
fn replace_limit_dpoint(mut p: DPoint, bound: f32) -> DPoint {
    replace_limit(&mut p.dp, bound);
    replace_limit(&mut p.px, bound);
    replace_limit(&mut p.rel, bound);
    p
}

impl FromLua for PropBag {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let props = value.as_table().ok_or(LuaError::RuntimeError(format!(
            "Expected PropBag, got {}",
            value.type_name(),
        )))?;
        let mut bag = PropBag::new();

        load_prop(PropBag::set_wrap, &mut bag, props, "wrap")?;
        load_prop(PropBag::set_zindex, &mut bag, props, "zindex")?;
        load_prop(PropBag::set_order, &mut bag, props, "order")?;
        load_prop(PropBag::set_grow, &mut bag, props, "grow")?;
        load_prop(PropBag::set_shrink, &mut bag, props, "shrink")?;
        load_prop(PropBag::set_basis, &mut bag, props, "basis")?;
        load_prop(PropBag::set_padding, &mut bag, props, "padding")?;
        load_prop(PropBag::set_margin, &mut bag, props, "margin")?;
        load_prop(PropBag::set_anchor, &mut bag, props, "anchor")?;

        if props.contains_key("area")? {
            bag.set_area(replace_unsized_drect(props.get::<DRect>("area")?));
        }

        let mut limits: crate::DLimits = Default::default();
        let mut rlimits: crate::Limits<Relative> = Default::default();

        if props.contains_key("minsize")? {
            let p = replace_limit_dpoint(props.get::<LimitPoint>("minsize")?.0, f32::NEG_INFINITY);
            limits.dp.set_min(p.dp.to_vector().to_size());
            limits.px.set_min(p.px.to_vector().to_size());
            rlimits.set_min(p.rel.to_vector().to_size());
        }

        if props.contains_key("maxsize")? {
            let p = replace_limit_dpoint(props.get::<LimitPoint>("maxsize")?.0, f32::INFINITY);
            limits.dp.set_max(p.dp.to_vector().to_size());
            limits.px.set_max(p.px.to_vector().to_size());
            rlimits.set_max(p.rel.to_vector().to_size());
        }

        bag.set_limits(limits);
        bag.set_rlimits(rlimits);

        if props.contains_key("direction")? {
            bag.set_direction(props.get::<LuaEnum<crate::RowDirection>>("direction")?.0);
        }

        if props.contains_key("justify")? {
            bag.set_justify(props.get::<LuaEnum<flex::FlexJustify>>("justify")?.0);
        }

        if props.contains_key("align")? {
            bag.set_align(props.get::<LuaEnum<flex::FlexJustify>>("align")?.0);
        }

        if props.contains_key("domain")? {
            bag.set_domain(props.get::<LuaDomain>("domain")?.0);
        }

        if props.contains_key("obstacles")? {
            bag.set_obstacles(props.get::<Vec<DAbsRect>>("obstacles")?.as_slice());
        }

        if props.contains_key("dim")? {
            bag.set_dim(props.get::<LuaPoint<Pixel>>("dim")?.0.to_vector().to_size());
        }

        if props.contains_key("rows")? {
            bag.set_rows(props.get::<Vec<DValue>>("rows")?.as_slice());
        }

        if props.contains_key("columns")? {
            bag.set_columns(props.get::<Vec<DValue>>("columns")?.as_slice());
        }

        if props.contains_key("spacing")? {
            bag.set_spacing(props.get::<DPoint>("spacing")?);
        }

        if props.contains_key("coord")? {
            let coords = props.get::<Vec<usize>>("coord")?;
            bag.set_coord((coords[0], coords[1]));
        }

        if props.contains_key("span")? {
            let coords = props.get::<Vec<usize>>("span")?;
            bag.set_span((coords[0], coords[1]));
        }

        Ok(bag)
    }
}

fn prop_no_children(t: &LuaTable) -> LuaResult<PropBag> {
    let count = t.len()?;
    if count > 0 {
        return Err(LuaError::RuntimeError(format!(
            "Found {count} child components for a component that doesn't accept children!"
        )));
    }

    let bag: PropBag = get_required(t, "props")?;
    Ok(bag)
}

fn push_child<D: crate::layout::Desc + ?Sized>(
    t: &LuaTable,
    i: i64,
    children: &mut im::Vector<Option<Box<ChildOf<D>>>>,
) -> LuaResult<()>
where
    std::boxed::Box<dyn crate::component::Component<Props = PropBag>>:
        crate::component::ComponentWrap<<D as crate::layout::Desc>::Child>,
{
    let v = t.get::<LuaValue>(i)?;
    if v.is_nil() {
        return Ok(());
    } else if let Some(inner) = v.as_table()
        && let Some(mt) = inner.metatable()
        && let Ok(s) = mt.get::<String>("kind")
        && s == "multicomponent_mt"
    {
        for j in 1..=inner.len()? {
            push_child::<D>(inner, j, children)?;
        }
    } else {
        let component: ComponentBag = t.get(i)?;
        children.push_back(Some(Box::new(component)));
    }

    Ok(())
}

#[allow(clippy::type_complexity)]
fn prop_children<D: crate::layout::Desc + ?Sized>(
    t: &LuaTable,
) -> LuaResult<(im::Vector<Option<Box<ChildOf<D>>>>, PropBag)>
where
    std::boxed::Box<dyn crate::component::Component<Props = PropBag>>:
        crate::component::ComponentWrap<<D as crate::layout::Desc>::Child>,
{
    let mut children: im::Vector<Option<Box<ChildOf<D>>>> = im::Vector::new();

    for i in 1..=t.len()? {
        push_child::<D>(t, i, &mut children)?;
    }

    let bag: PropBag = get_required(t, "props")?;
    Ok((children, bag))
}

fn check_args(name: &str, t: LuaTable, v: HashSet<&'static str>) -> LuaResult<()> {
    for pair in t.pairs() {
        let (k, _): (LuaValue, LuaValue) = pair?;
        if k.as_integer().is_none()
            && k.as_number().is_none()
            && let Some(s) = k.as_string()
        {
            let str = s.to_string_lossy();
            if !v.contains(str.as_str()) {
                return Err(LuaError::FromLuaConversionError {
                    from: "[attributes table]",
                    to: name.to_string(),
                    message: Some(format!("Unexpected key {str} in attributes table")),
                });
            }
        }
    }

    Ok(())
}

fn create_button(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let (children, bag) = prop_children::<dyn fixed::Prop>(&body)?;

    let mut args = HashSet::from_iter(["props"]);
    let onclick = get_arg_required(&mut args, &body, "onclick")?;
    check_args("button", body, args)?;

    let res: LuaResult<ComponentBag> = Ok(Box::new(Button::<PropBag>::new(
        id.0, bag, onclick, children,
    )));
    res
}

fn create_domain_line(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let bag = prop_no_children(&body)?;

    let mut args = HashSet::from_iter(["props"]);
    let domain: LuaDomain = get_arg_required(&mut args, &body, "domain")?;
    let start: LuaSourceID = get_arg_required(&mut args, &body, "start")?;
    let end: LuaSourceID = get_arg_required(&mut args, &body, "end")?;
    let fill = get_arg_default(&mut args, &body, "fill")?;
    check_args("domain-line", body, args)?;

    let res: LuaResult<ComponentBag> = Ok(Box::new(DomainLine::<PropBag> {
        id: id.0,
        domain: domain.0,
        start: start.0,
        end: end.0,
        props: bag.into(),
        fill,
    }));
    res
}

fn create_domain_point(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let bag = prop_no_children(&body)?;

    let args = HashSet::from_iter(["props"]);
    check_args("domain-point", body, args)?;

    let res: LuaResult<ComponentBag> = Ok(Box::new(DomainPoint::<PropBag>::new(id.0, bag)));
    res
}

fn create_flexbox(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let (children, bag) = prop_children::<dyn flex::Prop>(&body)?;
    let args = HashSet::from_iter(["props"]);
    check_args("flexbox", body, args)?;

    let res: LuaResult<ComponentBag> = Ok(Box::new(FlexBox::<PropBag>::new(id.0, bag, children)));
    res
}

fn create_gridbox(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let (children, bag) = prop_children::<dyn grid::Prop>(&body)?;
    let args = HashSet::from_iter(["props"]);
    check_args("gridbox", body, args)?;

    let res: LuaResult<ComponentBag> = Ok(Box::new(GridBox::<PropBag>::new(id.0, bag, children)));
    res
}

fn create_image(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let bag = prop_no_children(&body)?;

    let mut args = HashSet::from_iter(["props"]);
    let resource: String = get_arg_required(&mut args, &body, "resource")?;
    let size: DAbsPoint = get_arg_default(&mut args, &body, "size")?;
    let dynamic: bool = get_arg_default(&mut args, &body, "dynamic")?;
    check_args("image", body, args)?;

    let location = std::path::PathBuf::from(resource);

    let res: LuaResult<ComponentBag> = Ok(Box::new(Image::<PropBag>::new(
        id.0, bag, &location, size, dynamic,
    )));
    res
}

fn create_line(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let bag = prop_no_children(&body)?;

    let mut args = HashSet::from_iter(["props"]);
    let start: LuaPoint<Pixel> = get_arg_required(&mut args, &body, "start")?;
    let end: LuaPoint<Pixel> = get_arg_required(&mut args, &body, "end")?;
    let fill = get_arg_default(&mut args, &body, "fill")?;
    check_args("line", body, args)?;

    let res: LuaResult<ComponentBag> = Ok(Box::new(Line::<PropBag> {
        id: id.0,
        start: start.0,
        end: end.0,
        props: bag.into(),
        fill,
    }));
    res
}

fn create_listbox(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let (children, bag) = prop_children::<dyn list::Prop>(&body)?;
    let args = HashSet::from_iter(["props"]);
    check_args("listbox", body, args)?;

    let res: LuaResult<ComponentBag> = Ok(Box::new(ListBox::<PropBag>::new(id.0, bag, children)));
    res
}

fn create_mousearea(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let bag = prop_no_children(&body)?;

    let mut args = HashSet::from_iter(["props"]);
    let deadzone = get_arg_or(&mut args, &body, "deadzone", f32::INFINITY)?;
    let onclick = get_arg_default(&mut args, &body, "onclick")?;
    let ondblclick = get_arg_default(&mut args, &body, "ondblclick")?;
    let ondrag = get_arg_default(&mut args, &body, "ondrag")?;
    check_args("mousearea", body, args)?;

    let res: LuaResult<ComponentBag> = Ok(Box::new(MouseArea::<PropBag>::new(
        id.0,
        bag,
        Some(deadzone),
        [onclick, ondblclick, ondrag, None, None, None],
    )));
    res
}

fn create_region(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let (children, bag) = prop_children::<dyn fixed::Prop>(&body)?;
    let mut args = HashSet::from_iter(["props"]);
    let color: Option<sRGB> = get_arg_default(&mut args, &body, "color")?;
    let rotation: Option<f32> = get_arg_default(&mut args, &body, "rotation")?;
    check_args("region", body, args)?;

    Ok(Box::new(if color.is_some() || rotation.is_some() {
        Region::<PropBag>::new_layer(
            id.0,
            bag,
            color.unwrap_or(sRGB::white()).as_32bit(),
            rotation.unwrap_or_default(),
            children,
        )
    } else {
        Region::<PropBag>::new(id.0, bag, children)
    }))
}

fn create_scrollarea(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let (children, bag) = prop_children::<dyn fixed::Prop>(&body)?;
    let mut args = HashSet::from_iter(["props"]);
    let stepx = get_arg_default(&mut args, &body, "stepx")?;
    let stepy = get_arg_default(&mut args, &body, "stepy")?;
    let extension = get_arg_default(&mut args, &body, "extension")?;
    let onscroll = get_arg_default(&mut args, &body, "onscroll")?;
    check_args("scrollarea", body, args)?;

    let res: LuaResult<ComponentBag> = Ok(Box::new(ScrollArea::<PropBag>::new(
        id.0,
        bag,
        (stepx, stepy),
        extension,
        children,
        [onscroll],
    )));
    res
}

fn create_round_rect(lua: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let bag = prop_no_children(&body)?;

    let mut args: HashSet<&'static str> = HashSet::from_iter(["props"]);
    let border = get_arg_default(&mut args, &body, "border")?;
    let blur = get_arg_default(&mut args, &body, "blur")?;
    let fill = get_arg_default(&mut args, &body, "fill")?;
    let outline = get_arg_default(&mut args, &body, "outline")?;
    let corners = get_array_arg(&mut args, lua, &body, "corners", [0.0; 4])?;
    let size: DAbsPoint = get_arg_default(&mut args, &body, "size")?;
    check_args("round_rect", body, args)?;

    Ok(Box::new(shape::round_rect(
        id.0,
        bag,
        border,
        blur,
        wide::f32x4::new(corners),
        fill,
        outline,
        size,
    )))
}

fn create_arc(lua: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let bag = prop_no_children(&body)?;

    let mut args: HashSet<&'static str> = HashSet::from_iter(["props"]);
    let border = get_arg_default(&mut args, &body, "border")?;
    let blur = get_arg_default(&mut args, &body, "blur")?;
    let fill = get_arg_default(&mut args, &body, "fill")?;
    let outline = get_arg_default(&mut args, &body, "outline")?;
    let angles = get_array_arg(&mut args, lua, &body, "angles", [0.0; 2])?;
    let inner_radius = get_arg_default(&mut args, &body, "innerRadius")?;
    let size: DAbsPoint = get_arg_default(&mut args, &body, "size")?;
    check_args("arc", body, args)?;

    Ok(Box::new(shape::arcs(
        id.0,
        bag,
        border,
        blur,
        inner_radius,
        angles,
        fill,
        outline,
        size,
    )))
}

fn create_triangle(lua: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let bag = prop_no_children(&body)?;

    let mut args: HashSet<&'static str> = HashSet::from_iter(["props"]);
    let border = get_arg_default(&mut args, &body, "border")?;
    let blur = get_arg_default(&mut args, &body, "blur")?;
    let fill = get_arg_default(&mut args, &body, "fill")?;
    let outline = get_arg_default(&mut args, &body, "outline")?;
    let corners = get_array_arg(&mut args, lua, &body, "corners", [0.0; 3])?;
    let offset = get_arg_default(&mut args, &body, "offset")?;
    let size: DAbsPoint = get_arg_default(&mut args, &body, "size")?;
    check_args("triangle", body, args)?;

    Ok(Box::new(shape::triangle(
        id.0, bag, border, blur, corners, offset, fill, outline, size,
    )))
}

fn create_circle(lua: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let bag = prop_no_children(&body)?;

    let mut args: HashSet<&'static str> = HashSet::from_iter(["props"]);
    let border = get_arg_default(&mut args, &body, "border")?;
    let blur = get_arg_default(&mut args, &body, "blur")?;
    let fill = get_arg_default(&mut args, &body, "fill")?;
    let outline = get_arg_default(&mut args, &body, "outline")?;
    let radii = get_array_arg(&mut args, lua, &body, "radii", [0.0; 2])?;
    let size: DAbsPoint = get_arg_default(&mut args, &body, "size")?;
    check_args("circle", body, args)?;

    Ok(Box::new(shape::circle(
        id.0, bag, border, blur, radii, fill, outline, size,
    )))
}

// In CSS, 1.2 is usually used as the "reasonable" default line-height for a given font.
const DEFAULT_LINE_HEIGHT: f32 = 1.2;
// CSS defaults to 16 pixels, which is 12.8 points. We round up to 14 points as the next highest even number.
const DEFAULT_FONT_SIZE: f32 = 14.0;

fn to_style(style: u8) -> LuaResult<cosmic_text::Style> {
    match style {
        0 => Ok(cosmic_text::Style::Normal),
        1 => Ok(cosmic_text::Style::Italic),
        2 => Ok(cosmic_text::Style::Oblique),
        _ => Err(LuaError::RuntimeError(format!(
            "{style} is not a valid style enum value!"
        ))),
    }
}

fn to_wrap(wrap: u8) -> LuaResult<cosmic_text::Wrap> {
    match wrap {
        0 => Ok(cosmic_text::Wrap::None),
        1 => Ok(cosmic_text::Wrap::Glyph),
        2 => Ok(cosmic_text::Wrap::Word),
        3 => Ok(cosmic_text::Wrap::WordOrGlyph),
        _ => Err(LuaError::RuntimeError(format!(
            "{wrap} is not a valid wrap enum value!"
        ))),
    }
}

fn to_align(align: Option<u8>) -> LuaResult<Option<cosmic_text::Align>> {
    if let Some(a) = align {
        Ok(Some(match a {
            0 => cosmic_text::Align::Left,
            1 => cosmic_text::Align::Right,
            2 => cosmic_text::Align::Center,
            3 => cosmic_text::Align::Justified,
            4 => cosmic_text::Align::End,
            _ => {
                return Err(LuaError::RuntimeError(format!(
                    "{a} is not a valid align enum value!"
                )));
            }
        }))
    } else {
        Ok(None)
    }
}

fn create_text(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let bag = prop_no_children(&body)?;

    let mut args = HashSet::from_iter(["props"]);
    let text = get_arg_required(&mut args, &body, "text")?;
    let font_size = get_arg_or(&mut args, &body, "fontsize", DEFAULT_FONT_SIZE)?;
    let line_height = get_arg_or(
        &mut args,
        &body,
        "lineheight",
        font_size * DEFAULT_LINE_HEIGHT,
    )?;
    let font: LuaFontFamily = get_arg_default(&mut args, &body, "font")?;
    let color = get_arg_required(&mut args, &body, "color")?;
    let weight: u16 = get_arg_or(&mut args, &body, "weight", cosmic_text::Weight::NORMAL.0)?;
    let style: u8 = get_arg_default(&mut args, &body, "style")?;
    let wrap: u8 = get_arg_default(&mut args, &body, "wrap")?;
    let align: Option<u8> = get_arg_default(&mut args, &body, "align")?;
    check_args("text", body, args)?;

    let style = to_style(style)?;
    let wrap = to_wrap(wrap)?;
    let align = to_align(align)?;

    Ok(Box::new(Text::<PropBag>::new(
        id.0,
        bag,
        font_size,
        line_height,
        text,
        font.0,
        color,
        cosmic_text::Weight(weight),
        style,
        wrap,
        align,
    )))
}

fn create_textbox(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<ComponentBag> {
    let bag = prop_no_children(&body)?;

    let mut args = HashSet::from_iter(["props"]);
    let font_size = get_arg_or(&mut args, &body, "fontsize", DEFAULT_FONT_SIZE)?;
    let line_height = get_arg_or(
        &mut args,
        &body,
        "lineheight",
        font_size * DEFAULT_LINE_HEIGHT,
    )?;
    let font: LuaFontFamily = get_arg_default(&mut args, &body, "font")?;
    let color = get_arg_required(&mut args, &body, "color")?;
    let weight: u16 = get_arg_or(&mut args, &body, "weight", cosmic_text::Weight::NORMAL.0)?;
    let style: u8 = get_arg_default(&mut args, &body, "style")?;
    let wrap: u8 = get_arg_default(&mut args, &body, "wrap")?;
    let align: Option<u8> = get_arg_default(&mut args, &body, "align")?;
    check_args("textbox", body, args)?;

    let style = to_style(style)?;
    let wrap = to_wrap(wrap)?;
    let align = to_align(align)?;

    Ok(Box::new(TextBox::<PropBag>::new(
        id.0,
        bag,
        font_size,
        line_height,
        font.0,
        color,
        cosmic_text::Weight(weight),
        style,
        wrap,
        align,
    )))
}

fn create_id(_: &Lua, (parent, v): (Option<LuaSourceID>, LuaValue)) -> LuaResult<LuaSourceID> {
    if let Some(n) = v.as_integer() {
        if let Some(parent) = parent {
            Ok(LuaSourceID(parent.0.child(DataID::Int(n))))
        } else {
            #[cfg(debug_assertions)]
            panic!("NO PARENT! This means the layout was called incorrectly!");
            #[cfg(not(debug_assertions))]
            Err(LuaError::UserDataTypeMismatch)
        }
    } else if let Some(name) = v.as_string().map(|x| x.to_string_lossy()) {
        if let Some(parent) = parent {
            Ok(LuaSourceID(parent.0.child(DataID::Owned(name))))
        } else {
            #[cfg(debug_assertions)]
            panic!("NO PARENT! This means the layout was called incorrectly!");
            #[cfg(not(debug_assertions))]
            Err(LuaError::UserDataTypeMismatch)
        }
    } else {
        Err(LuaError::RuntimeError(format!(
            "Expected a number or string to construct an ID, but found {}",
            v.type_name()
        )))
    }
}

fn replace_dualpoint(p: LuaDualPoint, bound: f32) -> dpi::Size {
    match p {
        LuaDualPoint::Px(mut point2_d) => {
            replace_limit(&mut point2_d, bound);
            dpi::Size::Physical(dpi::PhysicalSize::<u32>::new(
                point2_d.x.ceil() as u32,
                point2_d.y.ceil() as u32,
            ))
        }
        LuaDualPoint::Dp(mut point2_d) => {
            replace_limit(&mut point2_d, bound);
            dpi::Size::Logical(dpi::LogicalSize::<f64>::new(
                point2_d.x as f64,
                point2_d.y as f64,
            ))
        }
    }
}

fn get_required_child<T: FromLua>(t: &LuaTable, idx: usize) -> LuaResult<T> {
    if !t.contains_key(idx)? {
        Err(LuaError::RuntimeError(format!(
            "Expected at least {idx} children, but found {}",
            t.raw_len()
        )))
    } else {
        t.get(idx)
    }
}

fn create_window(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> LuaResult<Window> {
    let title: LuaString = get_required(&body, "title")?;
    let child: ComponentBag = get_required_child(&body, 1)?;

    let mut attributes = winit::window::Window::default_attributes()
        .with_title(title.to_string_lossy())
        .with_resizable(get_or(&body, "resizeable", true)?)
        .with_maximized(get_or(&body, "maximized", false)?)
        .with_visible(get_or(&body, "visible", true)?)
        .with_transparent(get_or(&body, "transparent", false)?)
        .with_blur(get_or(&body, "blur", false)?)
        .with_decorations(get_or(&body, "decorated", true)?)
        .with_content_protected(get_or(&body, "protected", false)?)
        .with_active(get_or(&body, "focused", false)?);

    if body.contains_key("icon")? {
        attributes.window_icon = Some(
            crate::resource::load_icon(&std::path::PathBuf::from(body.get::<String>("icon")?))
                .map_err(|e| LuaError::ExternalError(Arc::new(Box::new(e))))?,
        );
    }

    if body.contains_key("minsize")? {
        attributes.min_inner_size = Some(replace_dualpoint(body.get("minsize")?, f32::NEG_INFINITY))
    }

    if body.contains_key("maxsize")? {
        attributes.max_inner_size = Some(replace_dualpoint(body.get("maxsize")?, f32::INFINITY))
    }

    if body.contains_key("size")? {
        attributes.inner_size = Some(replace_dualpoint(body.get("size")?, 0.0));
    }

    Ok(Window::new(id.0, attributes, Box::new(child)))
}

#[derive(PartialEq, Clone, Debug)]
pub struct LuaFragment {
    layout: LuaFunction,
    id_enter: LuaFunction,
}

impl FromLua for LuaFragment {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let t = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        Ok(Self {
            layout: t.get("layout")?,
            id_enter: t.get("id_enter")?,
        })
    }
}

impl IntoLua for LuaFragment {
    fn into_lua(self, lua: &Lua) -> LuaResult<LuaValue> {
        Ok(LuaValue::Table(lua.create_table_from([
            ("layout", self.layout),
            ("id_enter", self.id_enter),
        ])?))
    }
}

impl LuaFragment {
    pub fn call<T: IntoLuaMulti>(&self, args: T, mut id: ScopeID<'_>) -> LuaResult<ComponentBag> {
        self.id_enter.call::<ComponentBag>((
            LuaSourceID(id.id().clone()),
            self.layout.clone(),
            args,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct LuaContext {
    feather: LuaTable,
    modules: LuaTable,
    id_enter: LuaFunction,
    require: LuaFunction,
    load_in_sandbox: LuaFunction,
    lua_handlers: LuaTable,
    handler_slots: LuaTable,
    handler_count: Rc<AtomicU64>,
}

impl PartialEq for LuaContext {
    fn eq(&self, other: &Self) -> bool {
        self.feather == other.feather
            && self.modules == other.modules
            && self.id_enter == other.id_enter
            && self.require == other.require
            && self.load_in_sandbox == other.load_in_sandbox
            && self.lua_handlers == other.lua_handlers
            && self.handler_slots == other.handler_slots
            && self.handler_count.load(Ordering::Relaxed)
                == other.handler_count.load(Ordering::Relaxed)
    }
}

impl LuaContext {
    pub fn new(lua: &Lua) -> LuaResult<Self> {
        let preload = lua.create_table()?;

        preload.set("create_id", lua.create_function(create_id)?)?;
        preload.set("create_button", lua.create_function(create_button)?)?;
        preload.set(
            "create_domain_line",
            lua.create_function(create_domain_line)?,
        )?;
        preload.set(
            "create_domain_point",
            lua.create_function(create_domain_point)?,
        )?;
        preload.set("create_flexbox", lua.create_function(create_flexbox)?)?;
        preload.set("create_gridbox", lua.create_function(create_gridbox)?)?;
        preload.set("create_image", lua.create_function(create_image)?)?;
        preload.set("create_line", lua.create_function(create_line)?)?;
        preload.set("create_listbox", lua.create_function(create_listbox)?)?;
        preload.set("create_mousearea", lua.create_function(create_mousearea)?)?;
        //preload.set("create_paragraph", lua.create_function(create_paragraph)?)?;
        preload.set("create_region", lua.create_function(create_region)?)?;
        preload.set("create_scrollarea", lua.create_function(create_scrollarea)?)?;
        preload.set("create_round_rect", lua.create_function(create_round_rect)?)?;
        preload.set("create_arc", lua.create_function(create_arc)?)?;
        preload.set("create_triangle", lua.create_function(create_triangle)?)?;
        preload.set("create_circle", lua.create_function(create_circle)?)?;
        preload.set("create_text", lua.create_function(create_text)?)?;
        preload.set("create_textbox", lua.create_function(create_textbox)?)?;
        preload.set("create_window", lua.create_function(create_window)?)?;

        let preload_mt = lua.create_table()?;
        preload_mt.set("__index", lua.globals())?;
        preload.set_metatable(Some(preload_mt))?;

        let feather: LuaTable = lua
            .load(NamedChunk(FEATHER, "feather"))
            .set_environment(preload.clone())
            .eval()?;

        let id_enter = feather.get::<LuaTable>("ID")?.get::<LuaFunction>("enter")?;
        let lua_handlers = lua.create_table()?;
        let handlers = lua_handlers.clone();
        let handler_slots = lua.create_table()?;
        let slot_clone = handler_slots.clone();
        let handler_count = Rc::new(AtomicU64::new(0));
        let handler_count_clone = handler_count.clone();

        feather.set(
            "add_handler",
            lua.create_function(move |_, (name, func): (LuaString, LuaFunction)| {
                let count = handler_count_clone.fetch_add(1, Ordering::Relaxed);
                handlers.set(&name, func)?;
                slot_clone.set(name, Slot(APP_SOURCE_ID.into(), count))?;
                Ok(LuaNil)
            })?,
        )?;

        let modules = lua.create_table()?;
        preload.set("injected_dep", &modules)?;

        lua.load(NamedChunk(SANDBOX, "sandbox")).exec()?;

        let require: LuaFunction = lua
            .load(
                r#"
package = {preload = {}, loaded = injected_dep}
return function(name) -- require stub for inside sandbox
  if not package.loaded[name] then
    if not package.preload[name] then
      error("Couldn't find package.preload for " .. name)
    end
    package.loaded[name] = package.preload[name]()
  end
  return package.loaded[name]
end
        "#,
            )
            .set_environment(preload)
            .eval()?;

        lua.load(
            r#"
        jit.opt.start("maxtrace=10000")
        jit.opt.start("maxmcode=4096")
        jit.opt.start("recunroll=5")
        jit.opt.start("loopunroll=60")
        
        --local create_module = sandbox_impl(true)
        create_module = function(bytes, name, interface) 
            return load(bytes, name, "t", setmetatable(interface, {__index=_G}))
         end 


        function load_in_sandbox(bytes, name, additional_interface)
          local r, err = create_module(bytes, name, additional_interface)
          if r == nil then
            error(err)
          end
          
          return r()
        end
                "#,
        )
        .exec()?;

        let load_in_sandbox: LuaFunction = lua.load("load_in_sandbox").eval()?;

        Ok(Self {
            feather,
            id_enter,
            load_in_sandbox,
            require,
            modules,
            lua_handlers,
            handler_slots,
            handler_count,
        })
    }

    pub fn add_module(&self, lua: &Lua, name: &str, module: &[u8]) -> LuaResult<()> {
        let interface = lua.create_table()?;
        interface.set("f", self.feather.clone())?;
        interface.set("require", self.require.clone())?;

        self.modules.set(
            name,
            self.load_in_sandbox
                .call::<LuaValue>((lua.create_string(module)?, name, interface))?,
        )
    }

    pub fn get_module(&self, name: &str) -> Option<LuaValue> {
        self.modules.get(name).ok().and_then(|x| match x {
            Some(LuaNil) => None,
            x => x,
        })
    }

    pub fn reserve_handler(&self, name: &str) -> LuaResult<u64> {
        let index = self.handler_count.fetch_add(1, Ordering::Relaxed);
        self.handler_slots
            .set(name, Slot(APP_SOURCE_ID.into(), index))?;
        Ok(index)
    }

    fn load_layout<R: FromLuaMulti>(&self, lua: &Lua, layout: &[u8]) -> LuaResult<R> {
        let interface = lua.create_table()?;
        interface.set("handlers", self.handler_slots.clone())?;
        interface.set("f", self.feather.clone())?;
        interface.set("require", self.require.clone())?;

        self.load_in_sandbox
            .call((lua.create_string(layout)?, "layout", interface))
    }

    pub fn load_fragment(&self, lua: &Lua, layout: &[u8]) -> LuaResult<LuaFragment> {
        Ok(LuaFragment {
            layout: self.load_layout(lua, layout)?,
            id_enter: self.id_enter.clone(),
        })
    }

    pub fn get_handlers<AppData: FromLua + IntoLua + Clone>(
        &self,
        mut reserved: HashMap<u64, crate::AppEvent<AppData>>,
    ) -> LuaResult<Vec<crate::AppEvent<AppData>>> {
        use std::mem::MaybeUninit;

        let count = self.handler_count.load(Ordering::Relaxed) as usize;
        let mut handlers: Vec<crate::AppEvent<AppData>> = Vec::with_capacity(count);
        let spare = handlers.spare_capacity_mut();

        for v in self.handler_slots.pairs::<LuaString, Slot>() {
            let (key, slot) = v?;
            if self.lua_handlers.contains_key(&key)? {
                let func: LuaFunction = self.lua_handlers.get(&key)?;
                spare[slot.1 as usize] = MaybeUninit::new(Box::new(
                    move |pair: crate::DispatchPair, mut state: AccessCell<AppData>| {
                        let (appdata, v): (AppData, LuaValue) =
                            match func.call((pair.0, state.value.clone())) {
                                Ok(v) => v,
                                Err(e) => return InputResult::Error(e.into()),
                            };
                        *state = appdata;
                        if v.as_boolean().unwrap_or(true) {
                            InputResult::Consume(())
                        } else {
                            InputResult::Forward(())
                        }
                    },
                ));
            } else {
                spare[slot.1 as usize] = MaybeUninit::new(
                    reserved
                        .remove(&slot.1)
                        .expect("Reserved slot had no entry in `reserved` map!"),
                )
            }
        }

        unsafe {
            handlers.set_len(count);
        }

        Ok(handlers)
    }

    pub fn update_handler<AppData: FromLua + IntoLua + Clone + 'static>(
        &self,
        lua: &Lua,
        sender: std::sync::mpsc::Sender<crate::EventPair<AppData>>,
        count: AtomicU64,
    ) -> LuaResult<()> {
        let slot_clone = self.handler_slots.clone();
        self.feather.set(
            "add_handler",
            lua.create_function(move |_, (name, func): (LuaString, LuaFunction)| {
                let count = count.fetch_add(1, Ordering::Relaxed);
                sender
                    .send((
                        count,
                        Box::new(
                            move |pair: crate::DispatchPair, mut state: AccessCell<AppData>| {
                                let (appdata, v): (AppData, LuaValue) =
                                    match func.call((pair.0, state.value.clone())) {
                                        Ok(v) => v,
                                        Err(e) => return InputResult::Error(e.into()),
                                    };
                                *state = appdata;
                                if v.as_boolean().unwrap_or(true) {
                                    InputResult::Consume(())
                                } else {
                                    InputResult::Forward(())
                                }
                            },
                        ),
                    ))
                    .unwrap();
                let slot = Slot(APP_SOURCE_ID.into(), count);
                slot_clone.set(name, slot.clone())?;
                Ok(slot)
            })?,
        )?;

        Ok(())
    }
}

pub struct LuaApp<AppData: Clone + FromLua + IntoLua>(crate::App<AppData, LuaPersist<AppData>>);

impl<AppData: Clone + FromLua + IntoLua + PartialEq + 'static> LuaApp<AppData> {
    pub fn new<T>(
        lua: &Lua,
        app_state: AppData,
        handlers: Vec<(String, crate::AppEvent<AppData>)>,
        layout: &[u8],
    ) -> eyre::Result<(Self, crate::EventLoop<T>)> {
        let ctx = LuaContext::new(lua)?;

        let mut reserved = HashMap::new();
        for (k, v) in handlers.into_iter() {
            reserved.insert(ctx.reserve_handler(&k)?, v);
        }

        let (window, init): (LuaFunction, Option<LuaFunction>) = ctx.load_layout(lua, layout)?;

        let (app, event, sender, count) = crate::App::new(
            app_state.clone(),
            ctx.get_handlers(reserved)?,
            LuaPersist {
                window,
                id_enter: ctx.id_enter.clone(),
                init: if let Some(init) = init {
                    Ok(init)
                } else {
                    Err(app_state)
                },
                phantom: PhantomData,
            },
            |_| (),
        )?;

        // Change the add_handler to use the channel to send new handlers
        ctx.update_handler(lua, sender, count)?;
        Ok((Self(app), event))
    }
}

impl<AppData: Clone + FromLua + IntoLua + PartialEq + 'static, T: 'static> ApplicationHandler<T>
    for LuaApp<AppData>
{
    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        <crate::App<AppData, LuaPersist<AppData>> as ApplicationHandler<T>>::new_events(
            &mut self.0,
            event_loop,
            cause,
        );
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: T) {
        self.0.user_event(event_loop, event);
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        <crate::App<AppData, LuaPersist<AppData>> as ApplicationHandler<T>>::device_event(
            &mut self.0,
            event_loop,
            device_id,
            event,
        );
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        <crate::App<AppData, LuaPersist<AppData>> as ApplicationHandler<T>>::about_to_wait(
            &mut self.0,
            event_loop,
        );
    }

    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        <crate::App<AppData, LuaPersist<AppData>> as ApplicationHandler<T>>::suspended(
            &mut self.0,
            event_loop,
        );
    }

    fn exiting(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        <crate::App<AppData, LuaPersist<AppData>> as ApplicationHandler<T>>::exiting(
            &mut self.0,
            event_loop,
        );
    }

    fn memory_warning(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        <crate::App<AppData, LuaPersist<AppData>> as ApplicationHandler<T>>::memory_warning(
            &mut self.0,
            event_loop,
        );
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        <crate::App<AppData, LuaPersist<AppData>> as ApplicationHandler<T>>::resumed(
            &mut self.0,
            event_loop,
        );
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        <crate::App<AppData, LuaPersist<AppData>> as ApplicationHandler<T>>::window_event(
            &mut self.0,
            event_loop,
            window_id,
            event,
        )
    }
}
