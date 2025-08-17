// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::{sRGB, sRGB32};
use crate::component::ChildOf;
use crate::component::button::Button;
use crate::component::region::Region;
use crate::component::shape::{Shape, ShapeKind};
use crate::component::text::Text;
use crate::component::window::Window;
use crate::layout::{fixed, flex};
use crate::propbag::PropBag;
use crate::{
    APP_SOURCE_ID, DAbsPoint, DAbsRect, DPoint, DRect, DValue, DataID, FnPersist, Logical, Pixel,
    Rect, Relative, Slot, SourceID, StateMachineChild, UNSIZED_AXIS,
};
use guillotiere::euclid::Point2D;
use mlua::UserData;
use mlua::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;
use wide::f32x4;
use winit::application::ApplicationHandler;

use winit::dpi;

const SANDBOX: &[u8] = include_bytes!("./sandbox.lua");
const FEATHER: &[u8] = include_bytes!("./feather.lua");

struct NamedChunk<'a>(&'a [u8], &'a str);

impl mlua::AsChunk<'static> for NamedChunk<'static> {
    fn name(&self) -> Option<String> {
        Some(self.1.into())
    }

    fn source(self) -> std::io::Result<std::borrow::Cow<'static, [u8]>> {
        Ok(std::borrow::Cow::Borrowed(self.0))
    }
}

#[derive(Clone)]
struct LuaSourceID(Arc<SourceID>);

struct LuaEnum<T>(T);

impl<T: TryFrom<u8>> FromLua for LuaEnum<T>
where
    <T as TryFrom<u8>>::Error: std::error::Error + 'static + Sync + Send,
{
    fn from_lua(value: LuaValue, _: &mlua::Lua) -> mlua::Result<Self> {
        T::try_from(value.as_i32().ok_or(LuaError::UserDataTypeMismatch)? as u8)
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

fn is_dvalue(t: &LuaTable) -> LuaResult<bool> {
    if let Some(mt) = t.metatable() {
        mt.contains_key("__isvalue")
    } else {
        Ok(false)
    }
}

fn get_kind(t: &LuaTable) -> LuaResult<String> {
    if let Some(mt) = t.metatable() {
        mt.get("kind")
    } else {
        Err(LuaError::UserDataTypeMismatch)
    }
}

impl FromLua for DValue {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let t = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        if !is_dvalue(t)? {
            return Err(LuaError::UserDataTypeMismatch);
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

impl<U> Into<Point2D<f32, U>> for LuaPoint<U> {
    fn into(self) -> Point2D<f32, U> {
        self.0
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

        let t = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        if is_dvalue(t)? {
            const TYPES: [&str; 3] = ["dp", "px", "rel"];

            for ty in TYPES {
                if t.contains_key(ty)? != (U::KIND == ty) {
                    return Err(LuaError::UserDataTypeMismatch);
                }
            }
            Ok(LuaPoint(Point2D::<f32, U>::splat(t.get(U::KIND)?)))
        } else if get_kind(t)? == U::KIND {
            Ok(LuaPoint(Point2D::<f32, U>::new(t.get("x")?, t.get("y")?)))
        } else {
            Err(LuaError::UserDataTypeMismatch)
        }
    }
}

impl<U: LuaKind> FromLua for Rect<U> {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        if get_kind(v)? != U::KIND {
            return Err(LuaError::UserDataTypeMismatch);
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
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        let px = get_or_default(v, "px")?;
        let dp = get_or_default(v, "dp")?;
        Ok(DAbsRect { dp, px })
    }
}

impl FromLua for DRect {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        let px = get_or_default(v, "px")?;
        let dp = get_or_default(v, "dp")?;
        let rel = get_or_default(v, "rel")?;
        Ok(DRect { dp, px, rel })
    }
}

impl FromLua for DAbsPoint {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        let px = get_or_default::<LuaPoint<Pixel>>(v, "px")?.0;
        let dp = get_or_default::<LuaPoint<Logical>>(v, "dp")?.0;
        Ok(DAbsPoint { dp, px })
    }
}

impl FromLua for DPoint {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        let px = get_or_default::<LuaPoint<Pixel>>(v, "px")?.0;
        let dp = get_or_default::<LuaPoint<Logical>>(v, "dp")?.0;
        let rel = get_or_default::<LuaPoint<Relative>>(v, "rel")?.0;
        Ok(DPoint { dp, px, rel })
    }
}

struct LimitPoint(DPoint);

impl FromLua for LimitPoint {
    fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
        let v = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        let px = get_or::<LuaPoint<Pixel>>(v, "px", LuaPoint::nan())?.0;
        let dp = get_or::<LuaPoint<Logical>>(v, "dp", LuaPoint::nan())?.0;
        let rel = get_or::<LuaPoint<Relative>>(v, "rel", LuaPoint::nan())?.0;
        Ok(LimitPoint(DPoint { dp, px, rel }))
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
                Err(LuaError::UserDataTypeMismatch)
            }
        } else {
            Err(LuaError::UserDataTypeMismatch)
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
        let name = value.as_str().ok_or(LuaError::UserDataTypeMismatch)?;

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

#[derive(Clone)]
struct LuaDomain(std::sync::Arc<crate::CrossReferenceDomain>);

impl UserData for LuaDomain {}
gen_from_lua!(LuaDomain);

impl UserData for Window {}
gen_from_lua!(Window);

impl UserData for LuaSourceID {}
gen_from_lua!(LuaSourceID);

impl UserData for Slot {}
gen_from_lua!(Slot);

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

/// This defines the "lua" app that knows how to handle a lua value that contains the
/// expected rust objects, and hand them off for processing. This is analogous to the
/// pure-rust [App] struct defined in lib.rs
pub struct LuaPersist<AppData> {
    pub window: LuaFunction, // takes a Store and an appstate and returns a Window
    pub init: LuaFunction,
    phantom: PhantomData<AppData>,
}

impl<AppData: Clone + FromLua + IntoLua>
    FnPersist<AppData, im::HashMap<Arc<SourceID>, Option<Window>>> for LuaPersist<AppData>
{
    type Store = AppData;

    fn init(&self) -> Self::Store {
        let r = self.init.call::<AppData>(());
        match r {
            Err(LuaError::RuntimeError(s)) => panic!("{}", s),
            Err(e) => panic!("{e:?}"),
            Ok(v) => v,
        }
    }
    fn call(
        &mut self,
        store: Self::Store,
        args: &AppData,
    ) -> (Self::Store, im::HashMap<Arc<SourceID>, Option<Window>>) {
        let mut h = im::HashMap::new();
        let (store, w) = self
            .window
            .call::<(AppData, crate::component::window::Window)>((store, args.clone()))
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
        let t = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
        if is_dvalue(t)? {
            if t.contains_key("dp")? && !t.contains_key("px")? {
                Ok(LuaDualPoint::Dp(Point2D::<f32, Logical>::splat(
                    t.get("dp")?,
                )))
            } else if t.contains_key("px")? && !t.contains_key("dp")? {
                Ok(LuaDualPoint::Px(Point2D::<f32, Pixel>::splat(t.get("px")?)))
            } else {
                return Err(LuaError::UserDataTypeMismatch);
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
            Err(LuaError::UserDataTypeMismatch)
        }
    }
}

fn load_prop<T: mlua::FromLua>(
    f: fn(&mut PropBag, T) -> Option<T>,
    bag: &mut PropBag,
    props: &LuaTable,
    name: &str,
) -> mlua::Result<()> {
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
        let props = value.as_table().ok_or(LuaError::UserDataTypeMismatch)?;
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
            bag.set_direction(props.get::<LuaEnum<crate::RowDirection>>("domain")?.0);
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

        Ok(bag)
    }
}

fn prop_children(
    t: &LuaTable,
) -> mlua::Result<(im::Vector<Option<Box<ChildOf<dyn fixed::Prop>>>>, PropBag)> {
    let mut children: im::Vector<Option<Box<ChildOf<dyn fixed::Prop>>>> = im::Vector::new();

    for i in 1..=t.len()? {
        let component: ComponentBag = t.get(i)?;
        children.push_back(Some(Box::new(component)));
    }

    let bag: PropBag = get_or_default(t, "props")?;
    Ok((children, bag))
}

fn create_region(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> mlua::Result<ComponentBag> {
    let (children, bag) = prop_children(&body)?;

    Ok(Box::new(Region::<PropBag>::new(id.0, bag.into(), children)))
}

// In CSS, 1.2 is usually used as the "reasonable" default line-height for a given font.
const REASONABLE_LINE_HEIGHT: f32 = 1.2;
// CSS defaults to 16 pixels, which is 12.8 points. We round up to 14 points as the next highest even number.
const DEFAULT_FONT_SIZE: f32 = 14.0;

fn create_text(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> mlua::Result<ComponentBag> {
    let (_, bag) = prop_children(&body)?;

    let font_size = get_or(&body, "fontsize", DEFAULT_FONT_SIZE)?;
    let line_height = get_or(&body, "lineheight", font_size * REASONABLE_LINE_HEIGHT)?;
    let text = body.get("text")?;
    let font: LuaFontFamily = get_or_default(&body, "font")?;
    let color = body.get("color")?;
    let weight: u16 = get_or(&body, "weight", cosmic_text::Weight::NORMAL.0)?;
    let style: u8 = get_or_default(&body, "style")?;
    let wrap: u8 = get_or_default(&body, "wrap")?;

    let style = match style {
        0 => cosmic_text::Style::Normal,
        1 => cosmic_text::Style::Italic,
        2 => cosmic_text::Style::Oblique,
        _ => return Err(LuaError::UserDataTypeMismatch),
    };

    let wrap = match wrap {
        0 => cosmic_text::Wrap::None,
        1 => cosmic_text::Wrap::Glyph,
        2 => cosmic_text::Wrap::Word,
        3 => cosmic_text::Wrap::WordOrGlyph,
        _ => return Err(LuaError::UserDataTypeMismatch),
    };

    Ok(Box::new(Text::<PropBag>::new(
        id.0,
        bag.into(),
        font_size,
        line_height,
        text,
        font.0,
        color,
        cosmic_text::Weight(weight),
        style,
        wrap,
    )))
}

fn create_button(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> mlua::Result<ComponentBag> {
    let (children, bag) = prop_children(&body)?;

    let onclick = body.get("onclick")?;
    Ok(Box::new(Button::<PropBag>::new(
        id.0,
        bag.into(),
        onclick,
        children,
    )))
}

fn get_array_or<T: num_traits::FromPrimitive + FromLua + Clone + Copy, const N: usize>(
    lua: &Lua,
    t: &LuaTable,
    key: &str,
    d: [T; N],
) -> mlua::Result<[T; N]> {
    Ok(if t.contains_key(key)? {
        let v = t.get::<LuaValue>(key)?;
        if let Some(n) = v.as_number() {
            let num = T::from_f64(n).ok_or(LuaError::UserDataTypeMismatch)?;
            let test: [T; N] = [num; N];
            test
        } else {
            <[T; N] as FromLua>::from_lua(v, lua)?
        }
    } else {
        d
    })
}

fn create_round_rect(lua: &Lua, (id, body): (LuaSourceID, LuaTable)) -> mlua::Result<ComponentBag> {
    let (_, bag) = prop_children(&body)?;

    let border = get_or_default(&body, "border")?;
    let blur = get_or_default(&body, "blur")?;
    let fill = get_or_default(&body, "style")?;
    let outline = get_or_default(&body, "wrap")?;
    let corners = get_array_or(lua, &body, "corners", [0.0; 4])?;

    Ok(Box::new(
        Shape::<PropBag, { ShapeKind::RoundRect as u8 }>::new(
            id.0,
            bag.into(),
            border,
            blur,
            wide::f32x4::new(corners),
            fill,
            outline,
        ),
    ))
}

fn create_id(lua: &Lua, (parent, v): (Option<LuaSourceID>, LuaValue)) -> mlua::Result<LuaSourceID> {
    let stringcheck = v.as_string().map(|x| x.to_string_lossy());
    if let Some(n) = v.as_integer() {
        Ok(LuaSourceID(if let Some(parent) = parent {
            parent.0.child(DataID::Int(n))
        } else {
            Arc::new(SourceID::new(DataID::Int(n)))
        }))
    } else if let Ok(id) = LuaSourceID::from_lua(v, lua) {
        if let Some(parent) = parent {
            id.0.parent
                .set(parent.0)
                .map_err(|_| LuaError::UserDataTypeMismatch)?;
            Ok(id)
        } else {
            Err(LuaError::UserDataTypeMismatch)
        }
    } else if let Some(name) = stringcheck {
        Ok(LuaSourceID(if let Some(parent) = parent {
            parent.0.child(DataID::Owned(name))
        } else {
            Arc::new(SourceID::new(DataID::Owned(name)))
        }))
    } else {
        Err(LuaError::UserDataTypeMismatch)
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

fn create_window(_: &Lua, (id, body): (LuaSourceID, LuaTable)) -> mlua::Result<Window> {
    let title: LuaString = body.get("title")?;
    let child: ComponentBag = body.get(1)?;

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

//AppData: 'static + PartialEq
pub struct LuaApp<AppData: Clone + FromLua + IntoLua>(crate::App<AppData, LuaPersist<AppData>>);

impl<AppData: Clone + FromLua + IntoLua + PartialEq + 'static> LuaApp<AppData> {
    pub fn new<T>(
        lua: &Lua,
        app_state: AppData,
        handlers: Vec<(String, crate::AppEvent<AppData>)>,
        layout: &[u8],
    ) -> eyre::Result<(Self, crate::EventLoop<T>)> {
        let interface = lua.create_table()?;
        let handler_table = lua.create_table()?;

        for (i, (name, _)) in handlers.iter().enumerate() {
            handler_table.set(name.as_str(), Slot(APP_SOURCE_ID.into(), i as u64))?;
        }

        interface.set("handlers", handler_table)?;
        interface.set("create_id", lua.create_function(create_id)?)?;
        interface.set("create_window", lua.create_function(create_window)?)?;
        interface.set("create_region", lua.create_function(create_region)?)?;
        interface.set("create_button", lua.create_function(create_button)?)?;
        interface.set("create_text", lua.create_function(create_text)?)?;
        interface.set("create_round_rect", lua.create_function(create_round_rect)?)?;

        lua.load(NamedChunk(SANDBOX, "sandbox")).exec()?;
        lua.load(NamedChunk(FEATHER, "feather")).exec()?;

        lua.load(
            r#"
        jit.opt.start("maxtrace=10000")
        jit.opt.start("maxmcode=4096")
        jit.opt.start("recunroll=5")
        jit.opt.start("loopunroll=60")
        
        local create_module = sandbox_impl(true)
        
        function load_in_sandbox(bytes, additional_interface)
          local r, err = create_module(bytes, "layout", additional_interface)
          if r == nil then
            error(err)
          end
          
          return r()
        end
                "#,
        )
        .exec()?;

        let load_in_sandbox: LuaFunction = lua.load("load_in_sandbox").eval()?;
        let (window, init): (LuaFunction, LuaFunction) =
            load_in_sandbox.call((lua.create_string(layout)?, interface))?;

        let (app, event) = crate::App::new(
            app_state,
            handlers.into_iter().map(|(_, f)| f).collect(),
            LuaPersist {
                window,
                init,
                phantom: PhantomData,
            },
            |_| (),
        )?;

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
