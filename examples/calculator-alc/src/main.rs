// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use feather_ui::lua::AppState;
use feather_ui::lua::LuaApp;
use feather_ui::App;
use mlua::prelude::*;
use mlua::Function;
use std::any::Any;
use std::any::TypeId;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::RwLock;

uniffi::include_scaffolding!("calculator");

uniffi_alicorn::include_alicorn_scaffolding!("calculator");

pub trait Calculator: Send + Sync {
    fn add_digit(&self, digit: u8);
    fn backspace(&self);
    fn apply_op(&self);
    fn set_op(&self, op: CalcOp);
    fn get(&self) -> f64;
    fn toggle_decimal(&self);
    fn copy(&self) -> Arc<dyn Calculator>;
    fn eq(&self, rhs: Arc<dyn Calculator>) -> bool;
}

impl dyn Calculator {
    #[inline]
    pub fn is<T: Any>(&self) -> bool {
        // Get `TypeId` of the type this function is instantiated with.
        let t = TypeId::of::<T>();

        // Get `TypeId` of the type in the trait object (`self`).
        let concrete = self.type_id();

        // Compare both `TypeId`s on equality.
        t == concrete
    }

    #[inline]
    pub unsafe fn downcast_ref_unchecked<T: Any>(&self) -> &T {
        debug_assert!(self.is::<T>());
        // SAFETY: caller guarantees that T is the correct type
        unsafe { &*(self as *const dyn Calculator as *const T) }
    }

    #[inline]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        if self.is::<T>() {
            // SAFETY: just checked whether we are pointing to the correct type, and we can rely on
            // that check for memory safety because we have implemented Any for all types; no other
            // impls can exist as they would conflict with our impl.
            unsafe { Some(self.downcast_ref_unchecked()) }
        } else {
            None
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(u64)]
pub enum CalcOp {
    #[default]
    None,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Square,
    Sqrt,
    Inv,
    Negate,
    Clear,
}

impl From<u64> for CalcOp {
    fn from(i: u64) -> Self {
        match i {
            1 => CalcOp::Add,
            2 => CalcOp::Sub,
            3 => CalcOp::Mul,
            4 => CalcOp::Div,
            5 => CalcOp::Pow,
            6 => CalcOp::Square,
            7 => CalcOp::Sqrt,
            8 => CalcOp::Inv,
            9 => CalcOp::Negate,
            10 => CalcOp::Clear,
            _ => CalcOp::None,
        }
    }
}

#[derive(PartialEq, Clone, Debug, Default)]
struct CalcState {
    last: f64,
    cur: Option<f64>,
    digits: Vec<u8>,
    decimals: Vec<u8>,
    decimal_mode: bool,
    op: CalcOp,
}

impl CalcState {
    fn update_cur(&mut self) {
        let mut cur = 0.0;
        let mut mul = 1.0;
        for v in self.digits.iter().rev() {
            cur += *v as f64 * mul;
            mul *= 10.0;
        }
        let mut mul = 0.1;
        for v in self.decimals.iter() {
            cur += *v as f64 * mul;
            mul /= 10.0;
        }
        self.cur = Some(cur);
    }

    pub fn add_digit(&mut self, digit: u8) {
        if digit == 0 && !self.digits.is_empty() && !self.decimal_mode {
            return;
        }
        if self.decimal_mode {
            self.decimals.push(digit);
        } else {
            self.digits.push(digit);
        }
        self.update_cur();
    }
    pub fn backspace(&mut self) {
        if self.decimal_mode {
            self.decimals.pop();
        } else {
            self.digits.pop();
        }
        self.update_cur();
    }

    pub fn apply_op(&mut self) {
        if let Some(cur) = self.cur {
            self.last = match self.op {
                CalcOp::None => cur,
                CalcOp::Add => self.last + cur,
                CalcOp::Sub => self.last - cur,
                CalcOp::Mul => self.last * cur,
                CalcOp::Div => self.last / cur,
                CalcOp::Mod => self.last % cur,
                CalcOp::Pow => self.last.powf(cur),
                _ => cur, // If this is an instant op, move cur to last
            };
        }
        self.last = match self.op {
            CalcOp::Square => self.last * self.last,
            CalcOp::Sqrt => self.last.sqrt(),
            CalcOp::Inv => self.last.recip(),
            CalcOp::Negate => -self.last,
            CalcOp::Clear => {
                self.last = 0.0;
                self.cur = Some(0.0);
                self.op = CalcOp::None;
                self.decimal_mode = false;
                self.decimals.clear();
                self.digits.clear();
                return;
            }
            _ => self.last,
        };
        self.cur = None;
        self.op = CalcOp::None;
        self.decimal_mode = false;
        self.decimals.clear();
        self.digits.clear();
    }
    pub fn set_op(&mut self, op: CalcOp) {
        match op {
            CalcOp::Square | CalcOp::Sqrt | CalcOp::Inv | CalcOp::Negate | CalcOp::Clear => {
                self.op = op;
                self.apply_op();
            }
            _ => {
                self.apply_op();
                self.op = op;
            }
        };
    }
}

struct Calc(RwLock<CalcState>);

impl Calculator for Calc {
    fn add_digit(&self, digit: u8) {
        self.0.write().unwrap().add_digit(digit)
    }
    fn backspace(&self) {
        self.0.write().unwrap().backspace()
    }
    fn apply_op(&self) {
        self.0.write().unwrap().apply_op()
    }
    fn set_op(&self, op: CalcOp) {
        self.0.write().unwrap().set_op(op)
    }
    fn get(&self) -> f64 {
        let state = self.0.read().unwrap();
        state.cur.unwrap_or(state.last)
    }
    fn toggle_decimal(&self) {
        let prev = self.0.read().unwrap().decimal_mode;
        self.0.write().unwrap().decimal_mode = !prev;
    }

    fn copy(&self) -> Arc<dyn Calculator> {
        Arc::new(Calc(RwLock::new(self.0.read().unwrap().clone())))
    }

    fn eq(&self, rhs: Arc<dyn Calculator>) -> bool {
        let rhs = <Arc<dyn Calculator>>::as_ref(&rhs);
        let rhs = match rhs.downcast_ref::<Self>() {
            Some(rhs) => rhs,
            None => return false,
        };
        let lhs = self.0.read().unwrap();
        let rhs = rhs.0.read().unwrap();
        lhs.deref() == rhs.deref()
    }
}

fn wrap_luafunc(
    f: Function,
) -> impl FnMut(feather_ui::DispatchPair, AppState) -> Result<AppState, AppState> {
    move |pair, state| Ok(f.call((pair.0, state)).unwrap())
}

fn main() {
    let lua = Lua::new();
    let mut feather_interface = lua.create_table().unwrap();
    feather_ui::lua::init_environment(&lua, &mut feather_interface).unwrap();
    let bindings = uniffi_alicorn_calc_setup(&lua, &feather_interface).unwrap();
    let alicorn = Box::new(alicorn::Alicorn::new(lua, &feather_interface).unwrap());

    // Load the autogenerated bindings
    //std::fs::write("OUTPUT.alc", bindings).unwrap();

    // Load the built-in GLSL prelude from alicorn
    alicorn.load_glsl_prelude().unwrap();

    let bindings = include_str!("../OUTPUT.alc");
    alicorn
        .include(bindings, "uniffi_alicorn_calc_setup")
        .unwrap();

    // Because of constraints on lifetimes, this needs to technically last forever.
    let alicorn = Box::leak(alicorn);
    {
        // This compiles and executes an alicorn program, which then calls back into the lua environment we have created in feather-ui/src/lua.rs
        let (window, init, state, onclick1, onclick2, onclick3, onclickplus, onclickeq): (
            Function,
            Function,
            LuaValue,
            Function,
            Function,
            Function,
            Function,
            Function,
        ) = alicorn
            .execute(include_str!("../layout.alc"), "layout.alc")
            .unwrap();

        let onclick1 = Box::new(wrap_luafunc(onclick1));
        let onclick2 = Box::new(wrap_luafunc(onclick2));
        let onclick3 = Box::new(wrap_luafunc(onclick3));
        let onclickplus = Box::new(wrap_luafunc(onclickplus));
        let onclickeq = Box::new(wrap_luafunc(onclickeq));

        let outline = LuaApp { window, init };
        let (mut app, event_loop): (App<AppState, LuaApp>, winit::event_loop::EventLoop<()>) =
            App::new(
                state,
                vec![onclick1, onclick2, onclick3, onclickplus, onclickeq],
                outline,
            )
            .unwrap();

        event_loop.run_app(&mut app).unwrap();
    }
    //drop(unsafe { Box::from_raw(alicorn) });
}

pub fn register() -> Arc<dyn Calculator> {
    Arc::new(Calc(RwLock::new(CalcState {
        last: 0.0,
        cur: None,
        digits: Vec::new(),
        decimals: Vec::new(),
        decimal_mode: false,
        op: CalcOp::None,
    })))
}
