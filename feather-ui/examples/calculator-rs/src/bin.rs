// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use calculator::{CalcOp, Calculator};
use std::ops::Deref;
use std::sync::{Arc, RwLock};

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

fn main() {
    calculator::register(Arc::new(Calc(RwLock::new(CalcState {
        last: 0.0,
        cur: None,
        digits: Vec::new(),
        decimals: Vec::new(),
        decimal_mode: false,
        op: CalcOp::None,
    }))));
}
