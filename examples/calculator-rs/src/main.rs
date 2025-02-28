// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use feather_ui::gen_id;
use feather_ui::layout::basic::Basic;

use feather_ui::layout::root;
use feather_ui::layout::simple;
use feather_ui::outline::button::Button;
use feather_ui::outline::mouse_area;
use feather_ui::outline::region::Region;
use feather_ui::outline::round_rect::RoundRect;
use feather_ui::outline::text::Text;
use feather_ui::outline::window::Window;
use feather_ui::outline::OutlineFrom;
use feather_ui::persist::FnPersist;
use feather_ui::App;
use feather_ui::Slot;
use feather_ui::SourceID;
use feather_ui::WrapEventEx;
use feather_ui::FILL_URECT;
use std::any::Any;
use std::f32;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::RwLock;
use ultraviolet::Vec4;

uniffi::include_scaffolding!("calculator");

// Doesn't include instant actions that take no argument, like clear, backspace, pi, square, root, etc.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
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

trait InternalEq {
    fn internal_eq(&self, state: &CalcState) -> bool;
}

impl InternalEq for UniFFICallbackHandlerCalculator {
    fn internal_eq(&self, _: &CalcState) -> bool {
        panic!("Tried to compare rust implementation with foreign implementation!!!")
    }
}

trait Calculator: Send + Sync + InternalEq {
    fn add_digit(&self, digit: u8);
    fn backspace(&self);
    fn apply_op(&self);
    fn set_op(&self, op: CalcOp);
    fn get(&self) -> f64;
    fn toggle_decimal(&self);
    fn copy(&self) -> Arc<dyn Calculator>;
    fn eq(&self, rhs: Arc<dyn Calculator>) -> bool;
}

fn new_calc() -> Arc<dyn Calculator> {
    Arc::new(RwLock::new(CalcState {
        last: 0.0,
        cur: None,
        digits: Vec::new(),
        decimals: Vec::new(),
        decimal_mode: false,
        op: CalcOp::None,
    }))
}

impl Calculator for RwLock<CalcState> {
    fn add_digit(&self, digit: u8) {
        self.write().unwrap().add_digit(digit)
    }
    fn backspace(&self) {
        self.write().unwrap().backspace()
    }
    fn apply_op(&self) {
        self.write().unwrap().apply_op()
    }
    fn set_op(&self, op: CalcOp) {
        self.write().unwrap().set_op(op)
    }
    fn get(&self) -> f64 {
        let state = self.read().unwrap();
        state.cur.unwrap_or(state.last)
    }
    fn toggle_decimal(&self) {
        let prev = self.read().unwrap().decimal_mode;
        self.write().unwrap().decimal_mode = !prev;
    }

    fn copy(&self) -> Arc<dyn Calculator> {
        Arc::new(RwLock::new(self.read().unwrap().clone()))
    }

    fn eq(&self, rhs: Arc<dyn Calculator>) -> bool {
        rhs.internal_eq(&self.read().unwrap())
    }
}

impl InternalEq for RwLock<CalcState> {
    fn internal_eq(&self, state: &CalcState) -> bool {
        self.read().unwrap().eq(state)
    }
}
struct CalcFFI(Arc<dyn Calculator>);

impl PartialEq for CalcFFI {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(other.0.clone())
    }
}

impl Clone for CalcFFI {
    fn clone(&self) -> Self {
        CalcFFI(self.0.copy())
    }
}

struct CalcApp {}

const CLEAR_COLOR: Vec4 = Vec4::new(0.8, 0.3, 0.3, 1.0);
const OP_COLOR: Vec4 = Vec4::new(0.3, 0.3, 0.3, 0.7);
const NUM_COLOR: Vec4 = Vec4::new(0.3, 0.3, 0.3, 1.0);
const EQ_COLOR: Vec4 = Vec4::new(0.3, 0.8, 0.3, 1.0);

use std::sync::LazyLock;
type BoxedAction = Box<dyn Sync + Send + Fn(&Arc<dyn Calculator>)>;
static BUTTONS: LazyLock<[(&str, BoxedAction, Vec4); 24]> = LazyLock::new(|| {
    [
        (
            "C",
            Box::new(|calc| calc.set_op(CalcOp::Clear)),
            CLEAR_COLOR,
        ),
        ("x²", Box::new(|calc| calc.set_op(CalcOp::Square)), OP_COLOR),
        ("√x", Box::new(|calc| calc.set_op(CalcOp::Sqrt)), OP_COLOR),
        ("←", Box::new(|calc| calc.backspace()), OP_COLOR),
        ("yˣ", Box::new(|calc| calc.set_op(CalcOp::Pow)), OP_COLOR),
        ("¹∕ₓ", Box::new(|calc| calc.set_op(CalcOp::Inv)), OP_COLOR),
        ("%", Box::new(|calc| calc.set_op(CalcOp::Mod)), OP_COLOR),
        ("÷", Box::new(|calc| calc.set_op(CalcOp::Div)), OP_COLOR),
        ("7", Box::new(|calc| calc.add_digit(7)), NUM_COLOR),
        ("8", Box::new(|calc| calc.add_digit(8)), NUM_COLOR),
        ("9", Box::new(|calc| calc.add_digit(9)), NUM_COLOR),
        ("×", Box::new(|calc| calc.set_op(CalcOp::Mul)), OP_COLOR),
        ("4", Box::new(|calc| calc.add_digit(4)), NUM_COLOR),
        ("5", Box::new(|calc| calc.add_digit(5)), NUM_COLOR),
        ("6", Box::new(|calc| calc.add_digit(6)), NUM_COLOR),
        ("−", Box::new(|calc| calc.set_op(CalcOp::Sub)), OP_COLOR),
        ("1", Box::new(|calc| calc.add_digit(1)), NUM_COLOR),
        ("2", Box::new(|calc| calc.add_digit(2)), NUM_COLOR),
        ("3", Box::new(|calc| calc.add_digit(3)), NUM_COLOR),
        ("+", Box::new(|calc| calc.set_op(CalcOp::Add)), OP_COLOR),
        (
            "⁺∕₋",
            Box::new(|calc| calc.set_op(CalcOp::Negate)),
            OP_COLOR,
        ),
        ("0", Box::new(|calc| calc.add_digit(0)), NUM_COLOR),
        (".", Box::new(|calc| calc.toggle_decimal()), OP_COLOR),
        ("=", Box::new(|calc| calc.apply_op()), EQ_COLOR),
    ]
});

impl FnPersist<CalcFFI, im::HashMap<Rc<SourceID>, Option<Window>>> for CalcApp {
    type Store = (CalcFFI, im::HashMap<Rc<SourceID>, Option<Window>>);

    fn init(&self) -> Self::Store {
        (CalcFFI(new_calc()), im::HashMap::new())
    }
    fn call(
        &self,
        mut store: Self::Store,
        args: &CalcFFI,
    ) -> (Self::Store, im::HashMap<Rc<SourceID>, Option<Window>>) {
        if store.0.eq(args) {
            let mut children: im::Vector<Option<Box<OutlineFrom<Basic>>>> = im::Vector::new();

            let button_id = Rc::new(gen_id!());

            for (i, (txt, _, color)) in BUTTONS.iter().enumerate() {
                let rect = RoundRect::<()> {
                    id: gen_id!(button_id).into(),
                    fill: *color,
                    corners: Vec4::broadcast(10.0),
                    props: (),
                    rect: feather_ui::FILL_URECT,
                    ..Default::default()
                };

                let text = Text::<()> {
                    id: gen_id!(button_id).into(),
                    props: (),
                    text: txt.to_string(),
                    font_size: 30.0,
                    line_height: 42.0,
                    ..Default::default()
                };

                let mut btn_children: im::Vector<Option<Box<OutlineFrom<Basic>>>> =
                    im::Vector::new();
                btn_children.push_back(Some(Box::new(text)));
                btn_children.push_back(Some(Box::new(rect)));

                const ROW_COUNT: usize = 4;
                let (x, y) = (i % ROW_COUNT, (i / ROW_COUNT) + 1);
                let (w, h) = (1.0 / ROW_COUNT as f32, 1.0 / 7.0);

                let btn = Button::<()>::new(
                    gen_id!(button_id).into(),
                    (),
                    simple::Simple {
                        area: feather_ui::URect {
                            topleft: feather_ui::UPoint {
                                abs: ultraviolet::Vec2 { x: 4.0, y: 4.0 },
                                rel: feather_ui::RelPoint {
                                    x: w * x as f32,
                                    y: h * y as f32,
                                },
                            },
                            bottomright: feather_ui::UPoint {
                                abs: ultraviolet::Vec2 { x: -4.0, y: -4.0 },
                                rel: feather_ui::RelPoint {
                                    x: w * (x + 1) as f32,
                                    y: h * (y + 1) as f32,
                                },
                            },
                        },
                        margin: Default::default(),
                        limits: feather_ui::DEFAULT_LIMITS,
                        anchor: Default::default(),
                        zindex: 0,
                    },
                    Slot(feather_ui::APP_SOURCE_ID.into(), i as u64),
                    btn_children,
                );

                children.push_back(Some(Box::new(btn)));
            }

            let display = Text::<()> {
                id: gen_id!(button_id).into(),
                props: (),
                text: args.0.get().to_string(),
                font_size: 60.0,
                line_height: 42.0,
                ..Default::default()
            };

            let text_bg = RoundRect::<()> {
                id: gen_id!(button_id).into(),
                fill: Vec4::new(0.2, 0.2, 0.2, 1.0),
                corners: Vec4::broadcast(25.0),
                props: (),
                rect: feather_ui::URect {
                    topleft: feather_ui::UPoint {
                        abs: ultraviolet::Vec2 { x: 0.0, y: 0.0 },
                        rel: feather_ui::RelPoint { x: 0.0, y: 0.0 },
                    },
                    bottomright: feather_ui::UPoint {
                        abs: ultraviolet::Vec2 { x: 0.0, y: 0.0 },
                        rel: feather_ui::RelPoint {
                            x: 1.0,
                            y: 1.0 / 7.0,
                        },
                    },
                },
                ..Default::default()
            };

            children.push_back(Some(Box::new(display)));
            children.push_back(Some(Box::new(text_bg)));

            let region = Region {
                id: gen_id!().into(),
                props: root::Inherited { area: FILL_URECT },
                basic: Basic {
                    padding: Default::default(),
                    zindex: 0,
                },
                children,
            };
            let window = Window::new(
                gen_id!().into(),
                winit::window::Window::default_attributes()
                    .with_title("calculator-rs")
                    .with_resizable(true),
                Box::new(region),
            );

            store.1 = im::HashMap::new();
            store.1.insert(window.id.clone(), Some(window));
            store.0 = args.clone();
        }
        let windows = store.1.clone();
        (store, windows)
    }
}

fn main() {
    #[allow(clippy::type_complexity)]
    let mut inputs: Vec<
        Box<dyn FnMut((u64, Box<dyn Any>), CalcFFI) -> Result<CalcFFI, CalcFFI>>,
    > = Vec::new();

    for (_, f, _) in BUTTONS.iter() {
        inputs.push(Box::new(
            |_: mouse_area::MouseAreaEvent, mut appdata: CalcFFI| -> Result<CalcFFI, CalcFFI> {
                {
                    f(&appdata.0);
                    Ok(appdata)
                }
            }
            .wrap(),
        ));
    }

    let (mut app, event_loop): (App<CalcFFI, CalcApp>, winit::event_loop::EventLoop<()>) =
        App::new(
            CalcFFI(Arc::new(RwLock::new(CalcState {
                last: 0.0,
                cur: Some(0.0),
                digits: Vec::new(),
                decimals: Vec::new(),
                decimal_mode: false,
                op: CalcOp::None,
            }))),
            inputs,
            CalcApp {},
        )
        .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
