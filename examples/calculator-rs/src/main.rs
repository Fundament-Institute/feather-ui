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
use ultraviolet::Vec4;

// Doesn't include instant actions that take no argument, like clear, backspace, pi, square, root, etc.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
enum CalcOp {
    #[default]
    None,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
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
            };
        }
        self.cur = None;
        self.op = CalcOp::None;
        self.decimal_mode = false;
        self.decimals.clear();
        self.digits.clear();
    }
    pub fn set_op(&mut self, op: CalcOp) {
        self.apply_op();
        self.op = op;
    }
    pub fn instant_op(&mut self, op: impl Fn(f64) -> f64) {
        self.cur = Some(if let Some(cur) = self.cur {
            op(cur)
        } else {
            op(self.last)
        })
    }
}

struct CalcApp {}

const CLEAR_COLOR: Vec4 = Vec4::new(0.8, 0.3, 0.3, 1.0);
const OP_COLOR: Vec4 = Vec4::new(0.3, 0.3, 0.3, 0.7);
const NUM_COLOR: Vec4 = Vec4::new(0.3, 0.3, 0.3, 1.0);
const EQ_COLOR: Vec4 = Vec4::new(0.3, 0.8, 0.3, 1.0);

use std::sync::LazyLock;
type BoxedAction = Box<dyn Sync + Send + Fn(&mut CalcState)>;
static BUTTONS: LazyLock<[(&str, BoxedAction, Vec4); 24]> = LazyLock::new(|| {
    [
        (
            "C",
            Box::new(|calc| {
                calc.last = 0.0;
                calc.cur = Some(0.0);
                calc.op = CalcOp::None;
                calc.decimal_mode = false;
                calc.decimals.clear();
                calc.digits.clear();
            }),
            CLEAR_COLOR,
        ),
        ("x²", Box::new(|calc| calc.instant_op(|x| x * x)), OP_COLOR),
        (
            "√x",
            Box::new(|calc| calc.instant_op(|x| x.sqrt())),
            OP_COLOR,
        ),
        ("←", Box::new(|calc| calc.backspace()), OP_COLOR),
        ("yˣ", Box::new(|calc| calc.set_op(CalcOp::Pow)), OP_COLOR),
        (
            "¹∕ₓ",
            Box::new(|calc| calc.instant_op(|x| x.recip())),
            OP_COLOR,
        ),
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
        ("⁺∕₋", Box::new(|calc| calc.instant_op(|x| -x)), OP_COLOR),
        (
            "0",
            Box::new(|calc| {
                if !calc.digits.is_empty() || calc.decimal_mode {
                    calc.add_digit(0)
                }
            }),
            NUM_COLOR,
        ),
        (
            ".",
            Box::new(|calc| calc.decimal_mode = !calc.decimal_mode),
            OP_COLOR,
        ),
        ("=", Box::new(|calc| calc.apply_op()), EQ_COLOR),
    ]
});

impl FnPersist<CalcState, im::HashMap<Rc<SourceID>, Option<Window>>> for CalcApp {
    type Store = (CalcState, im::HashMap<Rc<SourceID>, Option<Window>>);

    fn init(&self) -> Self::Store {
        (Default::default(), im::HashMap::new())
    }
    fn call(
        &self,
        mut store: Self::Store,
        args: &CalcState,
    ) -> (Self::Store, im::HashMap<Rc<SourceID>, Option<Window>>) {
        if store.0 != *args {
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
                text: (if let Some(cur) = args.cur {
                    cur
                } else {
                    args.last
                })
                .to_string(),
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

//uniffi::include_scaffolding!("calculator");

fn main() {
    #[allow(clippy::type_complexity)]
    let mut inputs: Vec<
        Box<dyn FnMut((u64, Box<dyn Any>), CalcState) -> Result<CalcState, CalcState>>,
    > = Vec::new();

    for (_, f, _) in BUTTONS.iter() {
        inputs.push(Box::new(
            |_: mouse_area::MouseAreaEvent,
             mut appdata: CalcState|
             -> Result<CalcState, CalcState> {
                {
                    f(&mut appdata);
                    Ok(appdata)
                }
            }
            .wrap(),
        ));
    }

    let (mut app, event_loop): (App<CalcState, CalcApp>, winit::event_loop::EventLoop<()>) =
        App::new(
            CalcState {
                last: 0.0,
                cur: Some(0.0),
                digits: Vec::new(),
                decimals: Vec::new(),
                decimal_mode: false,
                op: CalcOp::None,
            },
            inputs,
            CalcApp {},
        )
        .unwrap();

    event_loop.run_app(&mut app).unwrap();
}
