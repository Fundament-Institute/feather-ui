// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use feather_macro::*;
use feather_ui::gen_id;
use feather_ui::layout::simple;
use feather_ui::outline::button::Button;
use feather_ui::outline::mouse_area;
use feather_ui::outline::region::Region;
use feather_ui::outline::shape::Shape;
use feather_ui::outline::text::Text;
use feather_ui::outline::window::Window;
use feather_ui::outline::OutlineFrom;
use feather_ui::persist::FnPersist;
use feather_ui::App;
use feather_ui::Slot;
use feather_ui::SourceID;
use feather_ui::URect;
use feather_ui::WrapEventEx;
use feather_ui::FILL_URECT;
use std::any::Any;
use std::any::TypeId;
use std::f32;
use std::rc::Rc;
use std::sync::Arc;
use ultraviolet::Vec4;

#[cfg(target_os = "windows")]
use winit::platform::windows::WindowAttributesExtWindows;

uniffi::include_scaffolding!("calculator");

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

#[derive(Default, Empty, Area, Margin, Anchor, Limits, ZIndex)]
struct SimpleData {
    area: URect,
    margin: URect,
    anchor: feather_ui::UPoint,
    limits: URect,
    zindex: i32,
}

impl simple::Prop for SimpleData {}

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

    /// # Safety
    /// caller guarantees that T is the correct type
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

struct CalcApp {
    init_calc: Arc<dyn Calculator>,
}

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
        (CalcFFI(self.init_calc.clone()), im::HashMap::new())
    }
    fn call(
        &self,
        mut store: Self::Store,
        args: &CalcFFI,
    ) -> (Self::Store, im::HashMap<Rc<SourceID>, Option<Window>>) {
        //if store.0.eq(args) {
        let mut children: im::Vector<Option<Box<OutlineFrom<dyn simple::Prop>>>> =
            im::Vector::new();

        let button_id = Rc::new(gen_id!());

        for (i, (txt, _, color)) in BUTTONS.iter().enumerate() {
            let rect = Shape::round_rect(
                gen_id!(button_id).into(),
                feather_ui::FILL_URECT.into(),
                0.0,
                0.0,
                Vec4::broadcast(10.0),
                *color,
                Vec4::zero(),
            );

            let text = Text::<URect> {
                id: gen_id!(button_id).into(),
                props: feather_ui::FILL_URECT.into(),
                text: txt.to_string(),
                font_size: 30.0,
                line_height: 42.0,
                ..Default::default()
            };

            let mut btn_children: im::Vector<Option<Box<OutlineFrom<dyn simple::Prop>>>> =
                im::Vector::new();
            btn_children.push_back(Some(Box::new(text)));
            btn_children.push_back(Some(Box::new(rect)));

            const ROW_COUNT: usize = 4;
            let (x, y) = (i % ROW_COUNT, (i / ROW_COUNT) + 1);
            let (w, h) = (1.0 / ROW_COUNT as f32, 1.0 / 7.0);

            let btn = Button::<SimpleData>::new(
                gen_id!(button_id).into(),
                SimpleData {
                    area: URect {
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

        let display = Text::<URect> {
            id: gen_id!(button_id).into(),
            props: feather_ui::FILL_URECT.into(),
            text: args.0.get().to_string(),
            font_size: 60.0,
            line_height: 42.0,
            ..Default::default()
        };

        let text_bg = Shape::round_rect(
            gen_id!(button_id).into(),
            Rc::new(URect {
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
            }),
            0.0,
            0.0,
            Vec4::broadcast(25.0),
            Vec4::new(0.2, 0.2, 0.2, 1.0),
            Default::default(),
        );

        children.push_back(Some(Box::new(display)));
        children.push_back(Some(Box::new(text_bg)));

        let region = Region::<SimpleData> {
            id: gen_id!().into(),
            props: SimpleData {
                area: FILL_URECT,
                ..Default::default()
            }
            .into(),
            children,
        };

        #[cfg(target_os = "windows")]
        let window = Window::new(
            gen_id!().into(),
            winit::window::Window::default_attributes()
                .with_title("calculator-rs")
                .with_drag_and_drop(false)
                .with_resizable(true),
            Box::new(region),
        );
        #[cfg(not(target_os = "windows"))]
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
        //}
        let windows = store.1.clone();
        (store, windows)
    }
}

pub fn register(calc: ::std::sync::Arc<dyn Calculator>) {
    #[allow(clippy::type_complexity)]
    let mut inputs: Vec<
        Box<dyn FnMut((u64, Box<dyn Any>), CalcFFI) -> Result<CalcFFI, CalcFFI>>,
    > = Vec::new();

    for (_, f, _) in BUTTONS.iter() {
        inputs.push(Box::new(
            |_: mouse_area::MouseAreaEvent, appdata: CalcFFI| -> Result<CalcFFI, CalcFFI> {
                {
                    f(&appdata.0);
                    Ok(appdata)
                }
            }
            .wrap(),
        ));
    }

    let init_calc = calc.copy();
    let (mut app, event_loop): (App<CalcFFI, CalcApp>, winit::event_loop::EventLoop<()>) =
        App::new(CalcFFI(calc), inputs, CalcApp { init_calc }).unwrap();

    event_loop.run_app(&mut app).unwrap();
}
