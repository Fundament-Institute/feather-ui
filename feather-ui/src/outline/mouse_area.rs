use crate::input::{MouseState, RawEvent, RawEventKind};
use crate::layout;
use crate::layout::empty::Empty;
use crate::DispatchPair;
use crate::Slot;
use crate::{Dispatchable, SourceID};
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use std::collections::HashSet;
use std::rc::Rc;

use super::StateMachine;

#[derive(Debug, EnumVariantType, Clone)]
#[evt(derive(Clone), module = "mouse_area_event")]
pub enum MouseAreaEvent {
    OnClick,
    Hover,
    Active,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum MouseAreaEventKind {
    OnClick = 1,
    Hover = 2,
    Active = 4,
}

impl Dispatchable for MouseAreaEvent {
    const SIZE: usize = 3;

    fn extract(self) -> DispatchPair {
        match self {
            MouseAreaEvent::OnClick => (
                MouseAreaEventKind::OnClick as u64,
                Box::new(mouse_area_event::OnClick::try_from(self)),
            ),
            MouseAreaEvent::Hover => (
                MouseAreaEventKind::Hover as u64,
                Box::new(mouse_area_event::Hover::try_from(self)),
            ),
            MouseAreaEvent::Active => (
                MouseAreaEventKind::Active as u64,
                Box::new(mouse_area_event::Active::try_from(self)),
            ),
        }
    }

    fn restore(pair: DispatchPair) -> eyre::Result<Self> {
        const KIND_ONCLICK: u64 = MouseAreaEventKind::OnClick as u64;
        const KIND_HOVER: u64 = MouseAreaEventKind::Hover as u64;
        const KIND_ACTIVE: u64 = MouseAreaEventKind::Active as u64;
        match pair.0 {
            KIND_ONCLICK => Ok(MouseAreaEvent::from(
                *pair
                    .1
                    .downcast::<mouse_area_event::OnClick>()
                    .map_err(|_| eyre::eyre!("enum object didn't match tag!"))?,
            )),
            KIND_HOVER => Ok(MouseAreaEvent::from(
                *pair
                    .1
                    .downcast::<mouse_area_event::Hover>()
                    .map_err(|_| eyre::eyre!("enum object didn't match tag!"))?,
            )),
            KIND_ACTIVE => Ok(MouseAreaEvent::from(
                *pair
                    .1
                    .downcast::<mouse_area_event::Active>()
                    .map_err(|_| eyre::eyre!("enum object didn't match tag!"))?,
            )),
            _ => Err(eyre::eyre!("event type doesn't match integer!")),
        }
    }
}

struct MouseAreaState {
    lastdown: HashSet<winit::event::DeviceId>,
    lasttouch: HashSet<(winit::event::DeviceId, u64)>,
}

impl MouseAreaState {}

#[derive_where(Clone)]
pub struct MouseArea<AppData, Parent: Clone> {
    pub id: Rc<SourceID>,
    props: Parent,
    slots: [Option<Slot>; MouseAreaEvent::SIZE],
}

impl<AppData: 'static, Parent: Clone + 'static> MouseArea<AppData, Parent> {
    pub fn new(
        id: Rc<SourceID>,
        props: Parent,
        slots: [Option<Slot>; MouseAreaEvent::SIZE],
    ) -> Self {
        Self { id, props, slots }
    }
}
impl<AppData: 'static, Parent: Clone + 'static> super::Outline<AppData, Parent>
    for MouseArea<AppData, Parent>
{
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn layout(
        &self,
        state: &mut crate::StateManager,
        driver: &crate::DriverState,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn crate::layout::Layout<Parent, AppData>> {
        let state = state.init(self.id(), || -> eyre::Result<StateMachine<MouseAreaEvent, MouseAreaState, {RawEvent::SIZE}, {MouseAreaEvent::SIZE}>> {
            let onclick = Box::new(
                crate::wrap_event::<RawEvent, MouseAreaEvent, MouseAreaState>(|e, area, data| {
                    match e {
                        RawEvent::Mouse {
                            device_id,
                            state,
                            pos,
                            ..
                        } => {
                            match state {
                                MouseState::Down => {
                                    if area.contains(pos) {
                                        data.lastdown.insert(device_id);
                                    }
                                } // TODO: if this is inside the area, capture the mouse input.
                                MouseState::Up => {
                                    if data.lastdown.contains(&device_id) {
                                        if area.contains(pos) {
                                            return Ok((data, vec![MouseAreaEvent::OnClick]));
                                        }
                                        data.lastdown.remove(&device_id);
                                    }
                                }
                                MouseState::DblClick => (), // TODO: some OSes may send this *instead of* a mouse up, in which case this should be treated as one.
                            }
                        }
                        RawEvent::Touch {
                            device_id,
                            index,
                            state,
                            pos,
                            ..
                        } => match state {
                            crate::input::TouchState::Start => {
                                if area.contains(pos.xy()) {
                                    data.lasttouch.insert((device_id, index));
                                }
                            }
                            crate::input::TouchState::Move => (),
                            crate::input::TouchState::End => {
                                if data.lasttouch.contains(&(device_id, index)) {
                                    if area.contains(pos.xy()) {
                                        return Ok((data, vec![MouseAreaEvent::OnClick]));
                                    }
                                    data.lasttouch.remove(&(device_id, index));
                                }
                            }
                        },
                        _ => (),
                    }
                    Err(data)
                }

            ));
            
            Ok(StateMachine
            {
                state: todo!(),
                input: [(RawEventKind::Mouse as u64 | RawEventKind::Touch as u64, )],
                output: self.slots,
            })
        });

        Box::new(layout::Node::<AppData, Empty, Parent> {
            props: (),
            imposed: self.props.clone(),
            children: Default::default(),
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
