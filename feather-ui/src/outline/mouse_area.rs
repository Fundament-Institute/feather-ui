use super::StateMachine;
use crate::input::{MouseState, RawEvent, RawEventKind};
use crate::layout::empty::Empty;
use crate::DispatchPair;
use crate::Slot;
use crate::{layout, Error};
use crate::{Dispatchable, SourceID};
use derive_where::derive_where;
use enum_variant_type::EnumVariantType;
use std::any::TypeId;
use std::collections::HashSet;
use std::rc::Rc;

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
                Box::new(mouse_area_event::OnClick::try_from(self).unwrap()),
            ),
            MouseAreaEvent::Hover => (
                MouseAreaEventKind::Hover as u64,
                Box::new(mouse_area_event::Hover::try_from(self).unwrap()),
            ),
            MouseAreaEvent::Active => (
                MouseAreaEventKind::Active as u64,
                Box::new(mouse_area_event::Active::try_from(self).unwrap()),
            ),
        }
    }

    fn restore(pair: DispatchPair) -> Result<Self, Error> {
        const KIND_ONCLICK: u64 = MouseAreaEventKind::OnClick as u64;
        const KIND_HOVER: u64 = MouseAreaEventKind::Hover as u64;
        const KIND_ACTIVE: u64 = MouseAreaEventKind::Active as u64;
        let typeid = pair.1.type_id();
        match pair.0 {
            KIND_ONCLICK => Ok(MouseAreaEvent::from(
                *pair
                    .1
                    .downcast::<mouse_area_event::OnClick>()
                    .map_err(|_| {
                        Error::MismatchedEnumTag(
                            pair.0,
                            TypeId::of::<mouse_area_event::OnClick>(),
                            typeid,
                        )
                    })?,
            )),
            KIND_HOVER => Ok(MouseAreaEvent::from(
                *pair.1.downcast::<mouse_area_event::Hover>().map_err(|_| {
                    Error::MismatchedEnumTag(
                        pair.0,
                        TypeId::of::<mouse_area_event::Hover>(),
                        typeid,
                    )
                })?,
            )),
            KIND_ACTIVE => Ok(MouseAreaEvent::from(
                *pair.1.downcast::<mouse_area_event::Active>().map_err(|_| {
                    Error::MismatchedEnumTag(
                        pair.0,
                        TypeId::of::<mouse_area_event::Active>(),
                        typeid,
                    )
                })?,
            )),
            _ => Err(Error::InvalidEnumTag(pair.0)),
        }
    }
}

#[derive(Default, Clone)]
struct MouseAreaState {
    lastdown: HashSet<winit::event::DeviceId>,
    lasttouch: HashSet<(winit::event::DeviceId, u64)>,
}

#[derive_where(Clone)]
pub struct MouseArea<Parent: Clone> {
    pub id: Rc<SourceID>,
    props: Parent,
    slots: [Option<Slot>; MouseAreaEvent::SIZE],
}

impl<Parent: Clone + 'static> MouseArea<Parent> {
    pub fn new(id: SourceID, props: Parent, slots: [Option<Slot>; MouseAreaEvent::SIZE]) -> Self {
        Self {
            id: id.into(),
            props,
            slots,
        }
    }
}
impl<Parent: Clone + 'static> super::Outline<Parent> for MouseArea<Parent> {
    fn id(&self) -> Rc<SourceID> {
        self.id.clone()
    }

    fn init_all(&self, _: &mut crate::StateManager) -> eyre::Result<()> {
        Ok(())
    }

    fn init(&self) -> Result<Box<dyn super::StateMachineWrapper>, crate::Error> {
        let onclick = Box::new(
            crate::wrap_event::<RawEvent, MouseAreaEvent, MouseAreaState>(|e, area, mut data| {
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
                                    return Ok((data, vec![]));
                                }
                            } // TODO: if this is inside the area, capture the mouse input.
                            MouseState::Up => {
                                if data.lastdown.contains(&device_id) {
                                    data.lastdown.remove(&device_id);
                                    if area.contains(pos) {
                                        return Ok((data, vec![MouseAreaEvent::OnClick]));
                                    }
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
                                return Ok((data, vec![]));
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
                Err((data, vec![]))
            }),
        );

        //<MouseAreaEvent, MouseAreaState, 1, 3>
        Ok(Box::new(StateMachine {
            state: Some(Default::default()),
            input: [(
                RawEventKind::Mouse as u64 | RawEventKind::Touch as u64,
                onclick,
            )],
            output: self.slots.clone(),
        }))
    }

    fn layout(
        &self,
        _: &crate::StateManager,
        _: &crate::DriverState,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn crate::layout::Layout<Parent>> {
        Box::new(layout::Node::<Empty, Parent> {
            props: (),
            imposed: self.props.clone(),
            children: Default::default(),
            id: Rc::downgrade(&self.id),
            renderable: None,
        })
    }
}
