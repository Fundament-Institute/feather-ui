use crate::input::{EventKind, MouseState};
use crate::layout::empty::Empty;
use crate::layout::EventList;
use crate::{layout, EventHandler};
use derive_where::derive_where;
use std::cell::RefCell;
use std::rc::Rc;

#[derive_where(Clone)]
pub struct MouseArea<AppData, Parent: Clone> {
    props: Parent,
    events: Rc<EventList<AppData>>,
}

impl<AppData: 'static, Parent: Clone + 'static> MouseArea<AppData, Parent> {
    pub fn new(props: Parent, mut onclick: EventHandler<AppData>) -> Self {
        let mut lastdown = false;
        let mut lasttouch = false;
        Self {
            props,
            events: Rc::new(vec![(
                EventKind::Mouse as u8 | EventKind::Touch as u8,
                // We cram all our important logic into our mouse event handler
                RefCell::new(Box::new(move |e, area, data| {
                    match e {
                        crate::input::Event::Mouse(mouse_state, pos, _, _, _) => {
                            match mouse_state {
                                MouseState::Down => lastdown = area.contains(pos), // TODO: if this is inside the area, capture the mouse input.
                                MouseState::Up => {
                                    if lastdown {
                                        if area.contains(pos) {
                                            return onclick(e, area, data);
                                        }
                                        lastdown = false;
                                    }
                                }
                                MouseState::DblClick => (), // TODO: some OSes may send this *instead of* a mouse up, in which case this should be treated as one.
                            }
                        }
                        crate::input::Event::Touch(touch_state, pos, _, _, _, _) => {
                            match touch_state {
                                crate::input::TouchState::Start => {
                                    lasttouch = area.contains(pos.xy())
                                }
                                crate::input::TouchState::Move => (),
                                crate::input::TouchState::End => {
                                    if lasttouch {
                                        if area.contains(pos.xy()) {
                                            return onclick(e, area, data);
                                        }
                                        lasttouch = false;
                                    }
                                }
                            }
                        }
                        _ => (),
                    }
                    Err(data)
                })),
            )]),
        }
    }
}
impl<AppData: 'static, Parent: Clone + 'static> super::Component<AppData, Parent>
    for MouseArea<AppData, Parent>
{
    fn layout(
        &self,
        _: &AppData,
        _: &crate::DriverState,
        _: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn crate::layout::Layout<Parent, AppData>> {
        Box::new(layout::Node::<AppData, Empty, Parent> {
            props: (),
            imposed: self.props.clone(),
            children: Default::default(),
            events: Rc::downgrade(&self.events),
            renderable: None,
        })
    }
}
