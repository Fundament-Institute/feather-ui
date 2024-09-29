/*pub trait ClickHandler<AppData> {
    fn on_click(&self, data: AppData) -> AppData;
}*/
use crate::{EventHandler, URect};

pub struct MouseArea<AppData> {
    onclick: EventHandler<AppData>,
    area: URect,
}

impl<AppData: 'static, Parent> super::Component<AppData, Parent> for MouseArea<AppData> {
    fn layout(&self, data: AppData) -> Box<dyn crate::layout::Layout<Parent, AppData>> {
        todo!()
    }
}
