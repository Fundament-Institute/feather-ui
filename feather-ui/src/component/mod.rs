use crate::layout::Layout;

//pub mod button;
pub mod mouse_area;
//pub mod round_rect;
pub mod text;

pub trait Component<AppData, Parent> {
    fn layout(&self, data: AppData) -> Box<dyn Layout<Parent, AppData>>;
}
