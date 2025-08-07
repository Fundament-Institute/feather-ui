// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Layout, leaf};
use crate::{DAbsPoint, SourceID, UNSIZED_AXIS};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct Image<T> {
    pub id: Arc<SourceID>,
    pub props: Rc<T>,
    pub resource: Box<dyn crate::resource::Location>,
    pub size: DAbsPoint,
    pub dynamic: bool,
}

impl<T: leaf::Padded + 'static> Image<T> {
    pub fn new(
        id: Arc<SourceID>,
        props: Rc<T>,
        resource: &dyn crate::resource::Location,
        size: DAbsPoint,
        dynamic: bool,
    ) -> Self {
        Self {
            id,
            props,
            resource: dyn_clone::clone_box(resource),
            size,
            dynamic,
        }
    }
}

fn zero_float(f: f32) -> i32 {
    if f.is_finite() && f != UNSIZED_AXIS {
        f.ceil() as i32
    } else {
        0
    }
}

impl<T: leaf::Padded + 'static> super::Component for Image<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    type Props = T;

    fn layout(
        &self,
        manager: &mut crate::StateManager,
        driver: &crate::graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        let winstate: &super::window::WindowStateMachine = manager.get(window).unwrap();
        let dpi = winstate
            .state
            .as_ref()
            .map(|x| x.dpi)
            .unwrap_or(crate::BASE_DPI);

        let size = self.size.resolve(dpi);

        // TODO: Layout cannot easily return an error because this messes up the persistent functions
        let uvsize = driver
            .load_and_resize(
                self.resource.as_ref(),
                guillotiere::Size::new(zero_float(size.x), zero_float(size.y)),
                dpi.x,
                self.dynamic,
            )
            .unwrap();

        Box::new(leaf::Sized::<T> {
            props: self.props.clone(),
            id: Arc::downgrade(&self.id),
            size: ultraviolet::Vec2::new(uvsize.width as f32, uvsize.height as f32),
            renderable: Some(Rc::new(crate::render::image::Instance {
                image: self.resource.clone(),
                padding: self.props.padding().resolve(dpi),
                dpi: dpi.x,
                resize: self.dynamic,
            })),
        })
    }
}
