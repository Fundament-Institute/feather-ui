// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::compositor::CompositorView;
use crate::AbsRect;
use crate::resource::Location;

pub struct Instance {
    pub image: Box<dyn Location>,
    pub padding: AbsRect,
    pub dpi: f32,
    pub resize: bool,
}

impl super::Renderable for Instance {
    fn render(
        &self,
        area: crate::AbsRect,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), crate::Error> {
        let dim = area.bottomright() - area.topleft() - self.padding.bottomright();
        if dim.x <= 0.0 || dim.y <= 0.0 {
            return Ok(());
        }

        driver.load(
            self.image.as_ref(),
            guillotiere::Size::new(dim.x.ceil() as i32, dim.y.ceil() as i32),
            self.dpi,
            self.resize,
            |region| {
                compositor.append_data(
                    area.topleft() + self.padding.topleft(),
                    dim,
                    region.uv.min.to_f32().to_array().into(),
                    region.uv.size().to_f32().to_array().into(),
                    0xFFFFFFFF,
                    0.0,
                    region.index,
                    false,
                );
                Ok(())
            },
        )
    }
}
