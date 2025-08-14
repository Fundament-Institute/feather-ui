// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::compositor::CompositorView;
use crate::Evaluated;
use crate::resource::Location;

pub struct Instance {
    pub image: Box<dyn Location>,
    pub padding: crate::Perimeter<Evaluated>,
    pub dpi: f32,
    pub resize: bool,
}

impl super::Renderable for Instance {
    fn render(
        &self,
        area: crate::AnyRect,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), crate::Error> {
        let dim = area.dim() - self.padding.to_untyped().bottomright();
        if dim.width <= 0.0 || dim.height <= 0.0 {
            return Ok(());
        }

        driver.load(
            self.image.as_ref(),
            dim.ceil().to_i32(),
            self.dpi,
            self.resize,
            |region| {
                compositor.append_data(
                    area.topleft() + self.padding.to_untyped().topleft().to_vector(),
                    dim,
                    region.uv.min.to_f32(),
                    region.uv.size().to_f32(),
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
