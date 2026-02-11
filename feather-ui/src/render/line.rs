// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::{
    PxPoint,
    color::sRGB,
    reactive::{DynSignal, SignalMap, sample},
};

use super::compositor::CompositorView;
use super::compositor::Data;

#[derive(Clone)]
pub struct Instance {
    values: DynSignal<Data>,
}

impl Instance {
    pub fn new(start: DynSignal<PxPoint>, end: DynSignal<PxPoint>, color: DynSignal<sRGB>) -> Self {
        use crate::reactive::SignalZip;

        Self {
            values: (start, end, color)
                .zip()
                .flatmap(|(p1, p2, color)| {
                    let p = *p2 - *p1;
                    Data::new(
                        ((*p1 + p2.to_vector()) * 0.5)
                            - (crate::PxVector::new(p.length() * 0.5, 0.0)),
                        [p.length(), 1.0].into(),
                        [0.0, 0.0].into(),
                        [0.0, 0.0].into(),
                        color.as_32bit().rgba,
                        p.y.atan2(p.x) % std::f32::consts::TAU,
                        u8::MAX,
                        false,
                        false,
                    )
                })
                .into(),
        }
    }
}

impl super::Prerender for Instance {
    type R = Instance;

    fn prerender(&self, _: crate::DynSignal<crate::PxRect>) -> Self::R {
        self.clone()
    }
}

impl super::Renderable for Instance {
    fn render(
        &self,
        _: crate::PxPoint,
        _: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), crate::RenderError> {
        compositor.append_data(*sample(&self.values));
        Ok(())
    }
}
