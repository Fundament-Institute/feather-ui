// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::graphics::point_to_pixel;
use crate::layout::{self, leaf};
use crate::reactive::{ConstSignal, DynSignal, MutableSignal, SignalZip, zip_pair};
use crate::{Limited, Resolve, Unsizable, reactive};
use derive_where::derive_where;
use std::rc::Rc;
use std::sync::Arc;

#[derive_where(Clone)]
#[derive(Debug)]
pub struct Text<T> {
    pub props: Rc<T>,
    pub font_size: DynSignal<f32>,
    pub line_height: DynSignal<f32>,
    pub text: DynSignal<String>,
    pub attributes: DynSignal<cosmic_text::AttrsOwned>,
    pub wrap: DynSignal<cosmic_text::Wrap>,
    pub align: DynSignal<Option<cosmic_text::Align>>, /* Alignment overrides whether text is LTR or RTL so
                                                       * we usually only want to set it if we're centering
                                                       * text */
}

impl<T: Default> Default for Text<T> {
    fn default() -> Self {
        Self {
            props: Default::default(),
            font_size: Default::default(),
            line_height: Default::default(),
            text: Default::default(),
            attributes: ConstSignal::new(cosmic_text::AttrsOwned::new(&cosmic_text::Attrs::new()))
                .into(),
            wrap: ConstSignal::new(cosmic_text::Wrap::None).into(),
            align: Default::default(),
        }
    }
}
impl<T: leaf::Padded + 'static> Text<T> {
    pub fn new(
        props: T,
        font_size: DynSignal<f32>,
        line_height: DynSignal<f32>,
        text: DynSignal<String>,
        attributes: DynSignal<cosmic_text::AttrsOwned>,
        wrap: DynSignal<cosmic_text::Wrap>,
        align: DynSignal<Option<cosmic_text::Align>>,
    ) -> Self {
        Self {
            props: props.into(),
            font_size,
            line_height,
            text,
            attributes,
            wrap,
            align,
        }
    }
}

impl<T: leaf::Padded + reactive::SignalDebug + 'static> super::Component for Text<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    type Props = T;
    type R = layout::text::Node<T, crate::render::text::PreInstance>;

    fn layout(
        &self,
        driver2: &Arc<crate::graphics::Driver>,
        dpi: MutableSignal<crate::RelDim>,
    ) -> Self::R {
        let inner_limits = reactive::defer::<crate::PxLimits, _>();
        let wdriver = Arc::downgrade(driver2);
        let wdriver2 = wdriver.clone();
        let wdriver3 = wdriver.clone();
        let wdriver4 = wdriver.clone();
        let wdriver5 = wdriver.clone();

        let text_buffer = MutableSignal::<cosmic_text::Buffer, _>::new_inputs(
            cosmic_text::Buffer::new(
                &mut driver2.font_system.write(),
                cosmic_text::Metrics::new(1.0, 1.0),
            ),
            (
                (
                    self.wrap.clone(),
                    move |b: &mut cosmic_text::Buffer, wrap: &cosmic_text::Wrap| {
                        if let Some(driver) = wdriver2.upgrade() {
                            b.set_wrap(&mut driver.font_system.write(), *wrap);
                        }
                    },
                ),
                (
                    (
                        dpi.clone(),
                        self.font_size.clone(),
                        self.line_height.clone(),
                    )
                        .zip()
                        .value(),
                    move |b: &mut cosmic_text::Buffer,
                          (dpi, fsize, line): &(crate::RelDim, f32, f32)| {
                        if let Some(driver) = wdriver3.upgrade() {
                            b.set_metrics(
                                &mut driver.font_system.write(),
                                cosmic_text::Metrics {
                                    font_size: point_to_pixel(*fsize, dpi.width),
                                    line_height: point_to_pixel(*line, dpi.height),
                                },
                            );
                        }
                    },
                ),
                (
                    (
                        self.text.clone(),
                        self.attributes.clone(),
                        self.align.clone(),
                    )
                        .zip()
                        .value(),
                    move |b: &mut cosmic_text::Buffer,
                          (text, attrs, align): &(
                        String,
                        cosmic_text::AttrsOwned,
                        Option<cosmic_text::Align>,
                    )| {
                        if let Some(driver) = wdriver4.upgrade() {
                            b.set_text(
                                &mut driver.font_system.write(),
                                text,
                                &attrs.as_attrs(),
                                cosmic_text::Shaping::Advanced,
                                *align,
                            );
                        }
                    },
                ),
                (
                    (
                        dpi.clone(),
                        self.props.area(),
                        self.props.padding(),
                        inner_limits.clone(),
                    )
                        .zip()
                        .value(),
                    move |buffer: &mut cosmic_text::Buffer,
                          (dpi, area, padding, limits): &(
                        crate::RelDim,
                        crate::DRect,
                        crate::UPerimeter,
                        crate::Limits<crate::Pixel>,
                    )| {
                        let padding = padding.resolve(*dpi);
                        if let Some(driver) = wdriver5.upgrade() {
                            let (limitx, limity) = {
                                let max = limits.max();
                                (
                                    max.width.is_finite().then_some(max.width),
                                    max.height.is_finite().then_some(max.height),
                                )
                            };

                            let inner = area.resolve(*dpi);
                            let mut font_system = driver.font_system.write();

                            let (unsized_x, unsized_y) = inner.is_unsized();
                            let dim: crate::PxDim =
                                unsafe { inner.abs.dim_unchecked().cast_unit() }.limit(*limits)
                                    - padding.total();

                            buffer.set_size(
                                &mut font_system,
                                if unsized_x {
                                    limitx
                                } else {
                                    Some(dim.width.max(0.0))
                                },
                                if unsized_y {
                                    limity
                                } else {
                                    Some(dim.height.max(0.0))
                                },
                            );
                        }
                    },
                ),
            ),
        );

        let final_buffer = reactive::defer::<cosmic_text::Buffer, _>();

        let render = crate::render::text::PreInstance {
            text_buffer: final_buffer.clone(),
            padding: zip_pair(self.props.padding(), dpi, |x, dpi| x.resolve(*dpi)),
            driver: Arc::downgrade(driver2),
        };

        layout::text::Node {
            props: self.props.clone(),
            buffer: text_buffer.into(),
            renderable: render.into(),
            //realign: self.align.is_some_and(|x| x != cosmic_text::Align::Left),
            driver: Arc::downgrade(driver2),
            machine: None,
            inner_limits,
            final_buffer,
        }
    }
}
