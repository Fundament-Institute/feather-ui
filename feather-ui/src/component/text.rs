// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use crate::color::sRGB;
use crate::graphics::point_to_pixel;
use crate::layout::{self, Layout, leaf};
use crate::reactive::SignalTupleZip;
use crate::reactive::{DynSignal, MutableSignal, SignalMap, zip_pair};
use crate::{graphics, reactive};
use cosmic_text::{LineIter, Metrics};
use derive_where::derive_where;
use std::cell::RefCell;
use std::convert::Infallible;
use std::rc::Rc;
use std::sync::Arc;

#[derive_where(Clone)]
pub struct Text<T> {
    pub props: Rc<T>,
    pub font_size: DynSignal<f32>,
    pub line_height: DynSignal<f32>,
    pub text: DynSignal<String>,
    pub font: DynSignal<cosmic_text::FamilyOwned>,
    pub color: DynSignal<sRGB>,
    pub weight: DynSignal<cosmic_text::Weight>,
    pub style: DynSignal<cosmic_text::Style>,
    pub wrap: DynSignal<cosmic_text::Wrap>,
    pub align: DynSignal<Option<cosmic_text::Align>>, /* Alignment overrides whether text is LTR or RTL so
                                                       * we usually only want to set it if we're centering
                                                       * text */
}

impl<T: leaf::Padded + 'static> Text<T> {
    pub fn new(
        props: T,
        font_size: DynSignal<f32>,
        line_height: DynSignal<f32>,
        text: DynSignal<String>,
        font: DynSignal<cosmic_text::FamilyOwned>,
        color: DynSignal<sRGB>,
        weight: DynSignal<cosmic_text::Weight>,
        style: DynSignal<cosmic_text::Style>,
        wrap: DynSignal<cosmic_text::Wrap>,
        align: DynSignal<Option<cosmic_text::Align>>,
    ) -> Self {
        Self {
            props: props.into(),
            font_size,
            line_height,
            text,
            font,
            color,
            weight,
            style,
            wrap,
            align,
            buffer: Rc::new(RefCell::new(cosmic_text::Buffer::new_empty(Metrics::new(
                1.0, 1.0,
            )))),
        }
    }
}

fn buffer_eq(s: &str, b: &cosmic_text::Buffer) -> bool {
    let mut ranges = LineIter::new(s);
    let mut lines = b.lines.iter();
    loop {
        match (lines.next(), ranges.next()) {
            (Some(line), Some((r, _))) => {
                if &s[r] != line.text() {
                    return false;
                }
            }
            (None, None) => return true,
            _ => return false,
        }
    }
}

impl<T: leaf::Padded + 'static> super::Component for Text<T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    type Props = T;
    type R = layout::text::Node<T>;

    fn layout(
        &self,
        driver2: Arc<crate::graphics::Driver>,
        dpi: MutableSignal<crate::RelDim>,
    ) -> Self::R {
        let inner_dim = reactive::defer::<crate::UnsizedDim, _>();
        let wdriver = Arc::downgrade(&driver2);

        (
            self.text.clone(),
            self.font_size.clone(),
            self.line_height.clone(),
            dpi.clone(),
            self.wrap.clone(),
            self.font.clone(),
            self.color.clone(),
            self.weight.clone(),
            self.style.clone(),
            self.align.clone(),
            inner_dim.clone(),
            self.props.padding(),
            self.props.limits(),
        )
            .zip()
            .map_mut(
                |(
                    text,
                    font_size,
                    line_height,
                    dpi,
                    wrap,
                    family,
                    color,
                    weight,
                    style,
                    align,
                    inner,
                    padding,
                    limits,
                ): &(
                    String,
                    f32,
                    f32,
                    crate::RelDim,
                    cosmic_text::Wrap,
                    cosmic_text::FamilyOwned,
                    crate::color::sRGB,
                    cosmic_text::Weight,
                    cosmic_text::Style,
                    Option<cosmic_text::Align>,
                    crate::UnsizedDim,
                    crate::DAbsRect,
                    crate::DLimits,
                ),
                 buffer: Option<cosmic_text::Buffer>|
                 -> cosmic_text::Buffer {
                    let driver = wdriver.upgrade().unwrap();
                    let mut font_system = driver.font_system.write();

                    let metrics = cosmic_text::Metrics::new(
                        point_to_pixel(*font_size, dpi.width),
                        point_to_pixel(*line_height, dpi.height),
                    );

                    let buffer = buffer
                        .unwrap_or_else(|| cosmic_text::Buffer::new(&mut font_system, metrics));
                    buffer.set_metrics(&mut font_system, metrics);
                    buffer.set_wrap(&mut font_system, *wrap);

                    let attrs = cosmic_text::Attrs::new()
                        .family(family.as_family())
                        .color((*color).into())
                        .weight(*weight)
                        .style(*style);
                    if *align != buffer.lines[0].align()
                        || buffer.lines[0].attrs_list().get_span(0) != attrs
                        || !buffer_eq(&text, &buffer)
                    {
                        buffer.set_text(
                            &mut font_system,
                            &self.text,
                            &attrs,
                            cosmic_text::Shaping::Advanced,
                            self.align,
                        );
                    }

                    let mut font_system = driver.font_system.write();

                    let (limitx, limity) = {
                        let max = limits.max();
                        (
                            max.width.is_finite().then_some(max.width),
                            max.height.is_finite().then_some(max.height),
                        )
                    };

                    let dim = inner - padding.total();
                    let (unsized_x, unsized_y) = check_unsized_dim(dim);

                    text_buffer.set_size(
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

                    // If we have indeterminate area, calculate the size
                    if unsized_x || unsized_y {
                        let mut h = 0.0;
                        let mut w: f32 = 0.0;
                        //let mut realign = self.realign;

                        // TODO: In order to extract the width and height back out of the text buffer, we have to unconditionally
                        // set the width/height here. If we replace buffer with a wrapper that can hold a realign and w/h values,
                        // then we can restore this optimization.
                        let realign = true;

                        for run in text_buffer.layout_runs() {
                            w = w.max(run.line_w);
                            // If a line is RTL and we're unsized, we ALWAYS have to re-evaluate it!
                            realign = realign || run.rtl;
                            h += run.line_height;
                        }

                        // Apply adjusted limits to inner size calculation
                        w = w.max(limits.min().width).min(limits.max().width);
                        h = h.max(limits.min().height).min(limits.max().height);

                        // If we are centered or right aligned, we have to set the size again now that
                        // we know how big it really is. This is true even if all the text
                        // was originally marked as RTL - the layout will still be wrong because
                        // it didn't know how big the text would be.
                        if realign {
                            text_buffer.set_size(&mut driver.font_system.write(), Some(w), Some(h))
                        }

                        // Set w and h
                        self.w = w + padding.total().width;
                        self.h = h + padding.total().height;
                    };

                    buffer
                },
            );

        let render = Rc::new(crate::render::text::Instance::new(
            textstate.clone(),
            zip_pair(self.props.padding(), dpi, |x, dpi| x.as_perimeter(*dpi)),
            area,
            driver,
        ));

        layout::text::Node::<T> {
            props: self.props.clone(),
            buffer: textstate.buffer.clone(),
            renderable: render,
            //realign: self.align.is_some_and(|x| x != cosmic_text::Align::Left),
            driver: std::sync::Arc::downgrade(&driver),
            machine: None,
        }
    }
}
