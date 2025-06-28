// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::{cell::RefCell, rc::Rc};

use cosmic_text::Cursor;
use unicode_segmentation::UnicodeSegmentation;

use crate::{
    Error,
    color::sRGB,
    render::{compositor, text},
};

pub struct Instance {
    pub text_buffer: Rc<RefCell<cosmic_text::Buffer>>,
    pub padding: crate::AbsRect,
    pub cursor: Cursor,
    pub selection: Option<(Cursor, Cursor)>,
    pub selection_bg: sRGB,
    pub selection_color: sRGB,
    pub color: sRGB,
    pub cursor_color: sRGB,
    pub scale: f32,
}

impl Instance {
    fn draw_box(
        x: f32,
        y: f32,
        mut w: f32,
        mut h: f32,
        maxwidth: f32,
        maxheight: f32,
        color: sRGB,
    ) -> compositor::Data {
        // When we are drawing boxes that need to line up with each other, this is a worst-case scenario for
        // the compositor's antialiasing. The only way to antialias arbitrary selection boxes correctly is
        // to use a texture cache or a custom shader. Instead of doing that, we just pixel-snap everything.
        w = w.min((maxwidth - x).max(0.0));
        h = h.min((maxheight - y).max(0.0));
        compositor::Data {
            pos: [x.round(), y.round()].into(),
            dim: [w.round(), h.round()].into(),
            uv: [0.0, 0.0].into(),
            uvdim: [0.0, 0.0].into(),
            color: color.as_32bit().rgba,
            texclip: 0x8000000 | 0x7FFF0000,
            ..Default::default()
        }
    }
}

impl crate::render::Renderable for Instance {
    fn render(
        &self,
        area: crate::AbsRect,
        driver: &crate::graphics::Driver,
        compositor: &mut compositor::Compositor,
    ) -> Result<(), Error> {
        let buffer = self.text_buffer.borrow();
        // Padding works differently in a textbox than in a static text field, because a textbox
        // cannot having non-clipping regions outside the text area, or you'll get rendering errors
        // when scrolling.
        let area = crate::AbsRect(area.0 + (self.padding.0 * crate::MINUS_BOTTOMRIGHT));
        let pos = area.topleft();

        let bounds_top = area.topleft().y as i32;
        let bounds_bottom = area.bottomright().y as i32;
        let bounds_min_x = (area.topleft().x as i32).max(0);
        let bounds_min_y = bounds_top.max(0);
        let bounds_max_x = area.bottomright().x as i32;
        let bounds_max_y = bounds_bottom;
        let color = cosmic_text::Color(self.color.as_32bit().rgba);
        let selection_color = cosmic_text::Color(self.selection_color.as_32bit().rgba);

        let is_run_visible = |run: &cosmic_text::LayoutRun| {
            let start_y_physical = (pos.y + (run.line_top * self.scale)) as i32;
            let end_y_physical = start_y_physical + (run.line_height * self.scale) as i32;

            start_y_physical <= bounds_bottom && bounds_top <= end_y_physical
        };

        for run in crate::editor::FixedRunIter::new(&buffer)
            .skip_while(|run| !is_run_visible(run))
            .take_while(is_run_visible)
        {
            let line_i = run.line_i;
            let line_top = run.line_top;
            let line_height = run.line_height;

            // Highlight selection
            if let Some((start, end)) = &self.selection {
                if line_i >= start.line && line_i <= end.line {
                    let mut range_opt = None;
                    for glyph in run.glyphs.iter() {
                        // Guess x offset based on characters
                        let cluster = &run.text[glyph.start..glyph.end];
                        let total = cluster.grapheme_indices(true).count();
                        let mut c_x = glyph.x;
                        let c_w = glyph.w / total as f32;
                        for (i, c) in cluster.grapheme_indices(true) {
                            let c_start = glyph.start + i;
                            let c_end = glyph.start + i + c.len();
                            if (start.line != line_i || c_end > start.index)
                                && (end.line != line_i || c_start < end.index)
                            {
                                range_opt = match range_opt.take() {
                                    Some((min, max)) => Some((
                                        std::cmp::min(min, c_x as i32),
                                        std::cmp::max(max, (c_x + c_w) as i32),
                                    )),
                                    None => Some((c_x as i32, (c_x + c_w) as i32)),
                                };
                            } else if let Some((min, max)) = range_opt.take() {
                                compositor.append(&Self::draw_box(
                                    min as f32 + pos.x,
                                    line_top + pos.y,
                                    std::cmp::max(0, max - min) as f32,
                                    line_height,
                                    bounds_max_x as f32,
                                    bounds_max_y as f32,
                                    self.selection_bg,
                                ));
                            }
                            c_x += c_w;
                        }
                    }

                    if run.glyphs.is_empty() && end.line > line_i {
                        // Highlight all of internal empty lines
                        range_opt = Some((0, buffer.size().0.unwrap_or(0.0) as i32));
                    }

                    if let Some((mut min, mut max)) = range_opt.take() {
                        if end.line > line_i {
                            // Draw to end of line
                            if run.rtl {
                                min = 0;
                            } else if let (Some(w), _) = buffer.size() {
                                max = w.round() as i32;
                            } else if max == 0 {
                                max = (buffer.metrics().font_size * 0.5) as i32;
                            }
                        }
                        compositor.append(&Self::draw_box(
                            min as f32 + pos.x,
                            line_top + pos.y,
                            std::cmp::max(0, max - min) as f32,
                            line_height,
                            bounds_max_x as f32,
                            bounds_max_y as f32,
                            self.selection_bg,
                        ));
                    }
                }
            }

            // Draw text
            for glyph in run.glyphs.iter() {
                let physical_glyph = glyph.physical((pos.x, pos.y), self.scale);

                let mut color = match glyph.color_opt {
                    Some(some) => some,
                    None => color,
                };

                if let Some((start, end)) = self.selection {
                    if line_i >= start.line
                        && line_i <= end.line
                        && (start.line != line_i || glyph.end > start.index)
                        && (end.line != line_i || glyph.start < end.index)
                    {
                        color = selection_color;
                    }
                }

                text::Instance::write_glyph(
                    physical_glyph.cache_key,
                    &mut driver.font_system.write(),
                    &mut driver.glyphs.write(),
                    &driver.device,
                    &driver.queue,
                    &mut driver.atlas.write(),
                    &mut driver.swash_cache.write(),
                )?;

                if let Some(data) = text::Instance::prepare_glyph(
                    physical_glyph.x,
                    physical_glyph.y,
                    run.line_y,
                    self.scale,
                    color,
                    bounds_min_x,
                    bounds_min_y,
                    bounds_max_x,
                    bounds_max_y,
                    text::Instance::get_glyph(physical_glyph.cache_key, &driver.glyphs.read())
                        .ok_or(Error::GlyphCacheFailure)?,
                )? {
                    compositor.append(&data);
                }
            }

            // Draw cursor
            if let Some((x, y)) = crate::editor::cursor_position(&self.cursor, &run) {
                compositor.append(&Self::draw_box(
                    x as f32 + pos.x,
                    y as f32 + pos.y,
                    1.0,
                    line_height,
                    bounds_max_x as f32,
                    bounds_max_y as f32,
                    self.cursor_color,
                ));
            }
        }

        Ok(())
    }
}
