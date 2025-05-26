// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::{AbsRect, DriverState};

pub mod chain;
pub mod domain;
pub mod line;
pub mod standard;
pub mod text;

pub trait Renderable {
    fn render(&self, area: AbsRect, driver: &DriverState) -> im::Vector<super::RenderInstruction>;
}
