// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use smallvec::SmallVec;

use crate::{AccessCell, Dispatchable, InputResult, PxRect};

pub trait EventRouter
where
    // : zerocopy::Immutable
    Self: std::marker::Sized,
{
    type Input: Dispatchable + 'static;
    type Output: Dispatchable + 'static;

    #[allow(unused_variables)]
    #[allow(clippy::type_complexity)]
    fn process(
        state: AccessCell<Self>,
        input: Self::Input,
        area: PxRect,
        extent: PxRect,
        dpi: crate::RelDim,
        driver: &std::sync::Weak<crate::Driver>,
    ) -> InputResult<SmallVec<[Self::Output; 1]>> {
        InputResult::Forward(SmallVec::new())
    }
}
