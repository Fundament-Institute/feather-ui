// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

// This represents are arbitrary user-defined component from a lua layout
#[derive(feather_macro::StateMachineChild)]
#[derive_where(Clone)]
pub struct LuaComponent<T> {
    pub id: Arc<SourceID>,
    pub child: Box<dyn ComponentWrap<dyn base::Empty>>,
}

impl<T: base::Empty + 'static> super::Component for LuaComponent<T>
where
    for<'a> &'a T: Into<&'a (dyn base::Empty + 'static)>,
{
    type Props = T;

    fn layout(
        &self,
        driver: &mut crate::StateManager,
        manager: &crate::graphics::Driver,
        window: &Arc<SourceID>,
    ) -> Box<dyn Layout<T>> {
        self.child.layout(manager, driver, window)
    }
}
