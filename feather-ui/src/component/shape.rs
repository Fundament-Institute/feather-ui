// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use crate::layout::{Layout, leaf};
use crate::shaders::gen_uniform;
use crate::{BASE_DPI, DriverState, SourceID, WindowStateMachine, layout};
use derive_where::derive_where;
use std::borrow::Cow;
use std::rc::Rc;
use ultraviolet::Vec4;
use wgpu::util::DeviceExt;

#[derive_where(Clone)]
pub struct Shape<'a, T: leaf::Padded + 'static> {
    pub id: std::rc::Rc<SourceID>,
    pub props: Rc<T>,
    pub uniforms: [Vec4; 4],
    pub fragment: Cow<'a, str>,
    pub label: &'static str,
}

impl<T: leaf::Padded + 'static> Shape<'_, T> {
    pub fn round_rect(
        id: std::rc::Rc<SourceID>,
        props: Rc<T>,
        border: f32,
        blur: f32,
        corners: Vec4,
        fill: Vec4,
        outline: Vec4,
    ) -> Self {
        Self {
            id,
            props,
            uniforms: [Vec4::new(0.0, 0.0, border, blur), corners, fill, outline],
            fragment: Cow::Borrowed(include_str!("../shaders/RoundRect.wgsl")),
            label: "RoundRect FS",
        }
    }

    pub fn arc(
        id: std::rc::Rc<SourceID>,
        props: Rc<T>,
        border: f32,
        blur: f32,
        arcs: Vec4,
        fill: Vec4,
        outline: Vec4,
    ) -> Self {
        Self {
            id,
            props,
            uniforms: [Vec4::new(0.0, 0.0, border, blur), arcs, fill, outline],
            fragment: Cow::Borrowed(include_str!("../shaders/Arc.wgsl")),
            label: "Arc FS",
        }
    }

    pub fn circle(
        id: std::rc::Rc<SourceID>,
        props: Rc<T>,
        border: f32,
        blur: f32,
        radii: crate::Vec2,
        fill: Vec4,
        outline: Vec4,
    ) -> Self {
        Self {
            id,
            props,
            uniforms: [
                Vec4::new(0.0, 0.0, border, blur),
                Vec4::new(radii.x, radii.y, 0.0, 0.0),
                fill,
                outline,
            ],
            fragment: Cow::Borrowed(include_str!("../shaders/Circle.wgsl")),
            label: "Circle FS",
        }
    }
}

impl<T: leaf::Padded + 'static> crate::StateMachineChild for Shape<'_, T> {
    fn id(&self) -> std::rc::Rc<SourceID> {
        self.id.clone()
    }
}

impl<T: leaf::Padded + 'static> super::Component<T> for Shape<'_, T>
where
    for<'a> &'a T: Into<&'a (dyn leaf::Padded + 'static)>,
{
    fn layout(
        &self,
        state: &crate::StateManager,
        driver: &DriverState,
        window: &Rc<SourceID>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Box<dyn Layout<T>> {
        let winstate: &WindowStateMachine = state.get(window).unwrap();
        let dpi = winstate.state.as_ref().map(|x| x.dpi).unwrap_or(BASE_DPI);

        let shader_idx =
            driver
                .shader_cache
                .write()
                .register_shader(&driver.device, self.label, &self.fragment);
        let pipeline =
            driver
                .shader_cache
                .write()
                .standard_pipeline(&driver.device, shader_idx, config);

        let mvp = gen_uniform(
            driver,
            "MVP",
            crate::shaders::mat4_proj(
                0.0,
                config.height as f32,
                config.width as f32,
                -(config.height as f32),
                0.2,
                10000.0,
            )
            .as_byte_slice(),
        );

        let posdim = driver
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PosDim"),
                contents: Vec4::new(0.0, 0.0, 400.0, 200.0).as_byte_slice(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let dimborderblur = gen_uniform(
            driver,
            "DimBorderBlur",
            (self.uniforms[0] * Vec4::broadcast(dpi.x)).as_byte_slice(),
        );
        let corners = gen_uniform(
            driver,
            "Corners",
            (self.uniforms[1] * Vec4::broadcast(dpi.x)).as_byte_slice(),
        );
        let fill = gen_uniform(driver, "Fill", self.uniforms[2].as_byte_slice());
        let outline = gen_uniform(driver, "Component", self.uniforms[3].as_byte_slice());
        let buffers = [mvp, posdim, dimborderblur, corners, fill, outline];
        let bindings: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, x)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: x.as_entire_binding(),
            })
            .collect();

        let bind_group = driver.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bindings,
            label: None,
        });

        Box::new(layout::Node::<T, dyn leaf::Prop> {
            props: self.props.clone(),
            children: Default::default(),
            id: Rc::downgrade(&self.id),
            renderable: Some(Rc::new_cyclic(|this| crate::render::standard::Pipeline {
                this: this.clone(),
                pipeline,
                group: bind_group,
                vertices: crate::shaders::default_vertex_buffer(driver),
                padding: self.props.padding().resolve(dpi),
                buffers,
            })),
        })
    }
}

crate::gen_component_wrap!('a, Shape, leaf::Padded);
