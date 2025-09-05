// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::compositor;
use crate::color::sRGB;
use crate::component::shape::ShapeKind;
use crate::graphics::{self, Vec2f, Vec4f};
use crate::render::atlas::Atlas;
use crate::render::compositor::CompositorView;
use crate::{PxDim, PxPoint, shaders};
use core::f32;
use guillotiere::euclid::Size2D;
use num_traits::Zero;
use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::num::NonZero;
use wgpu::BindGroupLayout;

pub struct Instance<PIPELINE> {
    pub padding: crate::PxPerimeter,
    pub border: f32,
    pub blur: f32,
    pub fill: sRGB,
    pub outline: sRGB,
    pub corners: [f32; 4],
    pub id: std::sync::Arc<crate::SourceID>,
    pub phantom: PhantomData<PIPELINE>,
}

impl<PIPELINE: crate::render::Pipeline<Data = Data> + 'static> super::Renderable
    for Instance<PIPELINE>
{
    fn render(
        &self,
        area: crate::PxRect,
        driver: &crate::graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), crate::Error> {
        let dim = area.dim() - self.padding.bottomright() - self.padding.topleft();

        if dim.width <= 0.0 || dim.height <= 0.0 {
            return Ok(());
        }

        // If a rect has no corners and no outline, it's just a flat color box and we can draw it directly in the compositor
        if self.corners.iter().all(|x| x.is_zero()) && self.border.is_zero() {
            compositor.append_data(
                area.topleft().add_size(&self.padding.topleft()),
                dim,
                [0.0, 0.0].into(),
                [0.0, 0.0].into(),
                self.fill.as_32bit().rgba,
                0.0,
                u8::MAX,
                false,
            );

            return Ok(());
        }

        let perimeter = [
            dim.height - self.corners[0] - self.corners[3],
            dim.width - self.corners[0] - self.corners[1],
            dim.height - self.corners[1] - self.corners[2],
            dim.width - self.corners[2] - self.corners[3],
        ];

        // RoundRects have a specific optimization, but only if no edge length is less than 2 pixels
        if TypeId::of::<PIPELINE>() == TypeId::of::<Shape<{ ShapeKind::RoundRect as u8 }>>()
            && perimeter.iter().all(|x| *x >= 2.0)
        {
            // If the border is larger than the corner itself, pretend the size of that corner is the border.
            let mut corners = self.corners.map(|x| x.max(self.border));
            let mut intcorners = corners.map(|x| x.ceil() as i32);

            let intsides = [
                intcorners[0].max(intcorners[3]),
                intcorners[0].max(intcorners[1]),
                intcorners[1].max(intcorners[2]),
                intcorners[2].max(intcorners[3]),
            ];

            // Here we generate a rounded block equal to exactly left side + 3 pixels + right side
            let inner = Size2D::<i32, crate::Pixel>::new(
                intsides[0] + intsides[2] + 3, // left + right
                intsides[1] + intsides[3] + 3, // top + bottom
            );

            // TODO: cache this if the inputs are identical
            // We reserve an additional 2 pixel border around each side of our rect for sampling purposes. It
            // must be 2 pixels because we have to inflate the rect by 1 pixel for fractional draws already,
            // which means we need an additional transparent pixel of buffer to cover all possible sampling
            // scenarios.
            let (region_uv, region_index) = {
                let mut atlas = driver.atlas.write();
                let region = atlas.cache_region(
                    &driver.device,
                    &self.id,
                    inner + Size2D::<i32, crate::Pixel>::new(4, 4),
                    None,
                    Some(&driver.queue),
                )?;
                (region.uv, region.index)
            };

            // We render the rect here with corners raised to the nearest pixel so we can cut out the corners neatly
            driver.with_pipeline::<PIPELINE>(|pipeline| {
                pipeline.append(
                    &Data {
                        pos: region_uv
                            .min
                            .add_size(&Size2D::splat(2))
                            .to_f32()
                            .to_array()
                            .into(),
                        dim: inner.to_f32().to_array().into(),
                        border: self.border,
                        blur: self.blur,
                        corners: intcorners.map(|x| x as f32).into(),
                        fill: self.fill.as_32bit().rgba,
                        outline: self.outline.as_32bit().rgba,
                    },
                    region_index,
                )
            });

            // The only reason this works is because we set the uvdim here to 0 on the axis that is being
            // extended, which ensures no interpolation of the UV coordinate happens along that axis

            // We add data here starting from the topleft corner and going clockwise around the rect:
            // 1 6 2
            // 5 9 7
            // 4 8 3

            // Pretend all corners are 1 pixel larger (this works because our buffer is 3 pixels)
            corners = corners.map(|x| x + 1.0);
            intcorners = intcorners.map(|x| x + 1);

            let topleft = area.topleft().add_size(&self.padding.topleft()).to_vector();
            // Add 2 to account for the 2 pixel transparent border
            let uvpos = region_uv.min.add_size(&Size2D::splat(2));
            let mut gen_corner = |pos: PxPoint, corner: f32, u: i32, v: i32| {
                let intdim = PxDim::splat(corner.ceil());
                compositor.append_data(
                    pos + topleft,
                    PxDim::splat(corner),
                    uvpos
                        .add_size(&Size2D::new(u, v))
                        .to_f32()
                        .to_array()
                        .into(),
                    intdim.to_array().into(),
                    0xFFFFFFFF,
                    0.0,
                    region_index,
                    true,
                );
            };

            // This is nontrivial, because this must be assembled in raw mode, which means we must do the directional inflation
            // here ourselves. This amounts to changing every 0 into a -1, but *not* changing the non-zero positions and instead
            // adding 1 to the dimensions, which on corners means adding yet another +1 to the corner size.
            gen_corner(PxPoint::new(-1.0, -1.0), corners[0] + 1.0, -1, -1);
            gen_corner(
                PxPoint::new(dim.width - corners[1], -1.0),
                corners[1] + 1.0,
                inner.width - intcorners[1],
                -1,
            );
            gen_corner(
                PxPoint::new(dim.width - corners[2], dim.height - corners[2]),
                corners[2] + 1.0,
                inner.width - intcorners[2],
                inner.height - intcorners[2],
            );
            gen_corner(
                PxPoint::new(-1.0, dim.height - corners[3]),
                corners[3] + 1.0,
                -1,
                inner.height - intcorners[3],
            );

            let sides = crate::PxRect::new(
                corners[0].max(corners[3]),
                corners[0].max(corners[1]),
                corners[1].max(corners[2]),
                corners[2].max(corners[3]),
            );

            // We can't just do sides.ceil() because the result is not the same as ceiling both corners and adding them.
            let intsides = [
                intcorners[0].max(intcorners[3]),
                intcorners[0].max(intcorners[1]),
                intcorners[1].max(intcorners[2]),
                intcorners[2].max(intcorners[3]),
            ];

            let mut gen_side = |dim: PxDim, pos: PxPoint, u: i32, v: i32, w: i32, h: i32| {
                compositor.append_data(
                    pos + topleft,
                    dim,
                    uvpos
                        .add_size(&Size2D::new(u, v))
                        .to_f32()
                        .to_array()
                        .into(),
                    [w as f32, h as f32].into(),
                    0xFFFFFFFF,
                    0.0,
                    region_index,
                    true,
                );
            };

            // Left Top Right Bottom side order
            // Once again, we must manually inflate the sides here, but these are more tricky. To make it
            // a bit easier, we only inflate exactly the one pixel of the side that actually matters.
            gen_side(
                PxDim::new(sides.left() + 1.0, dim.height - corners[0] - corners[3]),
                PxPoint::new(-1.0, corners[0]),
                -1,
                intsides[1],
                intsides[0] + 1,
                0,
            );
            gen_side(
                PxDim::new(dim.width - corners[0] - corners[1], sides.top() + 1.0),
                PxPoint::new(corners[0], -1.0),
                intsides[0],
                -1,
                0,
                intsides[1] + 1,
            );
            gen_side(
                PxDim::new(sides.right() + 1.0, dim.height - corners[1] - corners[2]),
                PxPoint::new(dim.width - sides.right(), corners[1]),
                inner.width - intsides[2],
                intsides[1],
                intsides[2] + 1,
                0,
            );
            gen_side(
                PxDim::new(dim.width - corners[3] - corners[2], sides.bottom() + 1.0),
                PxPoint::new(corners[3], dim.height - sides.bottom()),
                intsides[0],
                inner.height - intsides[3],
                0,
                intsides[3] + 1,
            );

            // Inner area is just a flat color
            compositor.append_data(
                PxPoint::splat(corners[0]) + topleft,
                dim - PxDim::splat(corners[0] + corners[2]),
                [0.0, 0.0].into(),
                [0.0, 0.0].into(),
                self.fill.as_32bit().rgba,
                0.0,
                u8::MAX,
                true,
            );

            return Ok(());
        }

        let (region_uv, region_index) = {
            let mut atlas = driver.atlas.write();
            let region =
                atlas.cache_region(&driver.device, &self.id, dim.ceil().cast(), None, None)?;
            (region.uv, region.index)
        };

        // The region dimensions here can be wrong, because the region is rounded up to the nearest pixel.
        // However, properly fixing this requires changing how the SDF shader works so it can properly
        // emulate conservative rasterization. For now, we keep our original behavior of rounding up and
        // then letting the compositor squish the result slightly, which is actually pretty accurate.
        // TODO: Change this to be pixel-perfect by outputting the exact dimensions instead of rounded ones.

        // TODO: cache this if the inputs are identical
        driver.with_pipeline::<PIPELINE>(|pipeline| {
            pipeline.append(
                &Data {
                    pos: region_uv.min.to_f32().to_array().into(),
                    dim: region_uv.size().to_f32().to_array().into(),
                    border: self.border,
                    blur: self.blur,
                    corners: self.corners.into(),
                    fill: self.fill.as_32bit().rgba,
                    outline: self.outline.as_32bit().rgba,
                },
                region_index,
            )
        });

        compositor.append_data(
            area.topleft().add_size(&self.padding.topleft()),
            dim,
            region_uv.min.to_f32().to_array().into(),
            region_uv.size().to_f32().to_array().into(),
            0xFFFFFFFF,
            0.0,
            region_index,
            false,
        );

        Ok(())
    }
}

// Renderdoc Format:
// struct Data {
// 	float corners[4];
// 	float pos[2];
// 	float dim[2];
// 	float border;
// 	float blur;
// 	uint32_t fill;
// 	uint32_t outline;
// };
// Data d[];

#[derive(Debug, Clone, Copy, Default, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct Data {
    pub corners: Vec4f,
    pub pos: Vec2f,
    pub dim: Vec2f,
    pub border: f32,
    pub blur: f32,
    pub fill: u32,
    pub outline: u32,
}

#[derive(Debug)]
pub struct Shape<const KIND: u8> {
    data: HashMap<u8, Vec<Data>>,
    buffer: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
}

impl<const KIND: u8> super::Pipeline for Shape<KIND> {
    type Data = Data;

    fn append(&mut self, data: &Self::Data, layer: u8) {
        self.data.entry(layer).or_default().push(*data);
    }

    fn draw(&mut self, driver: &graphics::Driver, pass: &mut wgpu::RenderPass<'_>, layer: u8) {
        if let Some(data) = self.data.get_mut(&layer) {
            let size = data.len() * size_of::<Data>();
            if (self.buffer.size() as usize) < size {
                self.buffer.destroy();
                self.buffer = driver.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Shape Data"),
                    size: size.next_power_of_two() as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.group = Self::rebind(
                    &self.buffer,
                    &self.pipeline.get_bind_group_layout(0),
                    &driver.device,
                    &driver.atlas.read(),
                );
            }

            driver
                .queue
                .write_buffer(&self.buffer, 0, bytemuck::cast_slice(data.as_slice()));

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[0]);
            pass.draw(0..(data.len() as u32 * 6), 0..1);
            data.clear();
        }
    }
}

impl<const KIND: u8> Shape<KIND> {
    pub fn layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shape Bind Group"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZero::new(size_of::<crate::Mat4x4>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZero::new(size_of::<u32>() as u64),
                    },
                    count: None,
                },
            ],
        });

        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shape Pipeline"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        })
    }

    pub fn shader(device: &wgpu::Device) -> wgpu::ShaderModule {
        shaders::load_wgsl(device, "Shape", shaders::get("shape.wgsl").unwrap())
    }

    fn pipeline(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        device: &wgpu::Device,
        entry_point: &str,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                targets: &[Some(compositor::TARGET_BLEND)],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn rebind(
        buffer: &wgpu::Buffer,
        layout: &BindGroupLayout,
        device: &wgpu::Device,
        atlas: &Atlas,
    ) -> wgpu::BindGroup {
        let bindings = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: atlas.mvp.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: atlas.extent_buf.as_entire_binding(),
            },
        ];

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &bindings,
            label: None,
        })
    }

    fn new(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
        entry_point: &str,
    ) -> Self {
        let buffer = driver.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shape Data"),
            size: 32 * size_of::<Data>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = Self::pipeline(layout, shader, &driver.device, entry_point);

        let group = Self::rebind(
            &buffer,
            &pipeline.get_bind_group_layout(0),
            &driver.device,
            &driver.atlas.read(),
        );

        Self {
            data: HashMap::new(),
            buffer,
            pipeline,
            group,
        }
    }
}

impl Shape<0> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::new(layout, shader, driver, "rectangle"))
    }
}

impl Shape<1> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::new(layout, shader, driver, "triangle"))
    }
}

impl Shape<2> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::new(layout, shader, driver, "circle"))
    }
}

impl Shape<3> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::AnyPipeline> {
        Box::new(Self::new(layout, shader, driver, "arcs"))
    }
}
