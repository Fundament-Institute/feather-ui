// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

use super::compositor;
use crate::color::sRGB;
//use crate::component::shape::ShapeKind;
use crate::graphics::{self, Vec2f, Vec4f};
use crate::reactive::{DynSignal, SignalMap, SignalTupleZip, const_signal, join, zip_pair};
use crate::render::atlas::{self, Atlas};
use crate::render::compositor::CompositorView;
use crate::{Canonicalize, PxDim, PxPoint, RenderError, shaders};
use core::f32;
use guillotiere::euclid::Size2D;
use num_traits::Zero;
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::sync::Arc;
use wgpu::BindGroupLayout;

#[derive(Clone)]
pub struct PreInstance<const KIND: u8> {
    pub padding: DynSignal<crate::PxPerimeter>,
    pub border: DynSignal<f32>,
    pub blur: DynSignal<f32>,
    pub fill: DynSignal<sRGB>,
    pub outline: DynSignal<sRGB>,
    pub corners: DynSignal<[f32; 4]>,
    pub driver: Arc<graphics::Driver>,
}

impl<const KIND: u8> crate::render::Prerender for PreInstance<KIND> {
    type R = Instance<KIND>;

    fn prerender(&self, area: crate::DynSignal<crate::PxRect>) -> Self::R {
        Instance::<KIND>::new(
            self.padding.clone(),
            self.border.clone(),
            self.blur.clone(),
            self.fill.clone(),
            self.outline.clone(),
            self.corners.clone(),
            area,
            self.driver.clone(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum InstanceResult {
    Empty,
    Single(compositor::Data),
    Rect([compositor::Data; 9]),
    Error(RenderError),
}

pub struct Instance<const KIND: u8> {
    values: DynSignal<InstanceResult>,
    cache: crate::reactive::Sampler<
        dyn crate::reactive::SignalProvider<
                Item = Result<(Data, Size2D<i32, crate::Pixel>), RenderError>,
            >,
    >,
}

impl<const KIND: u8> Instance<KIND> {
    pub fn new(
        padding: DynSignal<crate::PxPerimeter>,
        border: DynSignal<f32>,
        blur: DynSignal<f32>,
        fill: DynSignal<sRGB>,
        outline: DynSignal<sRGB>,
        corners: DynSignal<[f32; 4]>,
        area: DynSignal<crate::PxRect>,
        driver: Arc<graphics::Driver>,
    ) -> Self {
        let dim = zip_pair(area.clone(), padding.clone(), |a, p| {
            a.dim() - p.bottomright() - p.topleft()
        });

        let intcorners = zip_pair(corners.clone(), border.clone(), |c, b| {
            c.map(|x| x.max(*b)).map(|x| x.ceil() as i32)
        });

        // If the border is larger than the corner itself, pretend the size of that
        // corner is the border.
        let inner = intcorners.clone().map(|c| {
            let intsides = [
                c[0].max(c[3]),
                c[0].max(c[1]),
                c[1].max(c[2]),
                c[2].max(c[3]),
            ];

            // Here we generate a rounded block equal to exactly left side + 3 pixels +
            // right side
            atlas::Size::new(
                intsides[0] + intsides[2] + 3, // left + right
                intsides[1] + intsides[3] + 3, // top + bottom
            )
        });

        let driver2 = driver.clone();
        let opt_reservation = (
            intcorners.clone(),
            border.clone(),
            blur.clone(),
            fill.clone(),
            outline.clone(),
            inner.clone(),
        )
            .zip()
            .map_mut(
                move |(intcorners, border, blur, fill, outline, inner), old| {
                    // We reserve an additional 2 pixel border around each side of our rect for
                    // sampling purposes. It must be 2 pixels because we have to inflate
                    // the rect by 1 pixel for fractional draws already, which means we
                    // need an additional transparent pixel of buffer to cover all possible sampling
                    // scenarios.
                    match old {
                        None | Some(Ok(_)) => driver2
                            .with_pipeline::<Shape<KIND>, Result<(Data, atlas::Size), RenderError>>(
                                |pipeline| {
                                    pipeline.update(
                                        &driver2,
                                        *inner + atlas::Size::new(4, 4),
                                        Data {
                                            pos: [2.0; 2].into(),
                                            dim: inner.to_f32().to_array().into(),
                                            border: *border,
                                            blur: *blur,
                                            // We use corners raised to the nearest pixel so we can cut out the
                                            // corners neatly
                                            corners: intcorners.map(|x| x as f32).into(),
                                            fill: fill.as_32bit().rgba,
                                            outline: outline.as_32bit().rgba,
                                        },
                                        old.map(|x| x.unwrap()),
                                        true,
                                    )
                                },
                            ),
                        Some(Err(e)) => Err(e),
                    }
                },
            );

        let driver2 = driver.clone();
        let opt_region = opt_reservation.clone().map(move |key| {
            key.map(|k| {
                driver2.with_pipeline::<Shape<KIND>, (atlas::PxBox, u8)>(|pipeline| {
                    pipeline.get(&k).expect("This lookup should never fail!")
                })
            })
        });

        let opt_data = (
            opt_region.clone(),
            intcorners.clone(),
            corners.clone(),
            dim.clone(),
            area.clone(),
            padding.clone(),
            fill.clone(),
            inner.clone(),
        )
            .zip()
            .map(
                |(region, intcorners, corners, dim, area, padding, fill, inner)| {
                    match region {
                        Ok(r) => {
                            // The only reason this works is because we set the uvdim here to 0 on the axis
                            // that is being extended, which ensures no interpolation of the UV
                            // coordinate happens along that axis

                            // We add data here starting from the topleft corner and going clockwise around
                            // the rect:
                            // 1 6 2
                            // 5 9 7
                            // 4 8 3

                            // Pretend all corners are 1 pixel larger (this works because our buffer is 3
                            // pixels)
                            let corners = corners.map(|x| x + 1.0);
                            let intcorners = intcorners.map(|x| x + 1);
                            let mut data: [MaybeUninit<compositor::Data>; _] =
                                [const { MaybeUninit::uninit() }; 9];

                            let topleft = area.topleft().add_size(&padding.topleft()).to_vector();
                            // Add 2 to account for the 2 pixel transparent border
                            let uvpos = r.0.min.add_size(&Size2D::splat(2));
                            let mut index = 0;
                            let mut gen_corner = |pos: PxPoint, corner: f32, u: i32, v: i32| {
                                let intdim = PxDim::splat(corner.ceil());
                                data[index].write(compositor::Data::new(
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
                                    r.1,
                                    true,
                                    false,
                                ));
                                index += 1;
                            };

                            // This is nontrivial, because this must be assembled in raw mode, which means
                            // we must do the directional inflation here ourselves. This amounts
                            // to changing every 0 into a -1, but *not* changing the non-zero positions and
                            // instead adding 1 to the dimensions, which on corners means adding
                            // yet another +1 to the corner size.
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

                            // We can't just do sides.ceil() because the result is not the same as ceiling
                            // both corners and adding them.
                            let intsides = [
                                intcorners[0].max(intcorners[3]),
                                intcorners[0].max(intcorners[1]),
                                intcorners[1].max(intcorners[2]),
                                intcorners[2].max(intcorners[3]),
                            ];

                            let mut gen_side =
                                |dim: PxDim, pos: PxPoint, u: i32, v: i32, w: i32, h: i32| {
                                    data[index].write(compositor::Data::new(
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
                                        r.1,
                                        true,
                                        false,
                                    ));
                                    index += 1;
                                };

                            // Left Top Right Bottom side order
                            // Once again, we must manually inflate the sides here, but these are more
                            // tricky. To make it a bit easier, we only inflate exactly the one
                            // pixel of the side that actually matters.
                            gen_side(
                                PxDim::new(
                                    sides.left() + 1.0,
                                    dim.height - corners[0] - corners[3],
                                ),
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
                                PxDim::new(
                                    sides.right() + 1.0,
                                    dim.height - corners[1] - corners[2],
                                ),
                                PxPoint::new(dim.width - sides.right(), corners[1]),
                                inner.width - intsides[2],
                                intsides[1],
                                intsides[2] + 1,
                                0,
                            );
                            gen_side(
                                PxDim::new(
                                    dim.width - corners[3] - corners[2],
                                    sides.bottom() + 1.0,
                                ),
                                PxPoint::new(corners[3], dim.height - sides.bottom()),
                                intsides[0],
                                inner.height - intsides[3],
                                0,
                                intsides[3] + 1,
                            );

                            // Inner area is just a flat color
                            data[index].write(compositor::Data::new(
                                PxPoint::splat(corners[0]) + topleft,
                                *dim - PxDim::splat(corners[0] + corners[2]),
                                [0.0, 0.0].into(),
                                [0.0, 0.0].into(),
                                fill.as_32bit().rgba,
                                0.0,
                                u8::MAX,
                                true,
                                false,
                            ));
                            InstanceResult::Rect(unsafe {
                                std::mem::transmute::<_, [compositor::Data; 9]>(data)
                            })
                        }
                        Err(e) => InstanceResult::Error(*e),
                    }
                },
            )
            .into_dyn_signal();

        let driver2 = driver.clone();
        let reservation = (
            dim.clone(),
            corners.clone(),
            border.clone(),
            blur.clone(),
            fill.clone(),
            outline.clone(),
        )
            .zip()
            .map_mut(
                move |(dim, corners, border, blur, fill, outline), old| match old {
                    None | Some(Ok(_)) => driver2
                        .with_pipeline::<Shape<KIND>, Result<(Data, atlas::Size), RenderError>>(
                            |pipeline| {
                                pipeline.update(
                                    &driver2,
                                    dim.ceil().cast(),
                                    Data {
                                        pos: [0.0; 2].into(),
                                        dim: dim.to_array().into(),
                                        border: *border,
                                        blur: *blur,
                                        corners: corners.into(),
                                        fill: fill.as_32bit().rgba,
                                        outline: outline.as_32bit().rgba,
                                    },
                                    old.map(|x| x.unwrap()),
                                    false,
                                )
                            },
                        ),
                    Some(Err(e)) => Err(e),
                },
            );

        let driver2 = driver.clone();
        let region = reservation.clone().map(move |key| {
            key.map(|k| {
                driver2.with_pipeline::<Shape<KIND>, (atlas::PxBox, u8)>(|pipeline| {
                    pipeline.get(&k).expect("This lookup should never fail!")
                })
            })
        });

        // The region dimensions here can be wrong, because the region is rounded up to
        // the nearest pixel. However, properly fixing this requires changing
        // how the SDF shader works so it can properly emulate conservative
        // rasterization. For now, we keep our original behavior of rounding up and
        // then letting the compositor squish the result slightly, which is actually
        // pretty accurate. TODO: Change this to be pixel-perfect by outputting
        // the exact dimensions instead of rounded ones.

        let data = (area.clone(), padding.clone(), dim.clone(), region.clone())
            .zip()
            .map(|(area, padding, dim, region)| match region {
                Ok(r) => InstanceResult::Single(compositor::Data::new(
                    area.topleft().add_size(&padding.topleft()),
                    *dim,
                    r.0.min.to_f32().to_array().into(),
                    r.0.size().to_f32().to_array().into(),
                    0xFFFFFFFF,
                    0.0,
                    r.1,
                    false,
                    false,
                )),
                Err(e) => InstanceResult::Error(*e),
            })
            .into_dyn_signal();

        let empty = const_signal(InstanceResult::Empty).into_dyn_signal();
        let blank = (area.clone(), padding.clone(), dim.clone(), fill.clone())
            .zip()
            .map(|(area, padding, dim, fill)| {
                InstanceResult::Single(compositor::Data::new(
                    area.topleft().add_size(&padding.topleft()),
                    *dim,
                    [0.0, 0.0].into(),
                    [0.0, 0.0].into(),
                    fill.as_32bit().rgba,
                    0.0,
                    u8::MAX,
                    false,
                    false,
                ))
            })
            .into_dyn_signal();

        let choice = join((dim.clone(), corners.clone(), border.clone()).zip().map_ex(
            move |(dim, corners, border)| {
                if dim.width <= 0.0 || dim.height <= 0.0 {
                    return empty.clone();
                }

                if KIND == /*ShapeKind::RoundRect as u8*/ 0
                    && corners.iter().all(|x| x.is_zero())
                    && border.is_zero()
                {
                    return blank.clone();
                }

                let perimeter = [
                    dim.height - corners[0] - corners[3],
                    dim.width - corners[0] - corners[1],
                    dim.height - corners[1] - corners[2],
                    dim.width - corners[2] - corners[3],
                ];

                // RoundRects have a specific optimization, but only if no edge length is less
                // than 2 pixels
                if KIND == /*ShapeKind::RoundRect as u8*/ 0 && perimeter.iter().all(|x| *x >= 2.0) {
                    opt_data.clone()
                } else {
                    data.clone()
                }
            },
        ));

        let none_reservation = const_signal(Ok((
            Data::default(),
            Size2D::<i32, crate::Pixel>::default(),
        )));

        let region_choice = join((dim.clone(), corners.clone(), border.clone()).zip().map_ex(
            move |(dim, corners, border)| {
                let perimeter = [
                    dim.height - corners[0] - corners[3],
                    dim.width - corners[0] - corners[1],
                    dim.height - corners[1] - corners[2],
                    dim.width - corners[2] - corners[3],
                ];

                if dim.width <= 0.0
                    || dim.height <= 0.0
                    || (KIND == /*ShapeKind::RoundRect as u8*/ 0
                        && corners.iter().all(|x| x.is_zero())
                        && border.is_zero())
                {
                    none_reservation.clone().into_dyn_signal()
                } else if KIND == /*ShapeKind::RoundRect as u8*/ 0
                    && perimeter.iter().all(|x| *x >= 2.0)
                {
                    opt_reservation.clone().into_dyn_signal()
                } else {
                    reservation.clone().into_dyn_signal()
                }
            },
        ));

        Self {
            values: choice.into_dyn_signal(),
            cache: crate::Sampler::new(region_choice.into_dyn_signal()),
        }
    }
}

impl<const KIND: u8> super::Renderable for Instance<KIND> {
    fn render(
        &self,
        parent_pos: crate::PxPoint,
        driver: &graphics::Driver,
        compositor: &mut CompositorView<'_>,
    ) -> Result<(), crate::RenderError> {
        // Be sure we only draw our cache if it actually changed, or is in an error state.
        if let Some(cache) = self.cache.partial_sample(|x| x.is_err()) {
            match &*cache {
                Ok(key) => driver.with_pipeline::<Shape<KIND>, ()>(|pipeline| {
                    if let Some((uv, index)) = pipeline.get(key) {
                        pipeline.push(key.0, uv, index);
                    }
                }),
                Err(e) => return Err((*e).into()),
            }
        }

        let data = crate::reactive::sample(&self.values);
        match &*data {
            InstanceResult::Empty => (),
            InstanceResult::Single(data) => {
                compositor.append_data(data.offset(parent_pos));
            }
            InstanceResult::Rect(data) => {
                for datum in data {
                    compositor.append_data(datum.offset(parent_pos));
                }
            }
            InstanceResult::Error(e) => {
                return Err((*e).into());
            }
        }

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

// TODO: Maybe use NotNaN from ordered_float if this doesn't mess up alignment?
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

// We manually implement Eq because no NaNs should be in Data
impl Eq for Data {}

impl std::hash::Hash for Data {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.corners.hash(state);
        self.pos.hash(state);
        self.dim.hash(state);
        self.border.canonical_bits().hash(state);
        self.blur.canonical_bits().hash(state);
        self.fill.hash(state);
        self.outline.hash(state);
    }
}

#[derive(Debug)]
pub struct Shape<const KIND: u8> {
    data: HashMap<u8, Vec<Data>>,
    buffer: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    refcount: HashMap<(Data, atlas::Size), (atlas::Region, usize)>,
}

impl<const KIND: u8> Shape<KIND> {
    #[inline(always)]
    fn get(&self, key: &(Data, atlas::Size)) -> Option<(atlas::PxBox, u8)> {
        self.refcount.get(key).map(|(r, _)| (r.uv, r.index))
    }

    #[inline(always)]
    fn push(&mut self, mut data: Data, uv: atlas::PxBox, index: u8) {
        data.pos += uv.min.to_f32().to_array();
        self.data.entry(index).or_default().push(data);
    }

    fn update(
        &mut self,
        driver: &graphics::Driver,
        uvdim: atlas::Size,
        data: Data,
        old: Option<(Data, atlas::Size)>,
        clear: bool,
    ) -> Result<(Data, atlas::Size), RenderError> {
        // If we have an old value, see if it's the same as the one we passed in. If it isn't, we have to
        // delete the old value from the refcount.
        if let Some((data_old, uvdim_old)) = old {
            if data_old != data || uvdim_old != uvdim {
                if let std::collections::hash_map::Entry::Occupied(mut v) =
                    self.refcount.entry((data_old, uvdim_old))
                {
                    if v.get().1 <= 1 {
                        driver.atlas.write().destroy(&mut v.get_mut().0);
                        v.remove();
                    } else {
                        v.get_mut().1 -= 1;
                    }
                }
            } else if self.refcount.contains_key(&(data, uvdim)) {
                return Ok((data, uvdim));
            }
        }

        // We check to see if the data key we have is already being
        // used for something else, and increment the refcount if so. Otherwise, we
        // allocate a new region.
        match self.refcount.entry((data, uvdim)) {
            std::collections::hash_map::Entry::Occupied(mut occupied_entry) => {
                occupied_entry.get_mut().1 += 1;
                occupied_entry.into_mut()
            }
            std::collections::hash_map::Entry::Vacant(vacant_entry) => vacant_entry.insert((
                driver.atlas.write().reserve(
                    &driver.device,
                    uvdim,
                    None,
                    if clear { Some(&driver.queue) } else { None },
                )?,
                1,
            )),
        };

        Ok((data, uvdim))
    }
}

impl<const KIND: u8> super::Pipeline for Shape<KIND> {
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

    fn destroy(&mut self, driver: &graphics::Driver) {
        for (_, (mut region, _)) in self.refcount.drain() {
            driver.atlas.write().destroy(&mut region);
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
            refcount: HashMap::new(),
        }
    }
}

impl Shape<0> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::Pipeline> {
        Box::new(Self::new(layout, shader, driver, "rectangle"))
    }
}

impl Shape<1> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::Pipeline> {
        Box::new(Self::new(layout, shader, driver, "triangle"))
    }
}

impl Shape<2> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::Pipeline> {
        Box::new(Self::new(layout, shader, driver, "circle"))
    }
}

impl Shape<3> {
    pub fn create(
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        driver: &graphics::Driver,
    ) -> Box<dyn super::Pipeline> {
        Box::new(Self::new(layout, shader, driver, "arcs"))
    }
}
