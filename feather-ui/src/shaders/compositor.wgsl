// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

//#import "feather.wgsl"
const UNITX = array(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
const UNITY = array(0.0, 0.0, 1.0, 0.0, 1.0, 1.0);
const IDENTITY_MAT4 = mat4x4f(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

fn srgb_to_linear(c: f32) -> f32 {
  if c <= 0.04045 {
    return c / 12.92;
  }
  else {
    return pow((c + 0.055) / 1.055, 2.4);
  }
}

fn srgb_to_linear_vec4(c: vec4<f32>) -> vec4<f32> {
  return vec4f(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z), c.w);
}

fn linear_to_srgb(c: f32) -> f32 {
  if c < 0.0031308 {
    return c * 12.92;
  }
  else {
    return 1.055 * pow(c, (1.0 / 2.4)) - (0.055);
  }
}

fn linear_to_srgb_vec4(c: vec4f) -> vec4f {
  return vec4f(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z), c.w);
}

fn u32_to_vec4(c: u32) -> vec4<f32> {
  return vec4<f32>(f32((c & 0xff000000u) >> 24u) / 255.0, f32((c & 0x00ff0000u) >> 16u) / 255.0, f32((c & 0x0000ff00u) >> 8u) / 255.0, f32(c & 0x000000ffu) / 255.0);
}

// Rotates a point around the origin
fn rotate(p: vec2f, r: f32) -> vec2f {
  let sr = sin(r);
  let cr = cos(r);
  return vec2f(p.x * cr - p.y * sr, p.y * cr + p.x * sr);
}

@group(0) @binding(0)
var<uniform> MVP: mat4x4f;
@group(0) @binding(1)
var<storage, read> buf: array<Data>;
@group(0) @binding(2)
var atlas: texture_2d_array<f32>;
@group(0) @binding(3)
var sampling: sampler;
@group(0) @binding(4)
var<storage, read> cliprects: array<vec4f>;
@group(0) @binding(5)
var<uniform> extent: u32;

struct Data {
  pos: vec2f,
  dim: vec2f,
  uv: vec2f,
  uvdim: vec2f,
  color: u32,
  rotation: f32,
  texclip: u32,
}

struct VertexOutput {
  @invariant @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2f,
  @location(1) dist: vec2f,
  @location(2) @interpolate(flat) index: u32,
  @location(3) color: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
  let vert = idx % 6;
  let index = idx / 6;
  var vpos = vec2(UNITX[vert], UNITY[vert]);
  let d = buf[index];
  // Setting this flag *disables* inflation, so we invert it by comparing to 0
  let inflate = (d.texclip & 0x80000000) == 0;

  var inflate_dim = d.dim;
  var inflate_pos = d.pos;
  var inflate_uv = d.uv;
  var inflate_uvdim = d.uvdim;

  if (inflate) {
    // To emulate conservative rasterization, we must inflate the quad by 0.5 pixels outwards. This
    // is done by increasing the total dimension size by 1, then subtracing 0.5 from the position.
    inflate_dim += vec2f(1);
    inflate_pos -= vec2f(0.5);

    // We must also compensate the UV coordinates, but this is trickier because they could already be
    // scaled differently. We acquire the size of a UV pixel by dividing the UV dimensions by the true
    // dimensions. Thus, if we have a 2x2 UV lookup scaled to a 4x4 square, one scaled UV pixel is 0.5
    let uv_pixel = d.uvdim / d.dim;
    inflate_uvdim += vec2f(uv_pixel);
    inflate_uv -= vec2f(uv_pixel * 0.5);
  }

  var pos = vpos;
  pos *= inflate_dim;
  if d.rotation != 0.0f {
    // For complicated reasons, we have to rotate around the center of whatever we're rendering. This
    // allows lines to precisely position themselves, and also conveniently works with inflation enabled.
    let half = inflate_dim * vec2f(0.5);
    pos = rotate(pos - half, d.rotation) + half;
  }
  pos += inflate_pos;

  let out_pos = MVP * vec4(pos.x, pos.y, 1f, 1f);

  var source = IDENTITY_MAT4;
  let uv = inflate_uv / f32(extent);
  let uvdim = inflate_uvdim / f32(extent);

  var out_uv = vpos * uvdim;
  // If the rotation is negative it's applied to the UV rectangle as well
  if d.rotation < 0.0f {
    let half = uvdim * vec2f(0.5);
    out_uv = rotate(out_uv - half, d.rotation) + half;
  }
  out_uv += uv;

  let color = srgb_to_linear_vec4(u32_to_vec4(d.color));
  let dist = (vpos - vec2f(0.5f)) * inflate_dim;

  return VertexOutput(out_pos, out_uv.xy, dist, index, color);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let d = buf[input.index];
  let clip = d.texclip & 0x0000FFFF;
  let tex = (d.texclip & 0x7FFF0000) >> 16;
  let inflate = (d.texclip & 0x80000000) == 0;

  if clip > 0 {
    let r = cliprects[clip];
    if !(input.uv.x >= r.x && input.uv.y >= r.y && input.uv.x < r.z && input.uv.y < r.z) {
      discard;
    }
  }

  var color = vec4f(input.color.rgb * input.color.a, input.color.a);
  var uv = input.uv;

  if (inflate) {
    // TODO: Allow turning off the edge corrections
    // A pixel-perfect texture lookup at pixel 0,0 actually samples at 0.5,0.5, at the center of the
    // texel. Hence, if we simply clamp from 0,0 to height,width, this doesn't prevent bleedover when
    // we get a misaligned pixel that tries to sample the texel at 0,0, which will bleed over into the
    // texels next to it. As a result, we must clamp from 0.5,0.5 to width - 0.5, height - 0.5
    let uvmin = (d.uv + vec2f(0.5)) / f32(extent);
    let uvmax = (d.uv + d.uvdim - vec2f(0.5)) / f32(extent);
    uv = clamp(input.uv, uvmin, uvmax);

    // We get the pixel distance from the center of our quad, which we then use to do a precise alpha
    // dropoff, which recreates anti-aliasing.
    let dist = 1.0 - (abs(input.dist) - (d.dim * 0.5) + 0.5);
    color *= clamp(min(dist.x, dist.y), 0.0, 1.0);
  }

  if tex == 0x7FFF {
    return color;
  }

  return textureSample(atlas, sampling, uv, tex) * color;
}