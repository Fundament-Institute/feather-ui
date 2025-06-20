// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

//#import "feather.wgsl"
const UNITX = array(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
const UNITY = array(0.0, 0.0, 1.0, 0.0, 1.0, 1.0);
const IDENTITY_MAT4 = mat4x4f(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

fn linearstep(low: f32, high: f32, x: f32) -> f32 {
  return clamp((x - low) / (high - low), 0.0f, 1.0f);
}

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

fn scale_matrix(m: mat4x4f, x: f32, y: f32) -> mat4x4f {
  var r = m;
  r[0][0] *= x;
  r[1][1] *= y;
  return r;
}

fn translate_matrix(m: mat4x4f, x: f32, y: f32) -> mat4x4f {
  var r = m;
  r[3][0] += x;
  r[3][1] += y;
  return r;
}

fn rotation_matrix(x: f32, y: f32, r: f32) -> mat4x4f {
  let cr = cos(r);
  let sr = sin(r);

  return mat4x4f(cr, sr, 0, 0, - sr, cr, 0, 0, 0, 0, 1, 0, x - x * cr + y * sr, y - x * sr - y * cr, 0, 1);
}

fn u32_to_vec4(c: u32) -> vec4<f32> {
  return vec4<f32>(f32((c & 0xff000000u) >> 24u) / 255.0, f32((c & 0x00ff0000u) >> 16u) / 255.0, f32((c & 0x0000ff00u) >> 8u) / 255.0, f32(c & 0x000000ffu) / 255.0);
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
  var transform = IDENTITY_MAT4;

  if d.rotation != 0.0f {
    transform = rotation_matrix(d.pos.x, d.pos.y, d.rotation);
  }

  transform = scale_matrix(transform, d.dim.x, d.dim.y);
  transform = translate_matrix(transform, d.pos.x + d.dim.x * 0.5, d.pos.y + d.dim.y * 0.5);

  var px_pos = transform * vec4(vpos.x - 0.5f, vpos.y - 0.5f, 1f, 1f);
  let out_pos = MVP * px_pos;

  var source = IDENTITY_MAT4;
  let uv = d.uv / f32(extent);
  let uvdim = d.uvdim / f32(extent);

  // If the rotation is negative it's applied to the UV rectangle as well
  if d.rotation < 0.0f {
    source = rotation_matrix(uv.x, uv.y, d.rotation);
  }

  source = scale_matrix(source, uvdim.x, uvdim.y);
  source = translate_matrix(source, uv.x, uv.y);

  var out_uv = source * vec4(vpos.x, vpos.y, 1f, 1f);
  let color = srgb_to_linear_vec4(u32_to_vec4(d.color));
  let dist = px_pos.xy - d.pos - (d.dim * 0.5);
  return VertexOutput(out_pos, out_uv.xy, dist, index, color);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let d = buf[input.index];
  let clip = d.texclip & 0x0000FFFF;
  let tex = (d.texclip & 0xFFFF0000) >> 16;

  if clip > 0 {
    let r = cliprects[clip];
    if !(input.uv.x >= r.x && input.uv.y >= r.y && input.uv.x < r.z && input.uv.y < r.z) {
      discard;
    }
  }

  var color = vec4f(input.color.rgb * input.color.a, input.color.a);
  // A pixel-perfect texture lookup at pixel 0,0 actually samples at 0.5,0.5, at the center of the
  // texel. Hence, if we simply clamp from 0,0 to height,width, this doesn't prevent bleedover when
  // we get a misaligned pixel that tries to sample the texel at 0,0, which will bleed over into the
  // texels next to it. As a result, we must clamp from 0.5,0.5 to width - 0.5, height - 0.5
  let uvmin = (d.uv + vec2f(0.5)) / f32(extent);
  let uvmax = (d.uv + d.uvdim - vec2f(0.5)) / f32(extent);
  let uv = clamp(input.uv, uvmin, uvmax);
  //let uv = input.uv;

  //let dim = d.dim * 0.5;
  //let fade = max(linearstep(dim.x - 0.5, dim.x + 0.5, abs(input.dist.x)), linearstep(dim.y - 0.5, dim.y + 0.5, abs(input.dist.y)));
  //color *= fade;

  if tex == 0xFFFF {
    return color;
  }

  return textureSample(atlas, sampling, uv, tex) * color;
}