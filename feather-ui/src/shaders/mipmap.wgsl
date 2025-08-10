// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

const UNITX = array(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
const UNITY = array(0.0, 0.0, 1.0, 0.0, 1.0, 1.0);

@group(0) @binding(0)
var<uniform> MVP: mat4x4f;
@group(0) @binding(1)
var<uniform> buf: array<vec4<f32>, 256>;
@group(0) @binding(2)
var sampling: sampler;
@group(0) @binding(3)
var source: texture_2d<f32>;
@group(0) @binding(4)
var<uniform> basesize: vec2<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
  let vert = idx % 6;
  let index = idx / 6;
  var vpos = vec2(UNITX[vert], UNITY[vert]);
  let d = buf[index];

  let uv = d.xy / basesize;
  let uvdim = (d.zw - d.xy) / basesize;

  return VertexOutput(MVP * vec4(d.xy + ((d.zw - d.xy) * vpos), 1f, 1f), uv + (uvdim * vpos));
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(source, sampling, uv);
}