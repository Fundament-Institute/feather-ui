// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

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
  return vec4f(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z), srgb_to_linear(c.w));
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

fn linearstep(low: f32, high: f32, x: f32) -> f32 {
  return clamp((x - low) / (high - low), 0.0f, 1.0f);
}