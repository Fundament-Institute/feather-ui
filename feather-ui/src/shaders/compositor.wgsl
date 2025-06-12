@group(0) @binding(0)
var<uniform> MVP: mat4x4f;
@group(0) @binding(1)
var<storage, read> buffer: array<Data>;
@group(0) @binding(2)
var<uniform> atlas: texture_2d_array<f32>;
@group(0) @binding(3)
var<uniform> sampling: sampler;
@group(0) @binding(4)
var<uniform> cliprects: array<vec4f>;
@group(0) @binding(5)
var<uniform> extent: u32;

struct Data {
  pos: vec2f,
  dim: vec2f,
  uv: vec2<u32>,
  uvdim: vec2<u32>,
  color: u32,
  rotation: f32,
  texclip: u32,
}

struct VertexOutput {
  @invariant @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) index: u32,
  @location(2) @interpolate(linear) color: vec4f,
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

const identity_mat4 = mat4x4f(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

fn srgb_to_linear(c: f32) -> f32 {
  if c <= 0.04045 {
    return c / 12.92;
  }
  else {
    return pow((c + 0.055) / 1.055, 2.4);
  }
}

const VERTX = array(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
const VERTY = array(0.0, 0.0, 1.0, 0.0, 1.0, 1.0);

fn u32_to_vec4(c: u32) -> vec4<f32> {
  return vec4<f32>(f32((c & 0xff000000u) >> 24u) / 255.0, f32((c & 0x00ff0000u) >> 16u) / 255.0, f32((c & 0x0000ff00u) >> 8u) / 255.0, f32(c & 0x000000ffu) / 255.0);
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
  let vert = idx % 6;
  let index = idx / 6;
  var vpos = vec2(VERTX[vert], VERTY[vert]);
  let d = buffer[index];
  var transform = identity_mat4;

  if d.rotation != 0.0f {
    transform = rotation_matrix(d.pos.x, d.pos.y, d.rotation);
  }

  transform = scale_matrix(transform, d.dim.x, d.dim.y);
  transform = translate_matrix(transform, d.pos.x + d.dim.x * 0.5, d.pos.y + d.dim.y * 0.5);

  let out_pos = MVP * transform * vec4(vpos.x - 0.5f, vpos.y - 0.5f, 1f, 1f);

  var source = identity_mat4;
  let uv = vec2f(d.uv) / f32(extent);
  let uvdim = vec2f(d.uvdim) / f32(extent);

  // If the rotation is negative it's applied to the UV rectangle as well
  if d.rotation < 0.0f {
    source = rotation_matrix(uv.x, uv.y, d.rotation);
  }

  source = scale_matrix(source, uvdim.x, uvdim.y);
  source = translate_matrix(source, uv.x, uv.y);

  var out_uv = MVP * source * vec4(vpos.x, vpos.y, 0f, 0f);
  let color = u32_to_vec4(d.color);
  return VertexOutput(out_pos, out_uv.xy, index, color);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let d = buffer[input.index];
  let clip = d.texclip & 0x0000FFFF;
  let tex = (d.texclip & 0xFFFF0000) >> 16;

  if clip > 0 {
    let r = cliprects[clip];
    if !(input.uv.x >= r.x && input.uv.y >= r.y && input.uv.x < r.z && input.uv.y < r.z) {
      discard;
    }
  }
  if tex == 0xFFFF {
    return d.color;
  }
  return textureSample(atlas[tex], sampling, input.uv) * d.color;
}