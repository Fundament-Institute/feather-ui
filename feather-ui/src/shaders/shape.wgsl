@group(0) @binding(0)
var<uniform> MVP: mat4x4f;
@group(0) @binding(1)
var<storage, read> buffer: array<Data>;
@group(0) @binding(2)
var<uniform> extent: u32;

struct Data {
  pos: vec2f,
  dim: vec2f,
  border: f32,
  blur: f32,
  corners: vec4f,
  fill: u32,
  outline: u32,
}

struct VertexOutput {
  @invariant @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) index: u32,
}

const VERTX = array(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
const VERTY = array(0.0, 0.0, 1.0, 0.0, 1.0, 1.0);

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
  let vert = idx % 6;
  let index = idx / 6;
  var vpos = vec2(VERTX[vert], VERTY[vert]);
  let d = buffer[index];

  var mv: mat4x4f;
  mv[0] = vec4f(d.dim.x, 0f, 0f, 0f);
  mv[1] = vec4f(0f, d.dim.y, 0f, 0f);
  mv[2] = vec4f(0f, 0f, 1f, 0f);
  mv[3] = vec4f(d.pos.x + d.dim.x * 0.5f, d.pos.y + d.dim.y * 0.5f, 0f, 1f);
  let outpos = MVP * mv * vec4(vpos.x - 0.5f, vpos.y - 0.5f, 1f, 1f);

  return VertexOutput(outpos, vpos.xy, index);
}

fn u32_to_vec4(c: u32) -> vec4<f32> {
  return vec4<f32>(f32((c & 0xff000000u) >> 24u) / 255.0, f32((c & 0x00ff0000u) >> 16u) / 255.0, f32((c & 0x0000ff00u) >> 8u) / 255.0, f32(c & 0x000000ffu) / 255.0);
}

fn linearstep(low: f32, high: f32, x: f32) -> f32 {
  return clamp((x - low) / (high - low), 0.0f, 1.0f);
}

fn rectangle_sdf(samplePosition: vec2f, halfSize: vec2f, edges: vec4f) -> f32 {
  var edge: f32 = 20.0f;
  if (samplePosition.x > 0.0f) {
    edge = select(edges.z, edges.y, samplePosition.y < 0.0f);
  }
  else {
    edge = select(edges.w, edges.x, samplePosition.y < 0.0f);
  }

  let componentWiseEdgeDistance = abs(samplePosition) - halfSize + vec2f(edge);
  let outsideDistance = length(max(componentWiseEdgeDistance, vec2f(0.0f)));
  let insideDistance = min(max(componentWiseEdgeDistance.x, componentWiseEdgeDistance.y), 0.0f);
  return outsideDistance + insideDistance - edge;
}

@fragment
fn rectangle(input: VertexOutput) -> @location(0) vec4f {
  let d = buffer[input.index];
  // Ideally we would get DPI for both height and width, but for now we just assume DPI isn't weird
  let w = fwidth(d.dim.x * input.uv.x) * 0.5f * (1.0f + d.blur);
  let uv = (input.uv * d.dim) - (d.dim * 0.5f);

  let dist = rectangle_sdf(uv, d.dim * 0.5f, d.corners);
  let alpha = linearstep(w, - w, dist);
  let s = linearstep(w, - w, dist + d.border);
  let fill = u32_to_vec4(d.fill);
  let outline = u32_to_vec4(d.outline);

  return (vec4f(fill.rgb, 1f) * fill.a * s) + (vec4f(outline.rgb, 1f) * outline.a * clamp(alpha - s, 0.0f, 1.0f));
}