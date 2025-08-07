var<private> pos: array<vec2<f32>, 4> = array<vec2<f32>, 4>(vec2<f32>(- 1.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(- 1.0, - 1.0), vec2<f32>(1.0, - 1.0));

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;
  output.uv = pos[vertexIndex] * vec2<f32>(0.5, - 0.5) + vec2<f32>(0.5);
  output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
  return output;
}

@group(0) @binding(0)
var samp: sampler;
@group(0) @binding(1)
var atlas: texture_2d<f32>;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(atlas, samp, uv);
}