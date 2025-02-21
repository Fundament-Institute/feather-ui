struct VertexOutput {
    @location(0) vLineCenter: vec2f,
    @builtin(position) gl_Position: vec4f,
}

@group(0) @binding(0) 
var<uniform> MVP: mat4x4<f32>;
@group(0) @binding(1) 
var<uniform> ViewPort: vec2<f32>;

@vertex 
fn main(@location(0) vPos: vec2f) -> VertexOutput {
    let pp = (MVP * vec4f(vPos.x, vPos.y, 0f, 1f));
    let vLineCenter = ((0.5f * (pp.xy + vec2f(1f, 1f))) * ViewPort);
    return VertexOutput(vLineCenter, pp);
}
