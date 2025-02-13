@group(0) @binding(0) 
var<uniform> MVP: mat4x4f;
@group(0) @binding(1) 
var<uniform> PosDim: vec4f;

struct VertexOutput {
    @location(0) pos: vec2f,
    @builtin(position) gl_Position: vec4f,
}

@vertex 
fn main(@location(0) vPos: vec2f) -> VertexOutput {
  var mv: mat4x4f;
  mv[0] = vec4f(PosDim.z, 0f,0f,0f);
  mv[1] = vec4f(0f, PosDim.w,0f,0f);
  mv[2] = vec4f(0f, 0f, 1f, 0f);
  mv[3] = vec4f(PosDim.x + PosDim.z*0.5f, PosDim.y + PosDim.w*0.5f,0f,1f);
  let outpos = MVP * mv * vec4(vPos.x - 0.5f, vPos.y - 0.5f, 1f, 1f);  
  
  return VertexOutput(vPos.xy, outpos);
}  
