@group(0) @binding(1) 
var<uniform> PosDim: vec4f;
@group(0) @binding(2) 
var<uniform> DimBorderBlur: vec4f;
@group(0) @binding(3) 
var<uniform> Corners: vec4f;
@group(0) @binding(4) 
var<uniform> Fill: vec4f;
@group(0) @binding(5) 
var<uniform> Outline: vec4f;

fn linearstep(low: f32, high: f32, x: f32) -> f32 { return clamp((x - low) / (high - low), 0.0f, 1.0f); }  

fn rectangle(samplePosition: vec2f, halfSize: vec2f, edges: vec4f) -> f32 {
    var edge: f32 = 20.0f;  
    if(samplePosition.x > 0.0f) {
      edge = select(edges.z, edges.y, samplePosition.y < 0.0f);
    } else { 
      edge = select(edges.w, edges.x, samplePosition.y < 0.0f);
    }
    
    let componentWiseEdgeDistance = abs(samplePosition) - halfSize + vec2f(edge);  
    let outsideDistance = length(max(componentWiseEdgeDistance, vec2f(0.0f)));  
    let insideDistance = min(max(componentWiseEdgeDistance.x, componentWiseEdgeDistance.y), 0.0f);  
    return outsideDistance + insideDistance - edge;  
}  

@fragment 
fn main(@location(0) pos: vec2f) -> @location(0) vec4f {
    // Ideally we would get DPI for both height and width, but for now we just assume DPI isn't weird
    let w = fwidth(PosDim.z*pos.x) * 0.5f * (1.0f + DimBorderBlur.w);  
    let uv = (pos * PosDim.zw) - (PosDim.zw * 0.5f);  
    
    let dist = rectangle(uv, PosDim.zw * 0.5f, Corners);  
  	let alpha = linearstep(w, -w, dist);  
    let s = linearstep(w, -w, dist + DimBorderBlur.z);   
    
    return (vec4f(Fill.rgb, 1f)*Fill.a*s) + (vec4f(Outline.rgb, 1f)*Outline.a*clamp(alpha - s, 0.0f, 1.0f));  
}
