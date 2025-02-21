@group(0) @binding(2) 
var<uniform> Color: vec4f;

@fragment 
fn main(@location(0) vLineCenter: vec2f, @builtin(position) gl_FragCoord: vec4f) -> @location(0) vec4f {
    var col: vec4f = Color;          
    let d = length(vLineCenter - gl_FragCoord.xy);  
    let LineWidth = 1.1f;  
    col.a *= smoothstep(0.0f, 1.0f, LineWidth - d);  
    return vec4f(1.0f);
    //return vec4f(col.rgb*col.a, col.a);  
}