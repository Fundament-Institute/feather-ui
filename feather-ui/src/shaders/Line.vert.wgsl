struct VertexOutput {
    @location(0) vLineCenter: vec2<f32>,
    @builtin(position) gl_Position: vec4<f32>,
}

var<private> vPos_1: vec2<f32>;
@group(0) @binding(0) 
var<uniform> MVP: mat4x4<f32>;
@group(0) @binding(1) 
var<uniform> ViewPort: vec2<f32>;
var<private> vLineCenter: vec2<f32>;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    var pp: vec4<f32>;

    let _e4 = MVP;
    let _e5 = vPos_1;
    let _e6 = _e5.xy;
    pp = (_e4 * vec4<f32>(_e6.x, _e6.y, 0f, 1f));
    let _e17 = pp;
    gl_Position = _e17;
    let _e19 = pp;
    let _e28 = ViewPort;
    vLineCenter = ((0.5f * (_e19.xy + vec2<f32>(1f, 1f))) * _e28);
    return;
}

@vertex 
fn main(@location(0) vPos: vec2<f32>) -> VertexOutput {
    vPos_1 = vPos;
    main_1();
    let _e11 = vLineCenter;
    let _e13 = gl_Position;
    return VertexOutput(_e11, _e13);
}
