struct VertexOutput {
    @location(0) pos: vec2<f32>,
    @builtin(position) gl_Position: vec4<f32>,
}

@group(0) @binding(0) 
var<uniform> MVP: mat4x4<f32>;
@group(0) @binding(1) 
var<uniform> Inflate: vec2<f32>;
var<private> vPos_1: vec2<f32>;
var<private> pos: vec2<f32>;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    let _e4 = vPos_1;
    pos = _e4.xy;
    let _e6 = pos;
    pos = (_e6 - vec2<f32>(0.5f, 0.5f));
    let _e11 = pos;
    let _e12 = Inflate;
    pos = (_e11 * _e12);
    let _e14 = pos;
    pos = (_e14 + vec2<f32>(0.5f, 0.5f));
    let _e20 = MVP;
    let _e21 = pos;
    gl_Position = (_e20 * vec4<f32>(_e21.x, _e21.y, 0f, 1f));
    return;
}

@vertex 
fn main(@location(0) vPos: vec2<f32>) -> VertexOutput {
    vPos_1 = vPos;
    main_1();
    let _e11 = pos;
    let _e13 = gl_Position;
    return VertexOutput(_e11, _e13);
}
