struct VertexOutput {
    @location(0) pos: vec2<f32>,
    @builtin(position) gl_Position: vec4<f32>,
}

@group(0) @binding(0) 
var<uniform> MVP: mat4x4<f32>;
@group(0) @binding(1) 
var<uniform> PosDim: vec4<f32>;
var<private> vPos_1: vec2<f32>;
var<private> pos: vec2<f32>;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    var move_: mat4x4<f32>;

    let _e4 = vPos_1;
    pos = _e4.xy;
    let _e9 = PosDim;
    move_[0i] = vec4<f32>(_e9.z, 0f, 0f, 0f);
    let _e21 = PosDim;
    move_[1i] = vec4<f32>(0f, _e21.w, 0f, 0f);
    move_[2i] = vec4<f32>(0f, 0f, 1f, 0f);
    let _e42 = PosDim;
    let _e44 = PosDim;
    let _e49 = PosDim;
    let _e51 = PosDim;
    move_[3i] = vec4<f32>((_e42.x + (_e44.z * 0.5f)), (_e49.y + (_e51.w * 0.5f)), 0f, 1f);
    let _e62 = MVP;
    let _e63 = move_;
    let _e65 = pos;
    let _e69 = pos;
    gl_Position = ((_e62 * _e63) * vec4<f32>((_e65.x - 0.5f), (_e69.y - 0.5f), 1f, 1f));
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
