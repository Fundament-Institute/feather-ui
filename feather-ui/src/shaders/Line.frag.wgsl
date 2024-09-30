struct FragmentOutput {
    @location(0) fragColor: vec4<f32>,
}

@group(0) @binding(2) 
var<uniform> Color: vec4<f32>;
var<private> vLineCenter_1: vec2<f32>;
var<private> fragColor: vec4<f32>;
var<private> gl_FragCoord_1: vec4<f32>;

fn main_1() {
    var col: vec4<f32>;
    var d: f32;
    var LineWidth: f32 = 1f;

    let _e3 = Color;
    col = _e3;
    let _e6 = vLineCenter_1;
    let _e7 = gl_FragCoord_1;
    let _e10 = vLineCenter_1;
    let _e11 = gl_FragCoord_1;
    d = length((_e10 - _e11.xy));
    let _e19 = col;
    let _e23 = LineWidth;
    let _e24 = d;
    let _e28 = LineWidth;
    let _e29 = d;
    col.w = (_e19.w * smoothstep(0f, 1f, (_e28 - _e29)));
    let _e33 = col;
    let _e35 = col;
    let _e37 = (_e33.xyz * _e35.w);
    let _e38 = col;
    fragColor = vec4<f32>(_e37.x, _e37.y, _e37.z, _e38.w);
    return;
}

@fragment 
fn main(@location(0) vLineCenter: vec2<f32>, @builtin(position) gl_FragCoord: vec4<f32>) -> FragmentOutput {
    vLineCenter_1 = vLineCenter;
    gl_FragCoord_1 = gl_FragCoord;
    main_1();
    let _e11 = fragColor;
    return FragmentOutput(_e11);
}
