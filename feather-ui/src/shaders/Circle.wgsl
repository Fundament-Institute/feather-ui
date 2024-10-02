struct FragmentOutput {
    @location(0) fragColor: vec4<f32>,
}

const PI: f32 = 3.1415927f;

var<private> pos_1: vec2<f32>;
@group(0) @binding(1) 
var<uniform> PosDim: vec4<f32>;
@group(0) @binding(2) 
var<uniform> DimBorderBlur: vec4<f32>;
@group(0) @binding(3) 
var<uniform> Corners: vec4<f32>;
@group(0) @binding(4) 
var<uniform> Fill: vec4<f32>;
@group(0) @binding(5) 
var<uniform> Outline: vec4<f32>;
var<private> fragColor: vec4<f32>;

fn linearstep(low: f32, high: f32, x: f32) -> f32 {
    var low_1: f32;
    var high_1: f32;
    var x_1: f32;

    low_1 = low;
    high_1 = high;
    x_1 = x;
    let _e14 = x_1;
    let _e15 = low_1;
    let _e17 = high_1;
    let _e18 = low_1;
    let _e23 = x_1;
    let _e24 = low_1;
    let _e26 = high_1;
    let _e27 = low_1;
    return clamp(((_e23 - _e24) / (_e26 - _e27)), 0f, 1f);
}

fn main_1() {
    var l: f32;
    var uv: vec2<f32>;
    var w1_: f32;
    var border: f32;
    var t: f32;
    var r: f32;
    var inner: f32;
    var d0_: f32;
    var d1_: f32;
    var s: f32;
    var alpha: f32;

    let _e8 = PosDim;
    let _e10 = PosDim;
    l = ((_e8.z + _e10.w) * 0.5f);
    let _e16 = pos_1;
    uv = ((_e16 * 2f) - vec2(1f));
    let _e24 = DimBorderBlur;
    let _e27 = pos_1;
    let _e29 = pos_1;
    let _e31 = fwidth(_e29.x);
    w1_ = ((1f + _e24.w) * _e31);
    let _e34 = DimBorderBlur;
    let _e36 = l;
    border = ((_e34.z / _e36) * 2f);
    let _e42 = Corners;
    let _e44 = l;
    t = (0.5f - (_e42.x / _e44));
    let _e49 = t;
    let _e51 = w1_;
    r = ((1f - _e49) - _e51);
    let _e54 = Corners;
    let _e56 = l;
    inner = ((_e54.y / _e56) * 2f);
    let _e62 = uv;
    let _e64 = r;
    let _e66 = border;
    let _e70 = inner;
    let _e75 = uv;
    let _e77 = r;
    let _e79 = border;
    let _e83 = inner;
    let _e88 = t;
    let _e90 = border;
    let _e94 = inner;
    d0_ = (((abs((((length(_e75) - _e77) + (_e79 * 0.5f)) - (_e83 * 0.5f))) - _e88) + (_e90 * 0.5f)) + (_e94 * 0.5f));
    let _e100 = uv;
    let _e102 = r;
    let _e105 = uv;
    let _e107 = r;
    let _e110 = t;
    d1_ = (abs((length(_e105) - _e107)) - _e110);
    let _e113 = w1_;
    let _e118 = w1_;
    let _e122 = d0_;
    let _e123 = linearstep((_e118 * 2f), 0f, _e122);
    let _e125 = w1_;
    let _e130 = w1_;
    let _e134 = d0_;
    let _e135 = linearstep((_e130 * 2f), 0f, _e134);
    s = pow(_e135, 2.2f);
    let _e139 = w1_;
    let _e144 = w1_;
    let _e148 = d1_;
    let _e149 = linearstep((_e144 * 2f), 0f, _e148);
    let _e151 = w1_;
    let _e156 = w1_;
    let _e160 = d1_;
    let _e161 = linearstep((_e156 * 2f), 0f, _e160);
    alpha = pow(_e161, 2.2f);
    let _e165 = Fill;
    let _e166 = _e165.xyz;
    let _e173 = Fill;
    let _e176 = s;
    let _e178 = Outline;
    let _e179 = _e178.xyz;
    let _e186 = Outline;
    let _e189 = alpha;
    let _e190 = s;
    let _e194 = alpha;
    let _e195 = s;
    fragColor = (((vec4<f32>(_e166.x, _e166.y, _e166.z, 1f) * _e173.w) * _e176) + ((vec4<f32>(_e179.x, _e179.y, _e179.z, 1f) * _e186.w) * clamp((_e194 - _e195), 0f, 1f)));
    return;
}

@fragment 
fn main(@location(0) pos: vec2<f32>) -> FragmentOutput {
    pos_1 = pos;
    main_1();
    let _e19 = fragColor;
    return FragmentOutput(_e19);
}
