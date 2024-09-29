struct FragmentOutput {
    @location(0) fragColor: vec4<f32>,
}

const PI: f32 = 3.1415927f;

var<private> pos_1: vec2<f32>;
@group(0) @binding(0) 
var<uniform> DimBorderBlur: vec4<f32>;
@group(0) @binding(1) 
var<uniform> Corners: vec4<f32>;
@group(0) @binding(2) 
var<uniform> Fill: vec4<f32>;
@group(0) @binding(3) 
var<uniform> Outline: vec4<f32>;
var<private> fragColor: vec4<f32>;

fn linearstep(low: f32, high: f32, x: f32) -> f32 {
    var low_1: f32;
    var high_1: f32;
    var x_1: f32;

    low_1 = low;
    high_1 = high;
    x_1 = x;
    let _e13 = x_1;
    let _e14 = low_1;
    let _e16 = high_1;
    let _e17 = low_1;
    let _e22 = x_1;
    let _e23 = low_1;
    let _e25 = high_1;
    let _e26 = low_1;
    return clamp(((_e22 - _e23) / (_e25 - _e26)), 0f, 1f);
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

    let _e7 = DimBorderBlur;
    let _e9 = DimBorderBlur;
    l = ((_e7.x + _e9.y) * 0.5f);
    let _e15 = pos_1;
    uv = ((_e15 * 2f) - vec2(1f));
    let _e23 = DimBorderBlur;
    let _e26 = pos_1;
    let _e28 = pos_1;
    let _e30 = fwidth(_e28.x);
    w1_ = ((1f + _e23.w) * _e30);
    let _e33 = DimBorderBlur;
    let _e35 = l;
    border = ((_e33.z / _e35) * 2f);
    let _e41 = Corners;
    let _e43 = l;
    t = (0.5f - (_e41.x / _e43));
    let _e48 = t;
    let _e50 = w1_;
    r = ((1f - _e48) - _e50);
    let _e53 = Corners;
    let _e55 = l;
    inner = ((_e53.y / _e55) * 2f);
    let _e61 = uv;
    let _e63 = r;
    let _e65 = border;
    let _e69 = inner;
    let _e74 = uv;
    let _e76 = r;
    let _e78 = border;
    let _e82 = inner;
    let _e87 = t;
    let _e89 = border;
    let _e93 = inner;
    d0_ = (((abs((((length(_e74) - _e76) + (_e78 * 0.5f)) - (_e82 * 0.5f))) - _e87) + (_e89 * 0.5f)) + (_e93 * 0.5f));
    let _e99 = uv;
    let _e101 = r;
    let _e104 = uv;
    let _e106 = r;
    let _e109 = t;
    d1_ = (abs((length(_e104) - _e106)) - _e109);
    let _e112 = w1_;
    let _e117 = w1_;
    let _e121 = d0_;
    let _e122 = linearstep((_e117 * 2f), 0f, _e121);
    let _e124 = w1_;
    let _e129 = w1_;
    let _e133 = d0_;
    let _e134 = linearstep((_e129 * 2f), 0f, _e133);
    s = pow(_e134, 2.2f);
    let _e138 = w1_;
    let _e143 = w1_;
    let _e147 = d1_;
    let _e148 = linearstep((_e143 * 2f), 0f, _e147);
    let _e150 = w1_;
    let _e155 = w1_;
    let _e159 = d1_;
    let _e160 = linearstep((_e155 * 2f), 0f, _e159);
    alpha = pow(_e160, 2.2f);
    let _e164 = Fill;
    let _e165 = _e164.xyz;
    let _e172 = Fill;
    let _e175 = s;
    let _e177 = Outline;
    let _e178 = _e177.xyz;
    let _e185 = Outline;
    let _e188 = alpha;
    let _e189 = s;
    let _e193 = alpha;
    let _e194 = s;
    fragColor = (((vec4<f32>(_e165.x, _e165.y, _e165.z, 1f) * _e172.w) * _e175) + ((vec4<f32>(_e178.x, _e178.y, _e178.z, 1f) * _e185.w) * clamp((_e193 - _e194), 0f, 1f)));
    return;
}

@fragment 
fn main(@location(0) pos: vec2<f32>) -> FragmentOutput {
    pos_1 = pos;
    main_1();
    let _e17 = fragColor;
    return FragmentOutput(_e17);
}
