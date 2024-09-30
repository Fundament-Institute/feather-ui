struct FragmentOutput {
    @location(0) fragColor: vec4<f32>,
}

var<private> pos_1: vec2<f32>;
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
    let _e12 = x_1;
    let _e13 = low_1;
    let _e15 = high_1;
    let _e16 = low_1;
    let _e21 = x_1;
    let _e22 = low_1;
    let _e24 = high_1;
    let _e25 = low_1;
    return clamp(((_e21 - _e22) / (_e24 - _e25)), 0f, 1f);
}

fn linetopoint(p1_: vec2<f32>, p2_: vec2<f32>, p: vec2<f32>) -> f32 {
    var p1_1: vec2<f32>;
    var p2_1: vec2<f32>;
    var p_1: vec2<f32>;
    var n: vec2<f32>;

    p1_1 = p1_;
    p2_1 = p2_;
    p_1 = p;
    let _e12 = p2_1;
    let _e13 = p1_1;
    n = (_e12 - _e13);
    let _e16 = n;
    let _e18 = n;
    n = vec2<f32>(_e16.y, -(_e18.x));
    let _e23 = n;
    let _e25 = p1_1;
    let _e26 = p_1;
    let _e29 = n;
    let _e31 = p1_1;
    let _e32 = p_1;
    return dot(normalize(_e29), (_e31 - _e32));
}

fn main_1() {
    var d: vec2<f32>;
    var p_2: vec2<f32>;
    var c: vec4<f32>;
    var dist: vec2<f32>;
    var p2_2: vec2<f32>;
    var r1_: f32;
    var r2_: f32;
    var r: f32;
    var w: f32;
    var s: f32;
    var alpha: f32;

    let _e6 = DimBorderBlur;
    d = _e6.xy;
    let _e9 = pos_1;
    let _e10 = d;
    p_2 = ((_e9 * _e10) + vec2<f32>(-0.5f, 0.5f));
    let _e18 = Corners;
    c = _e18;
    let _e21 = c;
    let _e23 = d;
    p2_2 = vec2<f32>((_e21.w * _e23.x), 0f);
    let _e31 = d;
    let _e35 = p2_2;
    let _e37 = d;
    let _e40 = p_2;
    let _e41 = linetopoint(_e35, vec2<f32>(0f, _e37.y), _e40);
    r1_ = _e41;
    let _e46 = p2_2;
    let _e47 = d;
    let _e48 = p_2;
    let _e49 = linetopoint(_e46, _e47, _e48);
    r2_ = -(_e49);
    let _e54 = r1_;
    let _e55 = r2_;
    r = max(_e54, _e55);
    let _e59 = p_2;
    let _e61 = d;
    let _e64 = r;
    let _e65 = p_2;
    let _e67 = d;
    r = max(_e64, (_e65.y - _e67.y));
    let _e71 = p_2;
    let _e73 = p_2;
    let _e75 = fwidth(_e73.x);
    let _e77 = DimBorderBlur;
    w = (_e75 * (1f + _e77.w));
    let _e84 = DimBorderBlur;
    let _e87 = w;
    let _e92 = DimBorderBlur;
    let _e95 = w;
    let _e99 = DimBorderBlur;
    let _e102 = w;
    let _e107 = DimBorderBlur;
    let _e110 = w;
    let _e112 = r;
    let _e113 = linearstep(((1f - _e99.z) - (_e102 * 2f)), ((1f - _e107.z) - _e110), _e112);
    s = (1f - _e113);
    let _e117 = w;
    let _e120 = w;
    let _e126 = w;
    let _e129 = w;
    let _e133 = r;
    let _e134 = linearstep((1f - _e126), (1f - (_e129 * 2f)), _e133);
    alpha = _e134;
    let _e136 = Fill;
    let _e137 = _e136.xyz;
    let _e143 = Fill;
    let _e146 = s;
    let _e148 = Outline;
    let _e149 = _e148.xyz;
    let _e155 = Outline;
    let _e158 = alpha;
    let _e159 = s;
    let _e163 = alpha;
    let _e164 = s;
    fragColor = (((vec4<f32>(_e137.x, _e137.y, _e137.z, 1f) * _e143.w) * _e146) + ((vec4<f32>(_e149.x, _e149.y, _e149.z, 1f) * _e155.w) * clamp((_e163 - _e164), 0f, 1f)));
    return;
}

@fragment 
fn main(@location(0) pos: vec2<f32>) -> FragmentOutput {
    pos_1 = pos;
    main_1();
    let _e15 = fragColor;
    return FragmentOutput(_e15);
}
