struct FragmentOutput {
    @location(0) fragColor: vec4<f32>,
}

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

fn linetopoint(p1_: vec2<f32>, p2_: vec2<f32>, p: vec2<f32>) -> f32 {
    var p1_1: vec2<f32>;
    var p2_1: vec2<f32>;
    var p_1: vec2<f32>;
    var n: vec2<f32>;

    p1_1 = p1_;
    p2_1 = p2_;
    p_1 = p;
    let _e13 = p2_1;
    let _e14 = p1_1;
    n = (_e13 - _e14);
    let _e17 = n;
    let _e19 = n;
    n = vec2<f32>(_e17.y, -(_e19.x));
    let _e24 = n;
    let _e26 = p1_1;
    let _e27 = p_1;
    let _e30 = n;
    let _e32 = p1_1;
    let _e33 = p_1;
    return dot(normalize(_e30), (_e32 - _e33));
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

    let _e7 = PosDim;
    d = _e7.zw;
    let _e10 = pos_1;
    let _e11 = d;
    p_2 = ((_e10 * _e11) + vec2<f32>(-0.5f, 0.5f));
    let _e19 = Corners;
    c = _e19;
    let _e22 = c;
    let _e24 = d;
    p2_2 = vec2<f32>((_e22.w * _e24.x), 0f);
    let _e32 = d;
    let _e36 = p2_2;
    let _e38 = d;
    let _e41 = p_2;
    let _e42 = linetopoint(_e36, vec2<f32>(0f, _e38.y), _e41);
    r1_ = _e42;
    let _e47 = p2_2;
    let _e48 = d;
    let _e49 = p_2;
    let _e50 = linetopoint(_e47, _e48, _e49);
    r2_ = -(_e50);
    let _e55 = r1_;
    let _e56 = r2_;
    r = max(_e55, _e56);
    let _e60 = p_2;
    let _e62 = d;
    let _e65 = r;
    let _e66 = p_2;
    let _e68 = d;
    r = max(_e65, (_e66.y - _e68.y));
    let _e72 = p_2;
    let _e74 = p_2;
    let _e76 = fwidth(_e74.x);
    let _e78 = DimBorderBlur;
    w = (_e76 * (1f + _e78.w));
    let _e85 = DimBorderBlur;
    let _e88 = w;
    let _e93 = DimBorderBlur;
    let _e96 = w;
    let _e100 = DimBorderBlur;
    let _e103 = w;
    let _e108 = DimBorderBlur;
    let _e111 = w;
    let _e113 = r;
    let _e114 = linearstep(((1f - _e100.z) - (_e103 * 2f)), ((1f - _e108.z) - _e111), _e113);
    s = (1f - _e114);
    let _e118 = w;
    let _e121 = w;
    let _e127 = w;
    let _e130 = w;
    let _e134 = r;
    let _e135 = linearstep((1f - _e127), (1f - (_e130 * 2f)), _e134);
    alpha = _e135;
    let _e137 = Fill;
    let _e138 = _e137.xyz;
    let _e144 = Fill;
    let _e147 = s;
    let _e149 = Outline;
    let _e150 = _e149.xyz;
    let _e156 = Outline;
    let _e159 = alpha;
    let _e160 = s;
    let _e164 = alpha;
    let _e165 = s;
    fragColor = (((vec4<f32>(_e138.x, _e138.y, _e138.z, 1f) * _e144.w) * _e147) + ((vec4<f32>(_e150.x, _e150.y, _e150.z, 1f) * _e156.w) * clamp((_e164 - _e165), 0f, 1f)));
    return;
}

@fragment 
fn main(@location(0) pos: vec2<f32>) -> FragmentOutput {
    pos_1 = pos;
    main_1();
    let _e17 = fragColor;
    return FragmentOutput(_e17);
}
