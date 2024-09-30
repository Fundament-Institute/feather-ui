struct FragmentOutput {
    @location(0) fragColor: vec4<f32>,
}

const PI: f32 = 3.1415927f;

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

fn rotate(p: vec2<f32>, a: f32) -> vec2<f32> {
    var p_1: vec2<f32>;
    var a_1: f32;

    p_1 = p;
    a_1 = a;
    let _e11 = p_1;
    let _e14 = a_1;
    let _e17 = p_1;
    let _e20 = a_1;
    let _e24 = p_1;
    let _e27 = a_1;
    let _e30 = p_1;
    let _e33 = a_1;
    return vec2<f32>(((_e11.x * cos(_e14)) + (_e17.y * sin(_e20))), ((_e24.x * sin(_e27)) - (_e30.y * cos(_e33))));
}

fn sRGB(linearRGB: vec4<f32>) -> vec4<f32> {
    var linearRGB_1: vec4<f32>;

    linearRGB_1 = linearRGB;
    let _e10 = linearRGB_1;
    let _e24 = linearRGB_1;
    let _e39 = (1.055f * pow(_e24.xyz, vec3<f32>(0.36166665f, 0.36166665f, 0.36166665f)));
    let _e40 = linearRGB_1;
    return vec4<f32>(_e39.x, _e39.y, _e39.z, _e40.w);
}

fn linearRGB_2(sRGB_1: vec4<f32>) -> vec4<f32> {
    var sRGB_2: vec4<f32>;

    sRGB_2 = sRGB_1;
    let _e9 = sRGB_2;
    let _e19 = sRGB_2;
    let _e29 = pow(((_e19.xyz + vec3(0.055f)) / vec3(1.055f)), vec3(2.4f));
    let _e30 = sRGB_2;
    return vec4<f32>(_e29.x, _e29.y, _e29.z, _e30.w);
}

fn main_1() {
    var l: f32;
    var uv: vec2<f32>;
    var width: f32;
    var w1_: f32;
    var border: f32;
    var t: f32;
    var r: f32;
    var d0_: f32;
    var d1_: f32;
    var omega1_: vec2<f32>;
    var omega2_: vec2<f32>;
    var d: f32;
    var d2_: f32;
    var d3_: f32;
    var s: f32;
    var alpha: f32;

    let _e7 = DimBorderBlur;
    let _e9 = DimBorderBlur;
    l = ((_e7.x + _e9.y) * 0.5f);
    let _e15 = pos_1;
    uv = ((_e15 * 2f) - vec2(1f));
    let _e22 = pos_1;
    let _e24 = pos_1;
    let _e26 = fwidth(_e24.x);
    width = _e26;
    let _e29 = DimBorderBlur;
    let _e32 = width;
    w1_ = ((1f + _e29.w) * _e32);
    let _e35 = DimBorderBlur;
    let _e37 = l;
    border = ((_e35.z / _e37) * 2f);
    let _e43 = Corners;
    let _e45 = l;
    let _e48 = w1_;
    t = ((0.5f - (_e43.z / _e45)) + (_e48 * 1.5f));
    let _e54 = t;
    let _e56 = w1_;
    r = ((1f - _e54) + _e56);
    let _e60 = uv;
    let _e62 = r;
    let _e65 = uv;
    let _e67 = r;
    let _e70 = t;
    let _e72 = border;
    d0_ = ((abs((length(_e65) - _e67)) - _e70) + _e72);
    let _e76 = uv;
    let _e78 = r;
    let _e81 = uv;
    let _e83 = r;
    let _e86 = t;
    d1_ = (abs((length(_e81) - _e83)) - _e86);
    let _e90 = Corners;
    let _e92 = Corners;
    let _e95 = uv;
    let _e96 = Corners;
    let _e98 = Corners;
    let _e101 = rotate(_e95, (_e96.x - _e98.y));
    omega1_ = _e101;
    let _e104 = Corners;
    let _e106 = Corners;
    let _e109 = uv;
    let _e110 = Corners;
    let _e112 = Corners;
    let _e115 = rotate(_e109, (_e110.x + _e112.y));
    omega2_ = _e115;
    let _e118 = omega1_;
    let _e121 = omega1_;
    let _e125 = omega2_;
    let _e127 = omega2_;
    let _e131 = width;
    if ((abs(-(_e121.y)) + abs(_e127.y)) < _e131) {
        {
            let _e133 = Corners;
            let _e140 = width;
            d = ((((_e133.y / PI) - 0.5f) * 2f) * _e140);
        }
    } else {
        let _e142 = Corners;
        if (_e142.y > 1.5707964f) {
            {
                let _e148 = omega1_;
                let _e151 = omega2_;
                let _e153 = omega1_;
                let _e156 = omega2_;
                d = max(-(_e153.y), _e156.y);
            }
        } else {
            {
                let _e159 = omega1_;
                let _e162 = omega2_;
                let _e164 = omega1_;
                let _e167 = omega2_;
                d = min(-(_e164.y), _e167.y);
            }
        }
    }
    let _e170 = d;
    let _e171 = Corners;
    let _e176 = Corners;
    let _e186 = DimBorderBlur;
    let _e188 = width;
    let _e191 = border;
    d = (_e170 + ((((clamp((_e176.y / PI), 0f, 1f) - 0.5f) * 2f) * (_e186.w * _e188)) + _e191));
    let _e194 = d;
    let _e195 = border;
    let _e197 = w1_;
    d2_ = ((_e194 - _e195) + _e197);
    let _e201 = omega1_;
    let _e203 = Corners;
    let _e206 = d;
    let _e207 = omega1_;
    let _e209 = Corners;
    let _e213 = w1_;
    d3_ = (min(_e206, (_e207.x + _e209.y)) + _e213);
    let _e216 = w1_;
    let _e219 = d0_;
    let _e222 = d0_;
    let _e224 = d2_;
    let _e226 = w1_;
    let _e228 = w1_;
    let _e230 = w1_;
    let _e231 = d0_;
    let _e234 = d0_;
    let _e236 = d2_;
    let _e238 = w1_;
    let _e240 = linearstep(-(_e228), _e230, (min(-(_e234), _e236) - _e238));
    s = _e240;
    let _e242 = w1_;
    let _e245 = d1_;
    let _e248 = d1_;
    let _e250 = d3_;
    let _e252 = w1_;
    let _e254 = w1_;
    let _e256 = w1_;
    let _e257 = d1_;
    let _e260 = d1_;
    let _e262 = d3_;
    let _e264 = w1_;
    let _e266 = linearstep(-(_e254), _e256, (min(-(_e260), _e262) - _e264));
    alpha = _e266;
    let _e268 = Fill;
    let _e269 = _e268.xyz;
    let _e276 = Fill;
    let _e279 = s;
    let _e281 = Outline;
    let _e282 = _e281.xyz;
    let _e289 = Outline;
    let _e292 = alpha;
    let _e293 = s;
    let _e297 = alpha;
    let _e298 = s;
    fragColor = (((vec4<f32>(_e269.x, _e269.y, _e269.z, 1f) * _e276.w) * _e279) + ((vec4<f32>(_e282.x, _e282.y, _e282.z, 1f) * _e289.w) * clamp((_e297 - _e298), 0f, 1f)));
    return;
}

@fragment 
fn main(@location(0) pos: vec2<f32>) -> FragmentOutput {
    pos_1 = pos;
    main_1();
    let _e17 = fragColor;
    return FragmentOutput(_e17);
}
