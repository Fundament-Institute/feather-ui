// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use num_traits::NumCast;
use wide::f32x4;

macro_rules! gen_channel_accessors {
    ($rgba:ident, $r:ident, $g:ident, $b:ident, $a:ident) => {
        pub fn as_array(&self) -> &[f32; 4] {
            self.$rgba.as_array_ref()
        }

        pub fn new($r: f32, $g: f32, $b: f32, $a: f32) -> Self {
            Self {
                $rgba: f32x4::new([$r, $g, $b, $a]),
            }
        }

        pub fn $r(&self) -> f32 {
            self.as_array()[0]
        }

        pub fn $g(&self) -> f32 {
            self.as_array()[1]
        }

        pub fn $b(&self) -> f32 {
            self.as_array()[2]
        }

        pub fn $a(&self) -> f32 {
            self.as_array()[3]
        }
    };
}

pub fn mat4_x_vec4(l: f32x4, r: [f32x4; 4]) -> f32x4 {
    let v = l.as_array_ref();
    ((r[0]) * f32x4::splat(v[0]))
        + ((r[1]) * f32x4::splat(v[1]))
        + ((r[2]) * f32x4::splat(v[2]))
        + ((r[3]) * f32x4::splat(v[3]))
}

fn fconv<T: num_traits::ToPrimitive, U: NumCast>(v: T) -> U {
    U::from(v).unwrap()
}

pub fn srgb_to_linear<T: num_traits::Float>(c: T) -> T {
    if c <= fconv(0.04045) {
        return c / fconv(12.92);
    } else {
        return ((c + fconv(0.055)) / fconv(1.055)).powf(fconv(2.4));
    }
}

pub fn linear_to_srgb<T: num_traits::Float>(c: T) -> T {
    if c < fconv(0.0031308) {
        return c * fconv(12.92);
    } else {
        return (c.powf(<T as NumCast>::from(1.0).unwrap() / fconv(2.4)) * fconv(1.055))
            - fconv(0.055);
    }
}

/// Applies a function to all color channels, leaving the alpha channel untouched
pub fn map_color(c: f32x4, f: impl Fn(f32) -> f32) -> f32x4 {
    let v = c.to_array();
    f32x4::new([f(v[0]), f(v[1]), f(v[2]), v[3]])
}

pub trait ColorSpace: Premultiplied {
    /// Everything must provide a way to translate into standard XYZ space. The 4th component
    /// is alpha, which is left unchanged in all color space transforms
    fn xyz(&self) -> XYZ;

    /// This uses the standard formulation to transform into OkLab from XYZ
    /// but this can be overriden if there's a more efficient pathway
    fn oklab(&self) -> OkLab {
        let xyz = self.xyz();

        let mut lms = mat4_x_vec4(xyz.xyza, XYZ_OKLAB_M1);

        let v = lms.as_array_mut();
        for i in 0..2 {
            v[i] = v[i].powf(1.0 / 3.0);
        }

        OkLab {
            laba: mat4_x_vec4(lms, XYZ_OKLAB_M2),
        }
    }

    /// This pathway goes through the linear Raw_sRGB pathway first, because only linear Raw_sRGB to XYZ
    /// and vice-versa is defined, but a color space can override this with a faster pathway.
    fn srgb(&self) -> Raw_sRGB<false, false> {
        let srgb = self.linear_srgb();

        Raw_sRGB {
            rgba: map_color(srgb.rgba, linear_to_srgb),
        }
    }

    /// This automatically converts from XYZ, but can be overriden if there's a faster pathway
    fn linear_srgb(&self) -> Raw_sRGB<true, false> {
        let xyz = self.xyz();

        let linear_rgba = mat4_x_vec4(xyz.xyza, XYZ_SRGB);
        Raw_sRGB { rgba: linear_rgba }
    }
}

/// Represents a color space that is premultiplied. You cannot un-premultiply a colorspace, so
/// this is a one-way transformation.
pub trait Premultiplied {
    /// Transforms this colorspace into premultiplied non-linear sRGB. This should always
    /// linearize the colorspace before performing the premultiplication, which will then be
    /// delinearized back into sRGB after the premultiplication has happened.
    fn srgb_pre(&self) -> Raw_sRGB<false, true> {
        let srgb = self.linear_srgb_pre();

        Raw_sRGB {
            rgba: map_color(srgb.rgba, linear_to_srgb),
        }
    }

    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true>;
}

#[derive(Default, Clone, Copy, PartialEq)]
pub struct XYZ {
    xyza: f32x4,
}

impl XYZ {
    gen_channel_accessors! {xyza, x, y, z, a}
}

impl ColorSpace for XYZ {
    fn xyz(&self) -> XYZ {
        *self
    }
}

impl Premultiplied for XYZ {
    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        self.linear_srgb().premultiply()
    }
}

const XYZ_OKLAB_M1: [f32x4; 4] = [
    f32x4::new([0.8189330101, 0.3618667424, -0.1288597137, 0.0]),
    f32x4::new([0.0329845436, 0.9293118715, 0.0361456387, 0.0]),
    f32x4::new([0.0482003018, 0.2643662691, 0.6338517070, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];

const OKLAB_XYZ_M1: [f32x4; 4] = [
    f32x4::new([1.22701, -0.5578, 0.281256, 0.0]),
    f32x4::new([-0.0405802, 1.11226, -0.0716767, 0.0]),
    f32x4::new([-0.0763813, -0.421482, 1.58616, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];

const XYZ_OKLAB_M2: [f32x4; 4] = [
    f32x4::new([0.2104542553, 0.7936177850, -0.0040720468, 0.0]),
    f32x4::new([1.9779984951, -2.4285922050, 0.4505937099, 0.0]),
    f32x4::new([0.025904037, 0.7827717662, -0.808675766, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];

const OKLAB_XYZ_M2: [f32x4; 4] = [
    f32x4::new([1.0, 0.396338, 0.215804, 0.0]),
    f32x4::new([1.0, -0.105561, -0.0638542, 0.0]),
    f32x4::new([1.0, -0.0894842, -1.29149, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];

#[derive(Default, Clone, Copy, PartialEq)]
pub struct OkLab {
    laba: f32x4,
}

impl OkLab {
    gen_channel_accessors! {laba, l, c, h, a}
}

impl ColorSpace for OkLab {
    fn xyz(&self) -> XYZ {
        let mut lms = mat4_x_vec4(self.laba, OKLAB_XYZ_M2);

        let v = lms.as_array_mut();
        for i in 0..2 {
            v[i] = v[i].powf(3.0);
        }

        XYZ {
            xyza: mat4_x_vec4(lms, OKLAB_XYZ_M1),
        }
    }

    fn oklab(&self) -> OkLab {
        *self
    }

    // Based on the somewhat cursed code provided here: https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
    fn linear_srgb(&self) -> Raw_sRGB<true, false> {
        let v = self.laba.as_array_ref();
        let l_ = v[0] + 0.3963377774 * v[1] + 0.2158037573 * v[2];
        let m_ = v[0] - 0.1055613458 * v[1] - 0.0638541728 * v[2];
        let s_ = v[0] - 0.0894841775 * v[1] - 1.2914855480 * v[2];

        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;

        Raw_sRGB {
            rgba: f32x4::new([
                4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
                v[3],
            ]),
        }
    }
}

impl Premultiplied for OkLab {
    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        self.linear_srgb().premultiply()
    }
}

//  TODO: These use reference white D65, might need to use the one for D50???
const XYZ_SRGB: [f32x4; 4] = [
    f32x4::new([3.2404542, -1.5371385, -0.4985314, 0.0]),
    f32x4::new([-0.9692660, 1.8760108, 0.0415560, 0.0]),
    f32x4::new([0.0556434, -0.2040259, 1.0572252, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];
const SRGB_XYZ: [f32x4; 4] = [
    f32x4::new([0.4124564, 0.3575761, 0.1804375, 0.0]),
    f32x4::new([0.2126729, 0.7151522, 0.0721750, 0.0]),
    f32x4::new([0.0193339, 0.1191920, 0.9503041, 0.0]),
    f32x4::new([0.0, 0.0, 0.0, 1.0]),
];

#[derive(Default, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub struct Raw_sRGB<const LINEAR: bool, const PREMULTIPLY: bool> {
    pub rgba: f32x4,
}

impl<const LINEAR: bool, const PREMULTIPLY: bool> Raw_sRGB<LINEAR, PREMULTIPLY> {
    gen_channel_accessors! {rgba, r, g, b, a}

    // Conveniently, white and black are the same in EVERY sRGB variant.

    /// Returns transparent black (all zeroes)
    pub fn transparent() -> Self {
        Self { rgba: f32x4::ZERO }
    }

    /// Returns opaque black
    pub fn black() -> Self {
        Self {
            rgba: f32x4::new([0.0, 0.0, 0.0, 1.0]),
        }
    }

    /// Returns pure white (all ones)
    pub fn white() -> Self {
        Self { rgba: f32x4::ONE }
    }
}

impl Raw_sRGB<false, false> {
    /// Returns this color as an array of 8-bit channels. This is only implemented
    /// on nonlinear sRGB that hasn't been premultiplied, because otherwise you'll
    /// lose precision when sending the color to a shader, unless you are writing
    /// directly to the texture atlas, in which case you want `as_bgra` implemented
    /// on premultiplied sRGB.
    pub fn as_8bit(&self) -> [u8; 4] {
        self.as_array().map(|x| (x * 255.0).round() as u8)
    }

    /// Returns this color in a 32-bit integer. This is only valid for nonlinear sRGB
    /// that isn't premultiplied to avoid precision loss.
    pub fn as_32bit(&self) -> sRGB32 {
        sRGB32 {
            rgba: u32::from_be_bytes(self.as_8bit()),
        }
    }
}

impl Raw_sRGB<false, true> {
    /// Returns this color as an array of 8-bit channels, with the red and blue channels
    /// swapped. This is used exclusively to write values directly to a premultiplied
    /// sRGB texture like the texture atlas.
    pub fn as_bgra(&self) -> [u8; 4] {
        let mut rgba = self.as_array().map(|x| (x * 255.0).round() as u8);
        rgba.swap(0, 2);
        rgba
    }
}

impl<const LINEAR: bool, const PREMULTIPLY: bool> From<[f32; 4]> for Raw_sRGB<LINEAR, PREMULTIPLY> {
    fn from(value: [f32; 4]) -> Self {
        Self {
            rgba: f32x4::new(value),
        }
    }
}

impl ColorSpace for Raw_sRGB<false, false> {
    fn xyz(&self) -> XYZ {
        self.linear_srgb().xyz()
    }

    fn srgb(&self) -> Raw_sRGB<false, false> {
        *self
    }

    fn linear_srgb(&self) -> Raw_sRGB<true, false> {
        Raw_sRGB {
            rgba: map_color(self.rgba, srgb_to_linear),
        }
    }
}

impl Premultiplied for Raw_sRGB<false, false> {
    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        self.linear_srgb().premultiply()
    }
}

impl ColorSpace for Raw_sRGB<true, false> {
    fn xyz(&self) -> XYZ {
        // We can't really un-premultiply the alpha here, and XYZ can be premultiplied, so we just leave it
        XYZ {
            xyza: mat4_x_vec4(self.rgba, SRGB_XYZ),
        }
    }

    fn oklab(&self) -> OkLab {
        let v = self.as_array();
        let l = 0.4122214708 * v[0] + 0.5363325363 * v[1] + 0.0514459929 * v[2];
        let m = 0.2119034982 * v[0] + 0.6806995451 * v[1] + 0.1073969566 * v[2];
        let s = 0.0883024619 * v[0] + 0.2817188376 * v[1] + 0.6299787005 * v[2];

        let l_ = l.powf(1.0 / 3.0);
        let m_ = m.powf(1.0 / 3.0);
        let s_ = s.powf(1.0 / 3.0);

        OkLab {
            laba: f32x4::new([
                0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
                v[3],
            ]),
        }
    }

    fn linear_srgb(&self) -> Raw_sRGB<true, false> {
        *self
    }
}

impl Premultiplied for Raw_sRGB<true, false> {
    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        self.premultiply()
    }
}

impl Raw_sRGB<true, false> {
    // Premultiplies the alpha
    fn premultiply(&self) -> Raw_sRGB<true, true> {
        let a = self.a();
        Raw_sRGB {
            rgba: map_color(self.rgba, |x| x * a),
        }
    }
}

impl Premultiplied for Raw_sRGB<false, true> {
    fn srgb_pre(&self) -> Raw_sRGB<false, true> {
        *self
    }

    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        Raw_sRGB {
            rgba: map_color(self.rgba, srgb_to_linear),
        }
    }
}

impl Premultiplied for Raw_sRGB<true, true> {
    fn linear_srgb_pre(&self) -> Raw_sRGB<true, true> {
        *self
    }
}

/// Standard sRGB colorspace
#[allow(non_camel_case_types)]
pub type sRGB = Raw_sRGB<false, false>;
/// Linear sRGB colorspace
#[allow(non_camel_case_types)]
pub type Linear_sRGB = Raw_sRGB<true, false>;
/// Premultiplied sRGB colorspace
#[allow(non_camel_case_types)]
pub type Pre_sRGB = Raw_sRGB<false, true>;
/// Premultiplied Linear sRGB colorspace
#[allow(non_camel_case_types)]
pub type PreLinear_sRGB = Raw_sRGB<true, true>;

// We only implement this conversion for sRGB because cosmic_text expects sRGB colors.
impl Into<cosmic_text::Color> for sRGB {
    fn into(self) -> cosmic_text::Color {
        let v = self.as_8bit();
        cosmic_text::Color::rgba(v[0], v[1], v[2], v[3])
    }
}

/// Represents an sRGB color (not premultiplied) as a 32-bit signed integer
#[allow(non_camel_case_types)]
pub struct sRGB32 {
    pub rgba: u32,
}

impl sRGB32 {
    pub fn as_array(&self) -> [u8; 4] {
        self.rgba.to_be_bytes()
    }

    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            rgba: u32::from_be_bytes([r, g, b, a]),
        }
    }

    pub fn r(&self) -> u8 {
        self.as_array()[0]
    }

    pub fn g(&self) -> u8 {
        self.as_array()[1]
    }

    pub fn b(&self) -> u8 {
        self.as_array()[2]
    }

    pub fn a(&self) -> u8 {
        self.as_array()[3]
    }

    pub fn as_f32(&self) -> sRGB {
        sRGB {
            rgba: f32x4::new(self.as_array().map(|x| x as f32 / 255.0)),
        }
    }
}
