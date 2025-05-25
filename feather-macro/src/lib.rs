// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use core::panic;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::{Data, DataEnum, DeriveInput, Meta, parse_macro_input};

fn derive_base_prop(input: TokenStream, prop: &str, source: &str, result: &str) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let result: syn::Path = syn::parse_str(result).unwrap();
    let source: syn::Path = syn::parse_str(source).unwrap();
    let prop = format_ident!("{}", prop);
    let name = ast.ident;
    quote! {
        impl #source for #name {
            fn #prop(&self) -> &#result {
                &self.#prop
            }
        }
    }
    .into()
}

#[proc_macro_derive(Empty)]
pub fn derive_empty(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let sname = ast.ident;
    quote! {
        impl feather_ui::layout::base::Empty for #sname {}
    }
    .into()
}

#[proc_macro_derive(Area)]
pub fn derive_area(input: TokenStream) -> TokenStream {
    derive_base_prop(
        input,
        "area",
        "feather_ui::layout::base::Area",
        "feather_ui::DRect",
    )
}

#[proc_macro_derive(Padding)]
pub fn derive_padding(input: TokenStream) -> TokenStream {
    derive_base_prop(
        input,
        "padding",
        "feather_ui::layout::base::Padding",
        "feather_ui::DAbsRect",
    )
}

#[proc_macro_derive(Margin)]
pub fn derive_margin(input: TokenStream) -> TokenStream {
    derive_base_prop(
        input,
        "margin",
        "feather_ui::layout::base::Margin",
        "feather_ui::DRect",
    )
}

#[proc_macro_derive(Limits)]
pub fn derive_limits(input: TokenStream) -> TokenStream {
    derive_base_prop(
        input,
        "limits",
        "feather_ui::layout::base::Limits",
        "feather_ui::DLimits",
    )
}

#[proc_macro_derive(RLimits)]
pub fn derive_rlimits(input: TokenStream) -> TokenStream {
    derive_base_prop(
        input,
        "rlimits",
        "feather_ui::layout::base::RLimits",
        "feather_ui::RelLimits",
    )
}

#[proc_macro_derive(Anchor)]
pub fn derive_anchor(input: TokenStream) -> TokenStream {
    derive_base_prop(
        input,
        "anchor",
        "feather_ui::layout::base::Anchor",
        "feather_ui::DPoint",
    )
}

#[proc_macro_derive(TextEdit)]
pub fn derive_textedit(input: TokenStream) -> TokenStream {
    derive_base_prop(
        input,
        "textedit",
        "feather_ui::layout::base::TextEdit",
        "feather_ui::text::Snapshot",
    )
}

#[proc_macro_derive(FlexProp)]
pub fn derive_flex_prop(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let name = ast.ident;
    quote! {
        impl feather_ui::layout::flex::Prop for #name {
        fn wrap(&self) -> bool { self.wrap }
        fn justify(&self) -> feather_ui::layout::flex::FlexJustify { self.justify }
        fn align(&self) -> feather_ui::layout::flex::FlexJustify { self.align }
        }
    }
    .into()
}

#[proc_macro_derive(FlexChild)]
pub fn derive_flex_child(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let name = ast.ident;
    quote! {
        impl feather_ui::layout::flex::Child for #name {
            fn grow(&self) -> f32 { self.grow }
            fn shrink(&self) -> f32 { self.shrink }
            fn basis(&self) -> feather_ui::DValue { self.basis }
        }
    }
    .into()
}

#[proc_macro_derive(ZIndex)]
pub fn derive_zindex(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let sname = ast.ident;
    quote! {
        impl feather_ui::layout::base::ZIndex for #sname {
            fn zindex(&self) -> i32 {
                self.zindex
            }
        }
    }
    .into()
}

#[proc_macro_derive(Direction)]
pub fn derive_direction(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let sname = ast.ident;
    quote! {
        impl feather_ui::layout::base::Direction for #sname {
            fn direction(&self) -> feather_ui::RowDirection {
                self.direction
            }
        }
    }
    .into()
}

#[proc_macro_derive(RootProp)]
pub fn derive_root_prop(input: TokenStream) -> TokenStream {
    derive_base_prop(
        input,
        "dim",
        "feather_ui::layout::root::Prop",
        "feather_ui::AbsDim",
    )
}

fn data_enum(ast: &DeriveInput) -> &DataEnum {
    if let Data::Enum(data_enum) = &ast.data {
        data_enum
    } else {
        panic!("`Dispath` derive can only be used on an enum.");
    }
}

fn find_enum_module(attrs: &[syn::Attribute]) -> syn::Result<String> {
    // Extract EnumVariantType's module, since this has to be used in conjuction with our derive
    for attr in attrs.iter() {
        if attr.path().is_ident("evt") {
            let nested = attr
                .parse_args_with(
                    syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
                )
                .unwrap();

            for meta in nested {
                if let Meta::NameValue(name_value) = meta {
                    if let (true, syn::Expr::Lit(lit_str)) =
                        (name_value.path.is_ident("module"), name_value.value)
                    {
                        if let syn::Lit::Str(s) = lit_str.lit {
                            return Ok(s.value());
                        } else {
                            return Err(syn::Error::new(Span::call_site(), ""));
                        }
                    } else {
                        return Err(syn::Error::new(Span::call_site(), ""));
                    }
                }
            }

            // This would be a lot easier but it doesn't seem to work for #[evt(derive(Clone), module = "mouse_area_event")]
            /*let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("module") {
                    let value = meta.value()?;
                    let s: LitStr = value.parse()?;
                    enum_module = Some(s.value());
                }

                Ok(())
            });*/
        }
    }

    // Error here doesn't matter, we transform it into another error message upon return
    Err(syn::Error::new(Span::call_site(), ""))
}

#[proc_macro_derive(Dispatch)]
pub fn dispatchable(input: TokenStream) -> TokenStream {
    let crate_name = std::env::var("CARGO_PKG_NAME").unwrap();

    let crate_name = format_ident!(
        "{}",
        if crate_name == "feather-ui" {
            "crate"
        } else {
            "feather_ui"
        }
    );

    let ast = parse_macro_input!(input as DeriveInput);
    let enum_module = format_ident!(
        "{}",
        find_enum_module(&ast.attrs).expect(
        "Expected `evt` attribute argument in the form: `#[evt(module = \"some_module_name\")]`",
    ));

    let enum_name = &ast.ident;
    let data_enum = data_enum(&ast);
    let variants = &data_enum.variants;

    let mut extract_declarations = proc_macro2::TokenStream::new();
    let mut restore_declarations = proc_macro2::TokenStream::new();

    for (counter, variant) in variants.iter().enumerate() {
        let variant_name = &variant.ident;

        let idx = (1_u64)
            .checked_shl(counter as u32)
            .expect("Too many variants! Can't handle more than 64!");

        if variant.fields.is_empty() {
            extract_declarations.extend(quote! {
                #enum_name::#variant_name => (
                    #idx,
                    Box::new(#enum_module::#variant_name::try_from(self).unwrap()),
                ),
            });
        } else if variant.fields.iter().next().unwrap().ident.is_none() {
            let underscores = variant.fields.iter().map(|_| format_ident!("_"));
            extract_declarations.extend(quote! {
                #enum_name::#variant_name(#(#underscores),*) => (
                    #idx,
                    Box::new(#enum_module::#variant_name::try_from(self).unwrap()),
                ),
            });
        } else {
            extract_declarations.extend(quote! {
                #enum_name::#variant_name { .. } => (
                    #idx,
                    Box::new(#enum_module::#variant_name::try_from(self).unwrap()),
                ),
            });
        }

        restore_declarations.extend(quote! {
            #idx => Ok(#enum_name::from(
                *pair
                    .1
                    .downcast::<#enum_module::#variant_name>()
                    .map_err(|_| {
                        #crate_name::Error::MismatchedEnumTag(
                            pair.0,
                            std::any::TypeId::of::<#enum_module::#variant_name>(),
                            typeid,
                        )
                    })?,
            )),
        });
    }

    let counter = variants.len();
    quote! {
        impl #crate_name::Dispatchable for #enum_name {
            const SIZE: usize = #counter;

            fn extract(self) -> #crate_name::DispatchPair {
                match self {
                    #extract_declarations
                }
            }

            fn restore(pair: #crate_name::DispatchPair) -> Result<Self, #crate_name::Error> {
                let typeid = (*pair.1).type_id();
                match pair.0 {
                    #restore_declarations
                    _ => Err(#crate_name::Error::InvalidEnumTag(pair.0)),
                }
            }
        }
    }
    .into()
}
