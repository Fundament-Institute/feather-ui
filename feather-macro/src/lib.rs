// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DataEnum, DeriveInput, Meta};

/*#[proc_macro_derive(Properties)]
pub fn properties(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let struct_name = &ast.ident;

    let fields = if let syn::Data::Struct(syn::DataStruct {
        fields: syn::Fields::Named(ref fields),
        ..
    }) = ast.data
    {
        fields
    } else {
        panic!("This can only be applied to a Struct")
    };

    let mut keys = Vec::new();
    let mut idents = Vec::new();
    let mut types = Vec::new();

    for field in fields.named.iter() {
        let field_name: &syn::Ident = field.ident.as_ref().unwrap();
        let name: String = field_name.to_string();
        let literal_key_str = syn::LitStr::new(&name, field.span());
        let type_name = &field.ty;
        keys.push(quote! { #literal_key_str });
        idents.push(&field.ident);
        types.push(type_name.to_token_stream());
    }
}*/

fn data_enum(ast: &DeriveInput) -> &DataEnum {
    if let Data::Enum(data_enum) = &ast.data {
        data_enum
    } else {
        panic!("`Dispath` derive can only be used on an enum.");
    }
}

fn find_enum_module(attrs: &Vec<syn::Attribute>) -> syn::Result<String> {
    // Extract EnumVariantType's module, since this has to be used in conjuction with our derive
    for attr in attrs.iter() {
        if attr.path().is_ident("evt") {
            let nested = attr
                .parse_args_with(
                    syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
                )
                .unwrap();

            for meta in nested {
                match meta {
                    Meta::NameValue(name_value) => {
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
                    _ => (),
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
    return Err(syn::Error::new(Span::call_site(), ""));
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

    let mut counter = 0;
    let mut extract_declarations = proc_macro2::TokenStream::new();
    let mut restore_declarations = proc_macro2::TokenStream::new();

    for variant in variants.iter() {
        let variant_name = &variant.ident;

        let idx = (1 as u64)
            .checked_shl(counter)
            .expect("Too many variants! Can't handle more than 64!");
        counter += 1;

        if variant.fields.is_empty() {
            extract_declarations.extend(quote! {
                #enum_name::#variant_name => (
                    #idx,
                    Box::new(#enum_module::#variant_name::try_from(self).unwrap()),
                ),
            });
        } else {
            let underscores = variant.fields.iter().map(|_| format_ident!("_"));
            extract_declarations.extend(quote! {
                #enum_name::#variant_name(#(#underscores),*) => (
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
