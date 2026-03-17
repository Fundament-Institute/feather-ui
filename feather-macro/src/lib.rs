// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

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
            fn #prop(&self) -> DynSignal<#result> {
                self.#prop.clone().into()
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
        "feather_ui::UPerimeter",
    )
}

#[proc_macro_derive(Margin)]
pub fn derive_margin(input: TokenStream) -> TokenStream {
    derive_base_prop(
        input,
        "margin",
        "feather_ui::layout::base::Margin",
        "feather_ui::DPerimeter",
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
        "feather_ui::text::EditView",
    )
}

#[proc_macro_derive(ZIndex)]
pub fn derive_zindex(input: TokenStream) -> TokenStream {
    derive_base_prop(input, "zindex", "feather_ui::layout::base::ZIndex", "i32")
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

#[proc_macro_derive(Direction)]
pub fn derive_direction(input: TokenStream) -> TokenStream {
    derive_base_prop(
        input,
        "direction",
        "feather_ui::layout::base::Direction",
        "feather_ui::RowDirection",
    )
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
        panic!("`Dispatch` derive can only be used on an enum.");
    }
}

fn find_enum_module(attrs: &[syn::Attribute]) -> syn::Result<String> {
    // Extract EnumVariantType's module, since this has to be used in conjunction
    // with our derive
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

            // This would be a lot easier but it doesn't seem to work for
            // #[evt(derive(Clone), module = "mouse_area_event")]
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

    // Error here doesn't matter, we transform it into another error message upon
    // return
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

    let mut send_declarations = proc_macro2::TokenStream::new();
    let mut callback_declarations = proc_macro2::TokenStream::new();
    let mut callback_none = proc_macro2::TokenStream::new();
    let mut struct_declarations = proc_macro2::TokenStream::new();
    let mut impl_declarations = proc_macro2::TokenStream::new();
    let mut prism_declarations = proc_macro2::TokenStream::new();

    for (counter, variant) in variants.iter().enumerate() {
        let variant_name = &variant.ident;
        let idx = syn::Index::from(counter);

        struct_declarations.extend(quote! {
            pub #variant_name: #crate_name::event::PrismStream<'a, #enum_name, #enum_module::#variant_name, S>,
        });

        prism_declarations.extend(quote! {
            #variant_name: #crate_name::event::PrismStream::<'a, #enum_name, #enum_module::#variant_name, S>::new(&state),
        });

        callback_declarations.extend(quote! {
            Option<#crate_name::event::BoxedCallback<#enum_module::#variant_name>>,
        });

        callback_none.extend(quote! { None, });

        let prevariant = if variant.fields.is_empty() {
            quote! {
                Self::#variant_name
            }
        } else if variant.fields.iter().next().unwrap().ident.is_none() {
            quote! {
                Self::#variant_name(..)
            }
        } else {
            quote! {
                Self::#variant_name { .. }
            }
        };

        send_declarations.extend(quote! {
            v @ #prevariant => {
                if let Some(h) = &mut callback.#idx
                    && let Ok(x) = #enum_module::#variant_name::try_from(v)
                {
                    h.0.send(x)
                } else {
                    #crate_name::event::FORWARD
                }
            }
        });

        impl_declarations.extend(quote! {
            impl #crate_name::DispatchCallback<#enum_name> for #enum_module::#variant_name {
                fn subscribe<H: #crate_name::event::StreamCallback<Self> + 'static>(
                    h: H,
                    callback: &mut <#enum_name as #crate_name::Dispatchable>::Callback,
                ) {
                    assert!(matches!(callback.#idx.take(), None));
                    let _ = callback.#idx.insert(#crate_name::event::BoxedCallback::new(h));
                }

                fn unsubscribe<H: #crate_name::event::StreamCallback<Self> + 'static>(
                    callback: &mut <#enum_name as #crate_name::Dispatchable>::Callback,
                ) -> H {
                    let h = callback
                        .#idx
                        .take()
                        .expect("Tried to unsubscribe from empty callback!");
                    h.unbox()
                }
            }
        });
    }

    let prismident = format_ident!("{}Prism", enum_name);
    quote! {
        #[allow(non_snake_case)]
        pub struct #prismident<'a, S: #crate_name::event::EventStream<'a, #enum_name>> {
            #struct_declarations
        }

        impl #crate_name::Dispatchable for #enum_name {
            type Prism<'a, S: 'a + #crate_name::event::EventStream<'a, Self>> = #prismident<'a, S>;
            type Callback = (
                #callback_declarations
            );

            fn callback() -> Self::Callback {
                (#callback_none)
            }

            #[allow(non_snake_case)]
            fn prism<'a, S: #crate_name::event::EventStream<'a, Self>>(s: S) -> Self::Prism<'a, S> {
                let state = #crate_name::event::PrismInternal::<'a, Self, S>::new(s);
                #prismident {
                    #prism_declarations
                }
            }

            fn send(self, callback: &mut Self::Callback) -> #crate_name::event::EventRes {
                match self {
                    #send_declarations
                }
            }
        }

        #impl_declarations
    }
    .into()
}

#[proc_macro_derive(UserData)]
pub fn lua_user_data(input: TokenStream) -> TokenStream {
    /*let crate_name = std::env::var("CARGO_PKG_NAME").unwrap();

    let crate_name = format_ident!(
        "{}",
        if crate_name == "feather-ui" {
            "crate"
        } else {
            "feather_ui"
        }
    );*/
    let crate_name = format_ident!("feather_ui");

    let ast = parse_macro_input!(input as DeriveInput);
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let data = if let Data::Struct(data_enum) = &ast.data {
        data_enum
    } else {
        panic!("`UserData` derive can only be used on a struct.");
    };

    let mut field_methods = proc_macro2::TokenStream::new();
    for m in data.fields.members() {
        match m {
            syn::Member::Named(ident) => {
                field_methods.extend(quote! {
                    f.add_field_method_get(stringify!(#ident), |_, this| Ok(this.#ident.clone()));
                    f.add_field_method_set(stringify!(#ident), |_, this, v| Ok(this.#ident = v));
                });
            }
            syn::Member::Unnamed(_) => panic!(
                "You can't use a UserData derive on a tuple, because mlua knows how to parse tuples already!"
            ),
        }
    }

    let sname = ast.ident;
    quote! {
        impl #impl_generics #crate_name::mlua::UserData for #sname #ty_generics #where_clause {
            fn add_fields<F: #crate_name::mlua::UserDataFields<Self>>(f: &mut F) {
                #field_methods
            }
        }

        impl #impl_generics #crate_name::mlua::FromLua for #sname #ty_generics #where_clause {
            #[inline]
            fn from_lua(value: #crate_name::mlua::Value, _: &#crate_name::mlua::Lua) -> #crate_name::mlua::Result<Self> {
                match value {
                    #crate_name::mlua::Value::UserData(ud) => Ok(ud.borrow::<Self>()?.clone()),
                    _ => Err(#crate_name::mlua::Error::FromLuaConversionError {
                        from: value.type_name(),
                        to: stringify!(#sname).to_string(),
                        message: None,
                    }),
                }
            }
        }
    }
    .into()
}

#[proc_macro_attribute]
pub fn signal_def(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut ast = parse_macro_input!(input as DeriveInput);
    let mut list = std::vec![];
    match &mut ast.data {
        syn::Data::Struct(struct_data) => {
            match &mut struct_data.fields {
                syn::Fields::Named(fields) => {
                    for field in fields.named.iter_mut().filter(|x| match &x.ty {
                        syn::Type::Path(t) => {
                            if let Some(x) = t.path.segments.last()
                                && x.ident == format_ident!("Signal")
                            {
                                true
                            } else {
                                false
                            }
                        }
                        _ => false,
                    }) {
                        match &mut field.ty {
                            syn::Type::Path(signal) => {
                                match &mut signal.path.segments.last_mut().unwrap().arguments {
                                    syn::PathArguments::AngleBracketed(x) => {
                                        if x.args.len() == 1 {
                                            match &x.args[0] {
                                                syn::GenericArgument::AssocType(t) => {
                                                    list.push(t.ty.clone());
                                                    let idx = format_ident!("P{}", list.len());
                                                    x.args[0] = syn::GenericArgument::Type(
                                                        syn::parse_quote! { #idx },
                                                    )
                                                }
                                                _ => continue,
                                            }
                                        } else {
                                            continue;
                                        }
                                    }
                                    _ => continue,
                                }
                            }

                            _ => continue,
                        };
                    }
                }
                _ => (),
            }

            /*let crate_name = format_ident!(
                "{}",
                if std::env::var("CARGO_PKG_NAME").unwrap() == "feather-ui" {
                    "crate"
                } else {
                    "feather_ui"
                }
            );*/
            let crate_name = format_ident!("feather_ui");

            for i in 0..list.len() {
                let idx = format_ident!("P{}", i + 1);
                let param = &list[i];
                ast.generics.params.push(syn::GenericParam::Type(syn::parse_quote! { #idx : #crate_name::reactive::SignalProvider<Item = #param> + ?Sized }));
            }
        }
        _ => panic!("`signal_impl` is only for structs"),
    }

    return quote! {
        #ast
    }
    .into();
}
