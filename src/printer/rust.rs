use alloc::{format, vec};
use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use core::fmt::{Debug, Display, Formatter};

use crate::parser::{BlockExpression, Expression, Fields, FunctionDeclaration, FunctionPrototype, Mutability, parse_enum, parse_function_declaration, parse_struct, Struct, VariableSymbolDeclaration};
use crate::types::{EnumDef, EnumVariantFields, FunctionDef, Module, ModuleBuilder, PrimitiveType, Scope, StructDef, StructFields, Symbol, SymbolRef, TypeRef};

type Result<T> = core::result::Result<T, Cow<'static, str>>;

pub trait PrintContext {
    fn resolve(&self, r#ref: SymbolRef) -> Result<&Symbol>;
}

impl PrintContext for Module {
    fn resolve(&self, r#ref: SymbolRef) -> Result<&Symbol> {
        self.get_by_ref(r#ref)
    }
}

pub trait ToRustPrintable {
    fn print_as_rust(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result;
}

pub struct ToRustDisplay<'a, T, U>(&'a T, &'a U);

impl<'a, T: ToRustPrintable, U: PrintContext> Debug for ToRustDisplay<'a, T, U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        ToRustPrintable::print_as_rust(self.0, self.1, f)
    }
}

impl ToRustPrintable for SymbolRef {
    fn print_as_rust(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        match ctx.resolve(self.clone()) {
            Err(err) => write!(f, "<unresolved: {err}>"),
            // Ok(Symbol::Struct(sym)) => write!(f, "{}", sym.name),
            Ok(other) => if let Some(name) = Some(other.name()) {
                write!(f, "{name}")
            } else {
                write!(f, "<unnamed #{}>", self)
            },
        }
    }
}

impl ToRustPrintable for Symbol {
    fn print_as_rust(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        match self {
            Symbol::Struct(sym) => ToRustPrintable::print_as_rust(sym, ctx, f),
            Symbol::Enum(sym) => ToRustPrintable::print_as_rust(sym, ctx, f),
            Symbol::Function(sym) => ToRustPrintable::print_as_rust(sym, ctx, f),
            // Symbol::Enum(sym) => todo!("TODO print Enums"),
        }
    }
}

impl ToRustPrintable for StructDef {
    fn print_as_rust(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "{}struct ", match self.scope() {
            Scope::Private => "",
            Scope::Public => "pub "
        })?;

        let fields = self.fields();
        let should_add_semicolon = match fields {
            StructFields::Named(fields) => {
                let mut debug_struct = f.debug_struct(self.name());
                for field in fields {
                    debug_struct.field((match field.name() {
                        "struct" | "pub" => Cow::from(format!("r#{}", field.name())),
                        name => Cow::from(name)
                    }).as_ref(), &ToRustDisplay(field.r#type(), ctx));
                }
                debug_struct.finish()?;
                fields.is_empty()
            }
            StructFields::Tuple(fields) => {
                let mut debug_struct = f.debug_tuple(self.name());
                for field in fields {
                    debug_struct.field(&ToRustDisplay(field.r#type(), ctx));
                }
                debug_struct.finish()?;
                fields.is_empty()
            }
            StructFields::Unit => {
                f.debug_struct(self.name()).finish()?;
                true
            }
        };
        if should_add_semicolon {
            write!(f, ";")?;
        }
        Ok(())
    }
}

impl ToRustPrintable for EnumDef {
    fn print_as_rust(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "{}enum {} {{ ", match self.scope() {
            Scope::Private => "",
            Scope::Public => "pub "
        }, self.name())?;

        for variant in self.variants() {
            let name = match variant.name() {
                "struct" | "pub" => Cow::from(format!("r#{}", variant.name())),
                name => Cow::from(name)
            };
            Display::fmt(name.as_ref(), f)?;
            ToRustPrintable::print_as_rust(variant.fields(), ctx, f)?;
            write!(f, ", ")?;
        }

        write!(f, " }}")?;
        Ok(())
    }
}

impl ToRustPrintable for EnumVariantFields {
    fn print_as_rust(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        match self {
            EnumVariantFields::Named(fields) => {
                write!(f, " {{ ")?;
                for field in fields {
                    let name = match field.name() {
                        "struct" | "pub" => Cow::from(format!("r#{}", field.name())),
                        name => Cow::from(name)
                    };
                    write!(f, "{}: ", name.as_ref())?;
                    ToRustPrintable::print_as_rust(field.r#type(), ctx, f)?;
                    write!(f, ", ")?;
                }
                write!(f, " }}")?;
            }
            EnumVariantFields::Tuple(fields) => {
                write!(f, "(")?;
                for field in fields {
                    ToRustPrintable::print_as_rust(field.r#type(), ctx, f)?;
                    write!(f, ", ")?;
                }
                write!(f, ")")?;
            }
            EnumVariantFields::Unit => {}
        }
        Ok(())
    }
}

impl ToRustPrintable for FunctionDef {
    fn print_as_rust(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "{}fn {}(", match self.scope() {
            Scope::Private => "",
            Scope::Public => "pub "
        }, self.name())?;
        for (i, p) in self.params().iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            let mutable = if p.is_mutable() { "mut " } else { "" };
            let name = p.name();
            write!(f, "{mutable}{name}: ")?;
            ToRustPrintable::print_as_rust(p.typ(), ctx, f)?;
        }
        write!(f, ") ")?;
        if !matches!(self.ret_type(), TypeRef::Primitive(PrimitiveType::Unit)) {
            write!(f, "-> ")?;
            ToRustPrintable::print_as_rust(self.ret_type(), ctx, f)?;
        }
        write!(f, "{{\n    // TODO: print body :/\n  }}")?;
        // write!(f, "// TODO: print `fn {}({} param(s)) {:#?}`", self.name(), self.params().len(), self.body())?;
        Ok(())
    }
}

impl ToRustPrintable for TypeRef {
    fn print_as_rust(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        match &self {
            TypeRef::Primitive(typ) => ToRustPrintable::print_as_rust(typ, ctx, f),
            TypeRef::Ref(typ) => ToRustPrintable::print_as_rust(typ, ctx, f)
        }
    }
}

impl ToRustPrintable for PrimitiveType {
    fn print_as_rust(&self, _: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        Display::fmt(match &self {
            PrimitiveType::I16 => "i16",
            PrimitiveType::U32 => "u32",
            PrimitiveType::I64 => "i64",
            PrimitiveType::F64 => "f64",
            PrimitiveType::Unit => "()",
            PrimitiveType::Usize => "usize",
        }, f)
    }
}

impl ToRustPrintable for Module {
    fn print_as_rust(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "mod {} {{", self.name())?;
        for next in self.symbols() {
            write!(f, "\n  ")?;
            ToRustPrintable::print_as_rust(next, ctx, f)?;
            write!(f, "\n")?;
        }
        write!(f, "}}")
    }
}

impl Module {
    pub fn to_rust(&self) -> String {
        format!("{:?}", ToRustDisplay(self, self))
    }
}

#[test]
fn it_prints_in_rust() -> Result<()> {
    use crate::lexer::Token;
    use crate::parser::{Identifier, NamedField, Type, Visibility};

    let mut tokens = Token::parse_ascii(r#"
pub struct Token {
  struct: TokenKind,
  pub pub: usize,
}

enum TokenKind {
  Equal,
  Unexpected(pub usize),
  Tuple { a: Token, b: usize }
}
"#)?;

    let root = parse_struct(&mut tokens)?;

    assert_eq!(
        Struct {
            doc: Box::new([]),
            visibility: Visibility::Pub,
            name: Identifier("Token"),
            body: Fields::NamedFields(vec![
                NamedField {
                    doc: Box::new([]),
                    visibility: Visibility::Default,
                    name: Identifier("struct"),
                    typ: Type::Identifier(Identifier("TokenKind")),
                },
                NamedField {
                    doc: Box::new([]),
                    visibility: Visibility::Pub,
                    name: Identifier("pub"),
                    typ: Type::Usize,
                },
            ]),
        },
        root
    );

    let module = ModuleBuilder::new("my_first_module")
        .add_struct(root)
        .add_enum(parse_enum(&mut tokens)?)
        .build()?;

    assert_eq!(module.to_rust(), r#"mod my_first_module {
  pub struct Token { r#struct: TokenKind, r#pub: usize }

  enum TokenKind { Equal, Unexpected(usize, ), Tuple { a: Token, b: usize,  },  }
}"#);

    Ok(())
}


#[test]
fn it_prints_functions_in_rust() -> Result<()> {
    use crate::lexer::Token;
    use crate::parser::{Identifier, Type, Visibility};

    let mut tokens = Token::parse_ascii(r#"
pub fn life(mut unused: u32) -> i64 {
  40 + 2
}
"#)?;

    let root = parse_function_declaration(&mut tokens)?;

    assert_eq!(
        FunctionDeclaration {
            doc: Box::new([]),
            prototype: FunctionPrototype {
                visibility: Visibility::Pub,
                name: Identifier("life"),
                parameters: vec![VariableSymbolDeclaration {
                    mutability: Mutability::Mutable,
                    name: Identifier("unused"),
                    typ: Some(Type::U32),
                }],
                ret_type: Some(Type::I64),
            },
            body: BlockExpression {
                expressions: vec![],
                remainder: Some(Rc::new(Expression::Add(Rc::new(
                    Expression::Literal(40)
                ), Rc::new(
                    Expression::Literal(2)
                )))),
            },
        },
        root
    );

    let module = ModuleBuilder::new("my_first_module")
        .add_function(root)
        .build()?;

    assert_eq!(module.to_rust(), r#"mod my_first_module {
  pub fn life(mut unused: u32) -> i64{
    // TODO: print body :/
  }
}"#);

    Ok(())
}
