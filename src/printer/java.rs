// mod printer;

use alloc::{format, vec};
use alloc::borrow::Cow;
use alloc::string::String;
use core::fmt::{Debug, Display, Formatter};

use crate::parser::{Fields, parse_enum, parse_struct, Struct};
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

pub trait ToJavaPrintable {
    fn print_as_java(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result;
}

pub struct ToJavaDisplay<'a, T, U>(&'a T, &'a U);

impl<'a, T: ToJavaPrintable, U: PrintContext> Debug for ToJavaDisplay<'a, T, U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        ToJavaPrintable::print_as_java(self.0, self.1, f)
    }
}

impl ToJavaPrintable for SymbolRef {
    fn print_as_java(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        match ctx.resolve(self.clone()) {
            Err(err) => write!(f, "<unresolved: {err}>"),
            // Ok(Symbol::Struct(sym)) => write!(f, "{}", sym.name),
            Ok(other) => if let Some(name) = other.name() {
                write!(f, "{name}")
            } else {
                write!(f, "<unnamed #{}>", self)
            },
        }
    }
}

impl ToJavaPrintable for Symbol {
    fn print_as_java(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        match self {
            Symbol::Struct(sym) => ToJavaPrintable::print_as_java(sym, ctx, f),
            Symbol::Enum(sym) => ToJavaPrintable::print_as_java(sym, ctx, f),
            Symbol::Function(sym) => ToJavaPrintable::print_as_java(sym, ctx, f),
        }
    }
}

impl ToJavaPrintable for StructDef {
    fn print_as_java(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        let name = self.name();
        write!(f, "{}class {name} {{", match self.scope() {
            Scope::Private => "",
            Scope::Public => "public "
        })?;

        let fields = self.fields();
        match fields {
            StructFields::Named(fields) => {
                for field in fields {
                    let name = match field.name() {
                        "class" => Cow::from(format!("_{}", field.name())),
                        name => Cow::from(name)
                    };
                    write!(f, "\n  private ")?;
                    ToJavaPrintable::print_as_java(field.r#type(), ctx, f)?;
                    write!(f, " {name};\n")?;
                }
            }
            StructFields::Tuple(fields) => {
                for field in fields {
                    let offset = field.offset();
                    write!(f, "\n  private ")?;
                    ToJavaPrintable::print_as_java(field.r#type(), ctx, f)?;
                    write!(f, " _{offset};\n")?;
                }
            }
            StructFields::Unit => {}
        }
        let mut first = true;
        write!(f, "\n  public {name}(")?;
        match fields {
            StructFields::Named(fields) => {
                for field in fields {
                    let name = match field.name() {
                        "class" => Cow::from(format!("_{}", field.name())),
                        name => Cow::from(name)
                    };
                    if first {
                        first = false;
                    } else {
                        write!(f, ", ")?;
                    }
                    ToJavaPrintable::print_as_java(field.r#type(), ctx, f)?;
                    write!(f, " {name}")?;
                }
            }
            StructFields::Tuple(fields) => {
                for field in fields {
                    let offset = field.offset();
                    if first {
                        first = false;
                    } else {
                        write!(f, ", ")?;
                    }
                    ToJavaPrintable::print_as_java(field.r#type(), ctx, f)?;
                    write!(f, " _{offset}")?;
                }
            }
            StructFields::Unit => {}
        }
        write!(f, ") {{\n")?;
        match fields {
            StructFields::Named(fields) => {
                for field in fields {
                    let name = match field.name() {
                        "class" => Cow::from(format!("_{}", field.name())),
                        name => Cow::from(name)
                    };
                    write!(f, "    this.{name} = {name};\n")?;
                }
            }
            StructFields::Tuple(fields) => {
                for field in fields {
                    let offset = field.offset();
                    write!(f, "    this._{offset} = _{offset};\n")?;
                }
            }
            StructFields::Unit => {}
        }
        write!(f, "  }}\n")?;
        match fields {
            StructFields::Named(fields) => {
                for field in fields {
                    if let Scope::Public = field.scope() {
                        let name = match field.name() {
                            "class" => Cow::from(format!("_{}", field.name())),
                            name => Cow::from(name)
                        };
                        write!(f, "\n  public ")?;
                        ToJavaPrintable::print_as_java(field.r#type(), ctx, f)?;
                        write!(f, " get{}{}() {{\n    return {name};\n  }}\n", name[0..1].to_ascii_uppercase(), &name[1..])?;
                    }
                }
            }
            StructFields::Tuple(fields) => {
                for field in fields {
                    if let Scope::Public = field.scope() {
                        let offset = field.offset();
                        write!(f, "\n  public ")?;
                        ToJavaPrintable::print_as_java(field.r#type(), ctx, f)?;
                        write!(f, " get{offset}() {{\n    return _{offset};\n  }}\n")?;
                    }
                }
            }
            StructFields::Unit => {}
        }
        write!(f, "}}")
    }
}

impl ToJavaPrintable for EnumDef {
    fn print_as_java(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        let name = self.name();
        let visibility = match self.scope() {
            Scope::Private => "",
            Scope::Public => "public "
        };
        write!(f, r#"{visibility}abstract class {name} {{
  public interface Visitor {{
    void defaultAction({name} visited);
"#)?;

        for variant in self.variants() {
            let name = variant.name();
            write!(f, r#"
    default void visit{name}({name} visited) {{
      defaultAction(visited);
    }}
"#)?;
        }

        write!(f, r#"  }}

  private {name}() {{}}

  public abstract void visit(Visitor visitor);
"#)?;

        for variant in self.variants() {
            let name = variant.name();
            write!(f, "\n  public static {name} create{name}(")?;
            match variant.fields() {
                EnumVariantFields::Named(fields) => {
                    let mut first = true;
                    for field in fields {
                        if first {
                            first = false;
                        } else {
                            write!(f, ", ")?;
                        }
                        let typ = field.r#type();
                        let name = field.name();
                        ToJavaPrintable::print_as_java(typ, ctx, f)?;
                        write!(f, " {name}")?;
                    }
                }
                EnumVariantFields::Tuple(fields) => {
                    let mut first = true;
                    for field in fields {
                        if first {
                            first = false;
                        } else {
                            write!(f, ", ")?;
                        }
                        let typ = field.r#type();
                        let offset = field.offset();
                        ToJavaPrintable::print_as_java(typ, ctx, f)?;
                        write!(f, " _{offset}")?;
                    }
                }
                EnumVariantFields::Unit => {}
            }
            write!(f, ") {{\n    return new {name}(")?;
            match variant.fields() {
                EnumVariantFields::Named(fields) => {
                    let mut first = true;
                    for field in fields {
                        if first {
                            first = false;
                        } else {
                            write!(f, ", ")?;
                        }
                        let name = field.name();
                        Display::fmt(name, f)?;
                    }
                }
                EnumVariantFields::Tuple(fields) => {
                    let mut first = true;
                    for field in fields {
                        if first {
                            first = false;
                        } else {
                            write!(f, ", ")?;
                        }
                        let offset = field.offset();
                        write!(f, "_{offset}")?;
                    }
                }
                EnumVariantFields::Unit => {}
            }
            write!(f, ");\n  }}\n")?;
        }

        for variant in self.variants() {
            let name = variant.name();
            write!(f, "\n  public static final class {name} {{\n")?;
            match variant.fields() {
                EnumVariantFields::Named(fields) => {
                    for field in fields {
                        write!(f, "    private final ")?;
                        let typ = field.r#type();
                        let name = field.name();
                        ToJavaPrintable::print_as_java(typ, ctx, f)?;
                        write!(f, " {name};\n")?;
                    }
                }
                EnumVariantFields::Tuple(fields) => {
                    for field in fields {
                        write!(f, "    private final ")?;
                        let typ = field.r#type();
                        let offset = field.offset();
                        ToJavaPrintable::print_as_java(typ, ctx, f)?;
                        write!(f, " _{offset};\n")?;
                    }
                }
                EnumVariantFields::Unit => {}
            }
            write!(f, "\n    private {name}(")?;
            match variant.fields() {
                EnumVariantFields::Named(fields) => {
                    let mut first = true;
                    for field in fields {
                        if first {
                            first = false;
                        } else {
                            write!(f, ", ")?;
                        }
                        let typ = field.r#type();
                        let name = field.name();
                        ToJavaPrintable::print_as_java(typ, ctx, f)?;
                        write!(f, " {name}")?;
                    }
                }
                EnumVariantFields::Tuple(fields) => {
                    let mut first = true;
                    for field in fields {
                        if first {
                            first = false;
                        } else {
                            write!(f, ", ")?;
                        }
                        let typ = field.r#type();
                        let offset = field.offset();
                        ToJavaPrintable::print_as_java(typ, ctx, f)?;
                        write!(f, " _{offset}")?;
                    }
                }
                EnumVariantFields::Unit => {}
            }
            write!(f, ") {{\n")?;
            match variant.fields() {
                EnumVariantFields::Named(fields) => {
                    for field in fields {
                        let name = field.name();
                        write!(f, "      this.{name} = {name};\n")?;
                    }
                }
                EnumVariantFields::Tuple(fields) => {
                    for field in fields {
                        let offset = field.offset();
                        write!(f, "      this._{offset} = _{offset};\n")?;
                    }
                }
                EnumVariantFields::Unit => {}
            }
            write!(f, "    }}\n")?;
            match variant.fields() {
                EnumVariantFields::Named(fields) => {
                    for field in fields {
                        if let Scope::Public = field.scope() {
                            let name = match field.name() {
                                "class" => Cow::from(format!("_{}", field.name())),
                                name => Cow::from(name)
                            };
                            write!(f, "\n    public ")?;
                            ToJavaPrintable::print_as_java(field.r#type(), ctx, f)?;
                            write!(f, " get{}{}() {{\n      return {name};\n    }}\n", name[0..1].to_ascii_uppercase(), &name[1..])?;
                        }
                    }
                }
                EnumVariantFields::Tuple(fields) => {
                    for field in fields {
                        if let Scope::Public = field.scope() {
                            let offset = field.offset();
                            write!(f, "\n    public ")?;
                            ToJavaPrintable::print_as_java(field.r#type(), ctx, f)?;
                            write!(f, " get{offset}() {{\n      return _{offset};\n    }}\n")?;
                        }
                    }
                }
                EnumVariantFields::Unit => {}
            }
            write!(f, r#"
    @Override
    public void visit(Visitor visitor); {{
      visitor.visit{name}(this);
    }}
  }}
"#)?;
        }

        write!(f, "}}")
    }
}

impl ToJavaPrintable for FunctionDef {
    fn print_as_java(&self, _: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "// TODO: print `fn {}({} param(s))`", self.name(), self.params().len())?;
        Ok(())
    }
}

// impl ToJavaPrintable for EnumDef {
//     fn print_as_java(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
//         let name = self.name();
//         let visibility = match self.scope() {
//             Scope::Private => "",
//             Scope::Public => "public "
//         };
//         write!(f, r#"{visibility}abstract class {name} {{
//   public interface Visitor {{"#)?;
//
//         for variant in self.variants() {
//             let name = variant.name();
//             write!(f, "\n    void visit{name}(")?;
//             match variant.fields() {
//                 EnumVariantFields::Named(fields) => {
//                     let mut first = true;
//                     for field in fields {
//                         if first {
//                             first = false;
//                         } else {
//                             write!(f, ", ")?;
//                         }
//                         let typ = field.r#type();
//                         let name = field.name();
//                         ToJavaPrintable::print_as_java(typ, ctx, f)?;
//                         write!(f, " {name}")?;
//                     }
//                 }
//                 EnumVariantFields::Tuple(fields) => {
//                     let mut first = true;
//                     for field in fields {
//                         if first {
//                             first = false;
//                         } else {
//                             write!(f, ", ")?;
//                         }
//                         let typ = field.r#type();
//                         let offset = field.offset();
//                         ToJavaPrintable::print_as_java(typ, ctx, f)?;
//                         write!(f, " _{offset}")?;
//                     }
//                 }
//                 EnumVariantFields::Unit => {}
//             }
//             write!(f, ");\n")?;
//         }
//
//         write!(f, r#"}}
//
//   private {name}() {{}}
//
//   public abstract void visit(Visitor visitor);
// "#)?;
//
//         for variant in self.variants() {
//             let name = variant.name();
//             write!(f, "\n  public static {name} create{name}(")?;
//             match variant.fields() {
//                 EnumVariantFields::Named(fields) => {
//                     let mut first = true;
//                     for field in fields {
//                         if first {
//                             first = false;
//                         } else {
//                             write!(f, ", ")?;
//                         }
//                         let typ = field.r#type();
//                         let name = field.name();
//                         ToJavaPrintable::print_as_java(typ, ctx, f)?;
//                         write!(f, " {name}")?;
//                     }
//                 }
//                 EnumVariantFields::Tuple(fields) => {
//                     let mut first = true;
//                     for field in fields {
//                         if first {
//                             first = false;
//                         } else {
//                             write!(f, ", ")?;
//                         }
//                         let typ = field.r#type();
//                         let offset = field.offset();
//                         ToJavaPrintable::print_as_java(typ, ctx, f)?;
//                         write!(f, " _{offset}")?;
//                     }
//                 }
//                 EnumVariantFields::Unit => {}
//             }
//             write!(f, ") {{\n    return new {name}(")?;
//             match variant.fields() {
//                 EnumVariantFields::Named(fields) => {
//                     let mut first = true;
//                     for field in fields {
//                         if first {
//                             first = false;
//                         } else {
//                             write!(f, ", ")?;
//                         }
//                         let name = field.name();
//                         Display::fmt(name, f)?;
//                     }
//                 }
//                 EnumVariantFields::Tuple(fields) => {
//                     let mut first = true;
//                     for field in fields {
//                         if first {
//                             first = false;
//                         } else {
//                             write!(f, ", ")?;
//                         }
//                         let offset = field.offset();
//                         write!(f, "_{offset}")?;
//                     }
//                 }
//                 EnumVariantFields::Unit => {}
//             }
//             write!(f, ");\n  }}\n")?;
//         }
//
//         for variant in self.variants() {
//             let name = variant.name();
//             write!(f, "\n  private static class {name} {{\n")?;
//             match variant.fields() {
//                 EnumVariantFields::Named(fields) => {
//                     for field in fields {
//                         write!(f, "    private final ")?;
//                         let typ = field.r#type();
//                         let name = field.name();
//                         ToJavaPrintable::print_as_java(typ, ctx, f)?;
//                         write!(f, " {name};\n")?;
//                     }
//                 }
//                 EnumVariantFields::Tuple(fields) => {
//                     for field in fields {
//                         write!(f, "    private final ")?;
//                         let typ = field.r#type();
//                         let offset = field.offset();
//                         ToJavaPrintable::print_as_java(typ, ctx, f)?;
//                         write!(f, " _{offset};\n")?;
//                     }
//                 }
//                 EnumVariantFields::Unit => {}
//             }
//             write!(f, "\n    private {name}(")?;
//             match variant.fields() {
//                 EnumVariantFields::Named(fields) => {
//                     let mut first = true;
//                     for field in fields {
//                         if first {
//                             first = false;
//                         } else {
//                             write!(f, ", ")?;
//                         }
//                         let typ = field.r#type();
//                         let name = field.name();
//                         ToJavaPrintable::print_as_java(typ, ctx, f)?;
//                         write!(f, " {name}")?;
//                     }
//                 }
//                 EnumVariantFields::Tuple(fields) => {
//                     let mut first = true;
//                     for field in fields {
//                         if first {
//                             first = false;
//                         } else {
//                             write!(f, ", ")?;
//                         }
//                         let typ = field.r#type();
//                         let offset = field.offset();
//                         ToJavaPrintable::print_as_java(typ, ctx, f)?;
//                         write!(f, " _{offset}")?;
//                     }
//                 }
//                 EnumVariantFields::Unit => {}
//             }
//             write!(f, ") {{\n")?;
//             match variant.fields() {
//                 EnumVariantFields::Named(fields) => {
//                     for field in fields {
//                         let name = field.name();
//                         write!(f, "      this.{name} = {name};\n")?;
//                     }
//                 }
//                 EnumVariantFields::Tuple(fields) => {
//                     for field in fields {
//                         let offset = field.offset();
//                         write!(f, "      this._{offset} = _{offset};\n")?;
//                     }
//                 }
//                 EnumVariantFields::Unit => {}
//             }
//             write!(f, r#"    }}
//
//     @Override
//     public void visit(Visitor visitor); {{
//       visitor.visit{name}("#)?;
//
//             match variant.fields() {
//                 EnumVariantFields::Named(fields) => {
//                     let mut first = true;
//                     for field in fields {
//                         if first {
//                             first = false;
//                         } else {
//                             write!(f, ", ")?;
//                         }
//                         let name = field.name();
//                         Display::fmt(name, f)?;
//                     }
//                 }
//                 EnumVariantFields::Tuple(fields) => {
//                     let mut first = true;
//                     for field in fields {
//                         if first {
//                             first = false;
//                         } else {
//                             write!(f, ", ")?;
//                         }
//                         let offset = field.offset();
//                         write!(f, "_{offset}")?;
//                     }
//                 }
//                 EnumVariantFields::Unit => {}
//             }
//             write!(f, ");\n    }}\n  }}\n")?;
//         }
//
// //         write!(f, r#"}}
// //
// //   private {name}() {{}}
// //
// //   public abstract void visit(Visitor visitor);
// //
// //   // TODO add variants factory methods
// //   // TODO add variants sub-classes
// // "#)?;
//
//         // let variants = self.variants();
//         // for variant in variants {
//         //     let name = variant.name();
//         //     write!(f, "\n  public static create{name}()")?;
//         //     ToJavaPrintable::print_as_java(variant.fields(), ctx, f)?;
//         //     write!(f, " {};\n", name)?;
//         // }
//         // for variant in variants {
//         //     if let Scope::Public = variant.scope() {
//         //         let name = match variant.name() {
//         //             "class" => Cow::from(format!("_{}", variant.name())),
//         //             name => Cow::from(name)
//         //         };
//         //         write!(f, "\n  public ")?;
//         //         ToJavaPrintable::print_as_java(variant.r#type(), ctx, f)?;
//         //         write!(f, " get{}{}() {{\n    return {};\n  }}\n", name[0..1].to_ascii_uppercase(), &name[1..], name)?;
//         //     }
//         // }
//         write!(f, "}}")
//     }
// }

impl ToJavaPrintable for TypeRef {
    fn print_as_java(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        match &self {
            TypeRef::Primitive(typ) => ToJavaPrintable::print_as_java(typ, ctx, f),
            TypeRef::Ref(typ) => ToJavaPrintable::print_as_java(typ, ctx, f)
        }
    }
}

impl ToJavaPrintable for PrimitiveType {
    fn print_as_java(&self, _: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        Display::fmt(match &self {
            PrimitiveType::I16 => "short",
            PrimitiveType::U32 => "long",
            PrimitiveType::I64 => "long",
            PrimitiveType::F64 => "double",
            PrimitiveType::Unit => "void",
            PrimitiveType::Usize => "int",
        }, f)
    }
}

impl ToJavaPrintable for Module {
    fn print_as_java(&self, ctx: &impl PrintContext, f: &mut Formatter) -> core::fmt::Result {
        write!(f, "package {};\n", self.name())?;
        for next in self.symbols() {
            write!(f, "\n")?;
            ToJavaPrintable::print_as_java(next, ctx, f)?;
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Module {
    pub fn to_java(&self) -> String {
        format!("{:?}", ToJavaDisplay(self, self))
    }
}

#[test]
fn it_prints_in_java() -> Result<()> {
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
            visibility: Visibility::Pub,
            name: Identifier("Token"),
            body: Fields::NamedFields(vec![
                NamedField {
                    visibility: Visibility::Default,
                    name: Identifier("struct"),
                    typ: Type::Identifier(Identifier("TokenKind")),
                },
                NamedField {
                    visibility: Visibility::Pub,
                    name: Identifier("pub"),
                    typ: Type::Usize,
                },
            ]),
        },
        root
    );

    // let mut tokens = Token::parse_ascii(r#"struct TokenKind {}"#)?;

    let module = ModuleBuilder::new("my_first_module")
        .add_struct(root)
        .add_enum(parse_enum(&mut tokens)?)
        .build()?;

    assert_eq!(module.to_java(), r#"package my_first_module;

public class Token {
  private TokenKind struct;

  private int pub;

  public Token(TokenKind struct, int pub) {
    this.struct = struct;
    this.pub = pub;
  }

  public int getPub() {
    return pub;
  }
}

abstract class TokenKind {
  public interface Visitor {
    void defaultAction(TokenKind visited);

    default void visitEqual(Equal visited) {
      defaultAction(visited);
    }

    default void visitUnexpected(Unexpected visited) {
      defaultAction(visited);
    }

    default void visitTuple(Tuple visited) {
      defaultAction(visited);
    }
  }

  private TokenKind() {}

  public abstract void visit(Visitor visitor);

  public static Equal createEqual() {
    return new Equal();
  }

  public static Unexpected createUnexpected(int _0) {
    return new Unexpected(_0);
  }

  public static Tuple createTuple(Token a, int b) {
    return new Tuple(a, b);
  }

  public static final class Equal {

    private Equal() {
    }

    @Override
    public void visit(Visitor visitor); {
      visitor.visitEqual(this);
    }
  }

  public static final class Unexpected {
    private final int _0;

    private Unexpected(int _0) {
      this._0 = _0;
    }

    public int get0() {
      return _0;
    }

    @Override
    public void visit(Visitor visitor); {
      visitor.visitUnexpected(this);
    }
  }

  public static final class Tuple {
    private final Token a;
    private final int b;

    private Tuple(Token a, int b) {
      this.a = a;
      this.b = b;
    }

    @Override
    public void visit(Visitor visitor); {
      visitor.visitTuple(this);
    }
  }
}
"#);

    Ok(())
}
