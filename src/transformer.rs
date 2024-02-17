use alloc::{format, vec};
use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{Display, Formatter};

use crate::lexer::Token;
use crate::types::{EnumDef, EnumVariantFields, FunctionDef, Module, PrimitiveType, Scope, StructDef, StructFields, Symbol, SymbolRef, TypeRef};

type JavaSymbolRef = (usize, Rc<String>);

#[derive(Debug, Eq, PartialEq)]
enum JavaVisibility {
    Public,
    Protected,
    PackagePrivate,
    Private,
}

impl JavaVisibility {
    pub fn from_type_scope(value: Scope) -> Self {
        match value {
            Scope::Private => JavaVisibility::PackagePrivate,
            Scope::Public => JavaVisibility::Public,
        }
    }
}

impl Display for JavaVisibility {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            JavaVisibility::Public => write!(f, "public "),
            JavaVisibility::Protected => write!(f, "protected "),
            JavaVisibility::PackagePrivate => Ok(()),
            JavaVisibility::Private => write!(f, "private "),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum JavaType {
    Byte,
    Short,
    Int,
    Long,
    Float,
    Double,
    Record(JavaSymbolRef),
}

impl Display for JavaType {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            JavaType::Byte => write!(f, "byte"),
            JavaType::Short => write!(f, "short"),
            JavaType::Int => write!(f, "int"),
            JavaType::Long => write!(f, "long"),
            JavaType::Float => write!(f, "float"),
            JavaType::Double => write!(f, "double"),
            JavaType::Record((_, fqdn)) => write!(f, "{fqdn}"),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct JavaField {
    name: String,
    typ: JavaType,
}

impl Display for JavaField {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self {
            typ,
            name
        } = self;
        write!(f, "{typ} {name}")
    }
}

#[derive(Debug, Eq, PartialEq)]
struct JavaRecord {
    visibility: JavaVisibility,
    name: String,
    implements: Option<JavaSymbolRef>,
    fields: Vec<JavaField>,
}

impl Display for JavaRecord {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self {
            visibility,
            name,
            implements,
            fields
        } = self;
        write!(f, "{visibility}record {name}(")?;
        let mut fields = fields.iter();
        if let Some(field) = fields.next() {
            write!(f, "\n  {field}")?;
            while let Some(field) = fields.next() {
                write!(f, ",\n  {field}")?;
            }
            write!(f, "\n)")?;
        } else {
            write!(f, ")")?;
        }
        if let Some(implements) = implements {
            let implements = &implements.1;
            write!(f, " implements {implements}")?;
        }
        write!(f, " {{ }}")
    }
}

#[derive(Debug, Eq, PartialEq)]
struct JavaSealedInterface {
    visibility: JavaVisibility,
    name: String,
    permitted: Vec<JavaRecord>,
}

impl Display for JavaSealedInterface {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self {
            visibility,
            name,
            permitted
        } = self;
        write!(f, "{visibility}sealed interface {name} {{\n")?;
        let mut permitted = permitted.iter();
        while let Some(variant) = permitted.next() {
            write!(f, "\n{variant}")?;
        }
        write!(f, "\n\n}}\n")?;
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
enum JavaSymbolKind {
    Record(JavaRecord),
    SealedInterface(JavaSealedInterface),
}

impl Display for JavaSymbolKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            JavaSymbolKind::Record(r) => Display::fmt(r, f),
            JavaSymbolKind::SealedInterface(si) => Display::fmt(si, f),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct JavaSymbol {
    pkg: String,
    kind: JavaSymbolKind,
}

impl Display for JavaSymbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self {
            pkg,
            kind
        } = self;
        write!(f, "package {pkg};\n\n{kind}")
    }
}

enum JavaPendingSymbol<'a> {
    Record(&'a StructDef),
    SealedInterface(&'a EnumDef),
    Function(&'a FunctionDef),
}

impl<'a> From<&'a StructDef> for JavaPendingSymbol<'a> {
    fn from(value: &'a StructDef) -> Self {
        Self::Record(value)
    }
}

impl<'a> From<&'a EnumDef> for JavaPendingSymbol<'a> {
    fn from(value: &'a EnumDef) -> Self {
        Self::SealedInterface(value)
    }
}

impl<'a> From<&'a FunctionDef> for JavaPendingSymbol<'a> {
    fn from(value: &'a FunctionDef) -> Self {
        Self::Function(value)
    }
}

struct JavaModuleBuilder<'a> {
    fqdn: String,
    pending_symbols: Vec<(Rc<String>, JavaPendingSymbol<'a>)>,
    by_ref: BTreeMap<SymbolRef, JavaSymbolRef>,
    by_fqdn: BTreeMap<Rc<String>, JavaSymbolRef>,
}

impl<'a> TryFrom<&'a Module> for JavaModule {
    type Error = Cow<'static, str>;
    // pub fn new(fqdn: String) -> Self {
    //     Self {
    //         fqdn,
    //         pending_symbols: vec![],
    //         by_ref: Default::default(),
    //     }
    // }
    //
    // pub fn add_symbol<'b: 'a>(mut self, sym: impl Into<JavaPendingSymbol<'a>>) -> Self {
    //     self.pending_symbols.push(sym.into());
    //     self
    // }
    //
    // pub fn build(self) -> JavaModule {
    //     let module = JavaModule {
    //         symbols: vec![],
    //     };
    //     self.pending_symbols
    //     module
    // }

    fn try_from(src: &'a Module) -> Result<Self, Self::Error> {
        let root_pkg = src.name();
        let mut builder = JavaModuleBuilder {
            fqdn: format!("{}.{}", root_pkg, "Root"),
            pending_symbols: vec![],
            by_ref: Default::default(),
            by_fqdn: Default::default(),
        };
        for (id, s) in src.symbols_ref() {
            let java_ref: JavaSymbolRef = (
                builder.pending_symbols.len(),
                Rc::new(format!("{}.{}", root_pkg, s.name()))
            );
            builder.pending_symbols.push((java_ref.1.clone(), match s {
                Symbol::Struct(v) => v.into(),
                Symbol::Enum(v) => v.into(),
                Symbol::Function(v) => v.into()
            }));
            builder.by_ref.insert(id, java_ref.clone());
            builder.by_fqdn.insert(java_ref.1.clone(), java_ref);
        }
        let mut module = Self {
            symbols: vec![],
            by_fqdn: Default::default(),
        };
        for (fqdn, sym) in builder.pending_symbols {
            let id = module.symbols.len();
            let sym = match sym {
                JavaPendingSymbol::Record(s) => JavaSymbol {
                    pkg: root_pkg.to_string(),
                    kind: JavaSymbolKind::Record(JavaRecord {
                        visibility: JavaVisibility::from_type_scope(s.scope()),
                        name: s.name().to_string(),
                        implements: None,
                        fields: match s.fields() {
                            StructFields::Named(s) => s
                                .iter()
                                .filter_map(|f| Some(JavaField {
                                    name: f.name().to_string(),
                                    typ: match f.r#type() {
                                        TypeRef::Primitive(p) => match p {
                                            PrimitiveType::I16 => JavaType::Short,
                                            PrimitiveType::U32 => JavaType::Long,
                                            PrimitiveType::I64 => JavaType::Long,
                                            PrimitiveType::F64 => JavaType::Double,
                                            PrimitiveType::Unit => return None, // skip field
                                            PrimitiveType::Usize => JavaType::Int,
                                        }
                                        TypeRef::Ref(r) => JavaType::Record(
                                            builder.by_ref
                                                .get(r)
                                                .cloned()
                                                .expect("couldn't resolve type ref")
                                        )
                                    },
                                }))
                                .collect(),
                            StructFields::Tuple(s) => s
                                .iter()
                                .filter_map(|f| Some(JavaField {
                                    name: format!("_{}", f.offset()),
                                    typ: match f.r#type() {
                                        TypeRef::Primitive(p) => match p {
                                            PrimitiveType::I16 => JavaType::Short,
                                            PrimitiveType::U32 => JavaType::Long,
                                            PrimitiveType::I64 => JavaType::Long,
                                            PrimitiveType::F64 => JavaType::Double,
                                            PrimitiveType::Unit => return None, // skip field
                                            PrimitiveType::Usize => JavaType::Int,
                                        }
                                        TypeRef::Ref(r) => JavaType::Record(
                                            builder.by_ref
                                                .get(r)
                                                .cloned()
                                                .expect("couldn't resolve type ref")
                                        )
                                    },
                                }))
                                .collect(),
                            StructFields::Unit => Vec::new()
                        },
                    }),
                },
                JavaPendingSymbol::SealedInterface(e) => JavaSymbol {
                    pkg: root_pkg.to_string(),
                    kind: JavaSymbolKind::SealedInterface(JavaSealedInterface {
                        visibility: JavaVisibility::from_type_scope(e.scope()),
                        name: e.name().to_string(),
                        permitted: e.variants()
                            .iter()
                            .map(|v| JavaRecord {
                                visibility: JavaVisibility::PackagePrivate,
                                name: v.name().to_string(),
                                implements: Some(builder.by_fqdn
                                    .get(&*fqdn)
                                    .cloned()
                                    .expect("a parent interface")),
                                fields: match v.fields() {
                                    EnumVariantFields::Named(f) => f
                                        .iter()
                                        .filter_map(|f| Some(JavaField {
                                            name: f.name().to_string(),
                                            typ: match f.r#type() {
                                                TypeRef::Primitive(p) => match p {
                                                    PrimitiveType::I16 => JavaType::Short,
                                                    PrimitiveType::U32 => JavaType::Long,
                                                    PrimitiveType::I64 => JavaType::Long,
                                                    PrimitiveType::F64 => JavaType::Double,
                                                    PrimitiveType::Unit => return None, // skip field
                                                    PrimitiveType::Usize => JavaType::Int,
                                                }
                                                TypeRef::Ref(r) => JavaType::Record(
                                                    builder.by_ref
                                                        .get(r)
                                                        .cloned()
                                                        .expect("couldn't resolve type ref")
                                                )
                                            },
                                        }))
                                        .collect(),
                                    EnumVariantFields::Tuple(f) => f
                                        .iter()
                                        .filter_map(|f| Some(JavaField {
                                            name: format!("_{}", f.offset()),
                                            typ: match f.r#type() {
                                                TypeRef::Primitive(p) => match p {
                                                    PrimitiveType::I16 => JavaType::Short,
                                                    PrimitiveType::U32 => JavaType::Long,
                                                    PrimitiveType::I64 => JavaType::Long,
                                                    PrimitiveType::F64 => JavaType::Double,
                                                    PrimitiveType::Unit => return None, // skip field
                                                    PrimitiveType::Usize => JavaType::Int,
                                                }
                                                TypeRef::Ref(r) => JavaType::Record(
                                                    builder.by_ref
                                                        .get(r)
                                                        .cloned()
                                                        .expect("couldn't resolve type ref")
                                                )
                                            },
                                        }))
                                        .collect(),
                                    EnumVariantFields::Unit => Vec::with_capacity(0)
                                },
                            })
                            .collect(),
                    }),
                },
                JavaPendingSymbol::Function(f) => continue
            };
            module.symbols.push(sym);
            module.by_fqdn.insert(fqdn.to_string(), id);
        }
        Ok(module)
    }
}

struct JavaModule {
    symbols: Vec<JavaSymbol>,
    by_fqdn: BTreeMap<String, usize>,
}

impl JavaModule {
    pub fn resolve(&self, fqdn: &str) -> Option<&JavaSymbol> {
        self.by_fqdn
            .get(fqdn)
            .and_then(|id| self.symbols.get(*id))
    }
}

#[test]
fn it_generates_java() -> Result<(), Cow<'static, str>> {
    use crate::lexer::Token;

    let module = Module::parse_tokens("my_first_module", /*language=rust*/Token::parse_ascii(r#"
fn new_point(x: i16, y: i16) -> Point {
  Point {
    x,
    y: y + -1
  }
}

// declaration order shouldn't matter

struct Rectangle {
  origin: Point,
  size: Size,
}

pub struct Size {
  width: i16,
  height: i16,
}

struct Point {
  x: i16,
  y: i16,
}

enum Shape {
  Rect(Rectangle),
}
"#)?)?;

    let java = JavaModule::try_from(&module)?;

    assert_eq!(
        java.resolve("my_first_module.Shape").map(ToString::to_string),
        /*language=java*/Some(r#"package my_first_module;

sealed interface Shape {

record Rect(
  my_first_module.Rectangle _0
) implements my_first_module.Shape { }

}
"#.to_string())
    );

    assert_eq!(
        java.resolve("my_first_module.Size").map(ToString::to_string),
        /*language=java*/Some(r#"package my_first_module;

public record Size(
  short width,
  short height
) { }"#.to_string())
    );

    assert_eq!(
        java.resolve("my_first_module.Point").map(ToString::to_string),
        /*language=java*/Some(r#"package my_first_module;

record Point(
  short x,
  short y
) { }"#.to_string())
    );

    assert_eq!(
        java.resolve("my_first_module.Rectangle").map(ToString::to_string),
        /*language=java*/Some(r#"package my_first_module;

record Rectangle(
  my_first_module.Point origin,
  my_first_module.Size size
) { }"#.to_string())
    );

    assert_eq!(java.resolve("my_first_module.Size"), Some(&JavaSymbol {
        pkg: "my_first_module".to_string(),
        kind: JavaSymbolKind::Record(JavaRecord {
            visibility: JavaVisibility::Public,
            name: "Size".to_string(),
            implements: None,
            fields: vec![
                JavaField {
                    name: "width".to_string(),
                    typ: JavaType::Short,
                },
                JavaField {
                    name: "height".to_string(),
                    typ: JavaType::Short,
                },
            ],
        }),
    }));

    assert_eq!(java.resolve("my_first_module.Point"), Some(&JavaSymbol {
        pkg: "my_first_module".to_string(),
        kind: JavaSymbolKind::Record(JavaRecord {
            visibility: JavaVisibility::PackagePrivate,
            name: "Point".to_string(),
            implements: None,
            fields: vec![
                JavaField {
                    name: "x".to_string(),
                    typ: JavaType::Short,
                },
                JavaField {
                    name: "y".to_string(),
                    typ: JavaType::Short,
                },
            ],
        }),
    }));

    assert_eq!(java.resolve("my_first_module.Rectangle"), Some(&JavaSymbol {
        pkg: "my_first_module".to_string(),
        kind: JavaSymbolKind::Record(JavaRecord {
            visibility: JavaVisibility::PackagePrivate,
            name: "Rectangle".to_string(),
            implements: None,
            fields: vec![
                JavaField {
                    name: "origin".to_string(),
                    typ: JavaType::Record((0, Rc::from("my_first_module.Point".to_string()))),
                },
                JavaField {
                    name: "size".to_string(),
                    typ: JavaType::Record((3, Rc::from("my_first_module.Size".to_string()))),
                },
            ],
        }),
    }));

    Ok(())
}

#[test]
fn it_transform_enum() -> Result<(), Cow<'static, str>> {

    // tokenize
    let tokens = /*language=rust*/Token::parse_ascii(r#"pub enum Price {
  Limit,
  Market,
  StopLimit { stop_price: f64, },
}

struct Some(Price);

pub fn is_priced(maybe_price: Some) -> i16 {
  match maybe_price {
    Some(Price::Limit | Price::StopLimit) => 1,
    _ => 0
  }
}
"#)?;

    // parse + semantic analysis
    let module = Module::parse_tokens("skull_test_transform_enum", tokens)?;

    // transform
    let java = JavaModule::try_from(&module)?;

    assert_eq!(
        java.resolve("skull_test_transform_enum.Price").map(ToString::to_string),
        /*language=java*/Some(r#"package skull_test_transform_enum;

public sealed interface Price {

record Limit() implements skull_test_transform_enum.Price { }
record Market() implements skull_test_transform_enum.Price { }
record StopLimit(
  double stop_price
) implements skull_test_transform_enum.Price { }

}
"#.to_string())
    );

    assert_eq!(
        java.resolve("skull_test_transform_enum.Some").map(ToString::to_string),
        /*language=java*/Some(r#"package skull_test_transform_enum;

record Some(
  skull_test_transform_enum.Price _0
) { }"#.to_string())
    );

    Ok(())
}
