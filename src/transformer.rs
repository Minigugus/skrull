use alloc::{format, vec};
use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{Display, Formatter};
use core::ops::Deref;

use crate::bytecode::{SkBody, SkOp, SkTerminatorOp, SkValueKind};
use crate::lexer::Token;
use crate::mlir::ops::{BlockBuilder, Body as Body0, RefId, RuntimeValue, Typed};
use crate::types::{EnumDef, EnumVariantFields, FunctionDef, Module, PrimitiveType, Scope, StructDef, StructFields, Symbol, SymbolRef, TypeRef, value_kind_to_type_ref};

type JavaSymbolRef = (usize, Rc<String>);

trait ToJavaResolver {
    fn convert_refs(&self, r: &SymbolRef) -> JavaSymbolRef;
    fn get_fn_container(&self, r: &SymbolRef) -> Option<JavaSymbolRef>;

    fn convert_types(&self, typ: &TypeRef) -> JavaType {
        JavaType::from_type_ref(self, typ)
    }

    fn convert_kind(&self, typ: &SkValueKind) -> JavaValueKind {
        let kind = value_kind_to_type_ref(typ);
        match kind {
            Some(typ) => Self::convert_types(self, &typ).into(),
            None => JavaValueKind::Never
        }
    }
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
enum JavaType {
    Void,
    Byte,
    Short,
    Int,
    Long,
    Float,
    Double,
    Class(JavaSymbolRef),
}

impl Display for JavaType {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            JavaType::Void => write!(f, "void"),
            JavaType::Byte => write!(f, "byte"),
            JavaType::Short => write!(f, "short"),
            JavaType::Int => write!(f, "int"),
            JavaType::Long => write!(f, "long"),
            JavaType::Float => write!(f, "float"),
            JavaType::Double => write!(f, "double"),
            JavaType::Class((_, fqdn)) => write!(f, "{fqdn}"),
        }
    }
}

impl JavaType {
    pub fn from_type_ref(ctx: &(impl ToJavaResolver + ?Sized), value: &TypeRef) -> Self {
        match value {
            TypeRef::Primitive(p) => match p {
                PrimitiveType::I16 => JavaType::Short,
                PrimitiveType::U32 => JavaType::Long,
                PrimitiveType::I64 => JavaType::Long,
                PrimitiveType::F64 => JavaType::Double,
                PrimitiveType::Unit => JavaType::Void,
                PrimitiveType::Usize => JavaType::Int,
            }
            TypeRef::Ref(r) => JavaType::Class(ctx.convert_refs(r))
        }
    }
}

fn print_as_java_doc(f: &mut Formatter, indent: &str, doc: &Option<Rc<str>>, params: Option<&[JavaField]>) -> Result<(), core::fmt::Error> {
    let mut created = false;
    if let Some(doc) = doc {
        for line in doc.lines() {
            if !created {
                created = true;
                f.write_str(indent)?;
                f.write_str("/**\n")?;
            }
            f.write_str(indent)?;
            if line.trim().is_empty() {
                f.write_str(" * <p>\n")?;
            } else {
                write!(f, " *{line}\n")?;
            }
        }
    }

    if let Some(params) = params {
        let mut first = true;
        for param in params {
            let Some(doc) = &param.doc else { continue; };
            for line in doc.lines() {
                if line.trim().is_empty() {
                    continue;
                } else {
                    if !created {
                        created = true;
                        first = false;
                        f.write_str(indent)?;
                        f.write_str("/**\n")?;
                    } else if first {
                        first = false;
                        f.write_str(indent)?;
                        f.write_str(" *\n")?;
                    }
                    f.write_str(indent)?;
                    let line = if line.starts_with(' ') { &line[1..] } else { line };
                    write!(f, " * @param {} {line}\n", param.name)?;
                }
            }
        }
    }

    if created {
        f.write_str(indent)?;
        f.write_str(" */\n")?;
    }

    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
struct JavaField {
    doc: Option<Rc<str>>,
    name: String,
    typ: JavaType,
}

impl Display for JavaField {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self {
            typ,
            name,
            ..
        } = self;
        write!(f, "  {typ} {name}")
    }
}

#[derive(Debug, Eq, PartialEq)]
struct JavaRecord {
    doc: Option<Rc<str>>,
    visibility: JavaVisibility,
    name: String,
    implements: Option<JavaSymbolRef>,
    fields: Vec<JavaField>,
}

struct WithMethods<'a, T>(&'a T, &'a BTreeMap<usize, JavaFunction>);

impl<'a> Display for WithMethods<'a, JavaRecord> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self(
            JavaRecord {
                doc,
                visibility,
                name,
                implements,
                fields
            },
            methods
        ) = self;
        print_as_java_doc(f, "", doc, Some(fields.as_slice()))?;
        write!(f, "{visibility}record {name}(")?;
        let mut fields = fields.iter();
        if let Some(field) = fields.next() {
            write!(f, "\n{field}")?;
            while let Some(field) = fields.next() {
                write!(f, ",\n{field}")?;
            }
            write!(f, "\n)")?;
        } else {
            write!(f, ")")?;
        }
        if let Some(implements) = implements {
            let implements = &implements.1;
            write!(f, " implements {implements}")?;
        }
        if methods.is_empty() {
            write!(f, " {{ }}")?;
        } else {
            write!(f, " {{\n\n")?;
            for fun in methods.values() {
                write!(f, "{fun}")?;
            }
            write!(f, "\n}}")?;
        };
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
struct JavaSealedInterface {
    doc: Option<Rc<str>>,
    visibility: JavaVisibility,
    name: String,
    permitted: Vec<JavaRecord>,
}

impl<'a> Display for WithMethods<'a, JavaSealedInterface> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self(
            JavaSealedInterface {
                doc,
                visibility,
                name,
                permitted
            },
            methods
        ) = self;
        print_as_java_doc(f, "", doc, None)?;
        write!(f, "{visibility}sealed interface {name} {{\n")?;
        let mut permitted = permitted.iter();
        while let Some(variant) = permitted.next() {
            write!(f, "\n{}", WithMethods(variant, *methods))?;
        }
        write!(f, "\n\n}}\n")?;
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
struct JavaUtilClass {
    visibility: JavaVisibility,
    name: String,
}

impl<'a> Display for WithMethods<'a, JavaUtilClass> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self(
            JavaUtilClass {
                visibility,
                name,
            },
            methods
        ) = self;
        write!(f, "{visibility}final class {name} {{\n\n")?;
        for fun in methods.values() {
            write!(f, "{fun}")?;
        }
        write!(f, "\n}}")?;
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
struct JavaFunction {
    doc: Option<Rc<str>>,
    visibility: JavaVisibility,
    name: String,
    parameters: Vec<JavaFunctionParameter>,
    ret_type: JavaType,
    body: JavaBody,
}

#[derive(Debug, Eq, PartialEq)]
struct JavaFunctionParameter {
    name: String,
    fqdn: JavaType,
}

impl Display for JavaFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self {
            doc,
            visibility,
            name,
            parameters,
            ret_type,
            body
        } = self;
        print_as_java_doc(f, "  ", doc, None)?;
        write!(f, "  {visibility}static {ret_type} {name}(")?;
        let mut parameters = parameters.iter();
        let mut is_first = true;
        while let Some(JavaFunctionParameter { name, fqdn }) = parameters.next() {
            if is_first {
                is_first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{fqdn} {name}")?;
        }
        write!(f, ") {{\n")?;

        fn get(f: &mut Formatter<'_>, loc: &mut BTreeMap<usize, Rc<String>>, body: &JavaBody, v: &JavaValueRef) -> Result<Rc<String>, core::fmt::Error> {
            let op = body.op(v).ok_or(core::fmt::Error)?;
            if !matches!(v.typ(), JavaValueKind::Var(_)) && !matches!(op, JavaOpN::GetParam(_, _) | JavaOpN::ConstLong(_) | JavaOpN::ConstShort(_)) {
                let id = match v.id() {
                    RefId::Param(pi) => unreachable!(), // FIXME
                    RefId::Op(oi) => oi
                };
                if let Some(v) = loc.get(&id).cloned() {
                    Ok(v)
                } else {
                    let expr = print_rec(f, loc, body, op, &v.typ())?;
                    let var_name = Rc::new(format!("_{}", loc.len()));
                    loc.insert(id, var_name.clone());
                    write!(f, "    final {} {var_name} = {expr};\n", v.typ())?;
                    // write!(f, "    final var {var_name} = {expr};\n")?;
                    Ok(var_name)
                }
            } else {
                Ok(print_rec(f, loc, body, op, &v.typ())?)
            }
        }

        fn print_rec(f: &mut Formatter<'_>, loc: &mut BTreeMap<usize, Rc<String>>, body: &JavaBody, op: &JavaOpN, kind: &JavaValueKind) -> Result<Rc<String>, core::fmt::Error> {
            match kind {
                JavaValueKind::Var(v) => return Ok(Rc::new(v.0.clone())),
                _ => {}
            }
            match op {
                JavaOpN::ConstShort(n) => Ok(Rc::new(format!("{n}"))),
                JavaOpN::ConstLong(n) => Ok(Rc::new(format!("{n}"))),
                JavaOpN::Add(l, r) => {
                    let l = get(f, loc, body, l)?;
                    let r = get(f, loc, body, r)?;
                    Ok(Rc::new(format!("{l} + {r}")))
                }
                JavaOpN::Create((_, s), p) => {
                    let p = p
                        .iter()
                        .map(|v| get(f, loc, body, v).map(|s| s.to_string())) // FIXME
                        .collect::<Result<Vec<_>, core::fmt::Error>>()?
                        .join(", ");
                    Ok(Rc::new(format!("new {s}({p})")))
                }
                JavaOpN::Neg(v) => {
                    let v = get(f, loc, body, v)?;
                    Ok(Rc::new(format!("-{v}")))
                }
                JavaOpN::Nop(v) => get(f, loc, body, v),
                JavaOpN::GetParam(_, n) => Ok(n.clone()),
                JavaOpN::InvokeStatic((_, s), _, n, a) => {
                    let a = a
                        .iter()
                        .map(|v| get(f, loc, body, v).map(|s| s.to_string())) // FIXME
                        .collect::<Result<Vec<_>, core::fmt::Error>>()?
                        .join(", ");
                    Ok(Rc::new(format!("{s}.{n}({a})")))
                }
                JavaOpN::Error(msg) => {
                    Ok(Rc::new(format!("/* TODO {msg} */")))
                }
            }
        }

        match body.terminator_op() {
            JavaTerminatorOpN::Return => {}
            JavaTerminatorOpN::ReturnValue(v) => {
                let mut loc = BTreeMap::new();
                let expr = get(f, &mut loc, body, v)?;
                write!(f, "    return {expr};\n")?;
            }
        }
        write!(f, "  }}\n")?;
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
enum JavaSymbolKind {
    Record(JavaRecord),
    SealedInterface(JavaSealedInterface),
    UtilClass(JavaUtilClass),
}

impl<'a> Display for WithMethods<'a, JavaSymbolKind> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self(sym, methods) = self;
        match sym {
            JavaSymbolKind::Record(r) => Display::fmt(&WithMethods(r, methods), f),
            JavaSymbolKind::SealedInterface(si) => Display::fmt(&WithMethods(si, methods), f),
            JavaSymbolKind::UtilClass(uc) => Display::fmt(&WithMethods(uc, methods), f),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct JavaSymbol {
    pkg: String,
    kind: JavaSymbolKind,
    methods: BTreeMap<usize, JavaFunction>,
}

impl Display for JavaSymbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self {
            pkg,
            kind,
            methods
        } = self;
        write!(f, "package {pkg};\n\n{}", WithMethods(kind, methods))
    }
}

enum JavaPendingSymbol<'a> {
    Record(&'a StructDef),
    SealedInterface(&'a EnumDef),
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

struct JavaModuleBuilder<'a> {
    fqdn: String,
    pending_symbols: Vec<(Rc<String>, JavaPendingSymbol<'a>)>,
    pending_functions: Vec<(SymbolRef, &'a FunctionDef)>,
    by_ref: BTreeMap<SymbolRef, JavaSymbolRef>,
    by_fqdn: BTreeMap<Rc<String>, JavaSymbolRef>,
    fn_to_container: BTreeMap<SymbolRef, JavaSymbolRef>,
}

impl<'a> ToJavaResolver for JavaModuleBuilder<'a> {
    fn convert_refs(&self, r: &SymbolRef) -> JavaSymbolRef {
        self.by_ref
            .get(r)
            .cloned()
            .expect("SymbolRef should exist")
    }

    fn get_fn_container(&self, r: &SymbolRef) -> Option<JavaSymbolRef> {
        self.fn_to_container
            .get(r)
            .cloned()
    }
}

impl<'a> TryFrom<&'a Module> for JavaModule {
    type Error = Cow<'static, str>;

    fn try_from(src: &'a Module) -> Result<Self, Self::Error> {
        let root_pkg = src.name();
        let mut builder = JavaModuleBuilder {
            fqdn: format!("{}.{}", root_pkg, "Root"),
            pending_symbols: Default::default(),
            pending_functions: Default::default(),
            by_ref: Default::default(),
            by_fqdn: Default::default(),
            fn_to_container: Default::default(),
        };
        for (id, s) in src.symbols_ref() {
            let java_ref: JavaSymbolRef = (
                builder.pending_symbols.len(),
                Rc::new(format!("{root_pkg}.{}", s.name()))
            );
            builder.pending_symbols.push((java_ref.1.clone(), match s {
                Symbol::Struct(v) => v.into(),
                Symbol::Enum(v) => v.into(),
                Symbol::Function(v) => {
                    builder.pending_functions.push((id, v));
                    continue;
                }
            }));
            builder.by_ref.insert(id, java_ref.clone());
            builder.by_fqdn.insert(java_ref.1.clone(), java_ref);
        }
        let mut module = Self {
            symbols: vec![],
            by_fqdn: Default::default(),
        };
        for (fqdn, sym) in &builder.pending_symbols {
            let id = module.symbols.len();
            let sym = match sym {
                JavaPendingSymbol::Record(s) => JavaSymbol {
                    pkg: root_pkg.to_string(),
                    methods: Default::default(),
                    kind: JavaSymbolKind::Record(JavaRecord {
                        doc: s.doc().cloned(),
                        visibility: JavaVisibility::from_type_scope(s.scope()),
                        name: s.name().to_string(),
                        implements: None,
                        fields: match s.fields() {
                            StructFields::Named(s) => s
                                .iter()
                                .filter_map(|f| Some(JavaField {
                                    doc: f.doc().cloned(),
                                    name: f.name().to_string(),
                                    typ: match builder.convert_types(f.r#type()) {
                                        JavaType::Void => return None,
                                        t => t
                                    },
                                }))
                                .collect(),
                            StructFields::Tuple(s) => s
                                .iter()
                                .filter_map(|f| Some(JavaField {
                                    doc: None,
                                    name: format!("_{}", f.offset()),
                                    typ: match builder.convert_types(f.r#type()) {
                                        JavaType::Void => return None,
                                        t => t
                                    },
                                }))
                                .collect(),
                            StructFields::Unit => Vec::new()
                        },
                    }),
                },
                JavaPendingSymbol::SealedInterface(e) => JavaSymbol {
                    pkg: root_pkg.to_string(),
                    methods: Default::default(),
                    kind: JavaSymbolKind::SealedInterface(JavaSealedInterface {
                        doc: e.doc().cloned(),
                        visibility: JavaVisibility::from_type_scope(e.scope()),
                        name: e.name().to_string(),
                        permitted: e.variants()
                            .iter()
                            .map(|v| JavaRecord {
                                doc: v.doc().cloned(),
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
                                            doc: f.doc().cloned(),
                                            name: f.name().to_string(),
                                            typ: match builder.convert_types(f.r#type()) {
                                                JavaType::Void => return None,
                                                t => t
                                            },
                                        }))
                                        .collect(),
                                    EnumVariantFields::Tuple(f) => f
                                        .iter()
                                        .filter_map(|f| Some(JavaField {
                                            doc: None,
                                            name: format!("_{}", f.offset()),
                                            typ: match builder.convert_types(f.r#type()) {
                                                JavaType::Void => return None,
                                                t => t
                                            },
                                        }))
                                        .collect(),
                                    EnumVariantFields::Unit => Vec::with_capacity(0)
                                },
                            })
                            .collect(),
                    }),
                }
            };
            module.symbols.push(sym);
            module.by_fqdn.insert(fqdn.to_string(), id);
        }
        let utils_sym: JavaSymbolRef = (
            builder.pending_symbols.len(),
            Rc::new(format!("{root_pkg}.Utils"))
        );
        module.by_fqdn.insert(utils_sym.1.to_string(), utils_sym.0);
        module.symbols.push(JavaSymbol {
            pkg: root_pkg.to_string(),
            kind: JavaSymbolKind::UtilClass(JavaUtilClass {
                visibility: JavaVisibility::Public,
                name: "Utils".to_string(),
            }),
            methods: Default::default(),
        });
        for (id, def) in &builder.pending_functions {
            let Some(body) = def.body() else { continue; };

            let ret_type = builder.convert_types(def.ret_type());

            let function = JavaFunction {
                doc: def.doc().cloned(),
                visibility: JavaVisibility::from_type_scope(def.scope()),
                name: def.name().to_string(),
                parameters: def.params()
                    .iter()
                    .map(|p| JavaFunctionParameter {
                        name: p.name().to_string(),
                        fqdn: builder.convert_types(p.typ()),
                    }).collect(),
                ret_type: ret_type.clone(),
                body: TryFrom::try_from((body, &builder, &module))?,
            };

            let sym_ref = match ret_type {
                JavaType::Class(sym) => sym,
                _ => match (function.parameters.len(), function.parameters.get(0)) {
                    (1, Some(JavaFunctionParameter { fqdn: JavaType::Class(sym), .. })) => sym,
                    _ => &utils_sym
                }.clone()
            };
            let Some(sym) = module.by_fqdn
                .get(sym_ref.1.as_str())
                .cloned()
                .and_then(|fqdn| module.symbols.get_mut(fqdn)) else { continue; };
            sym.methods.insert(id.id(), function);
            builder.fn_to_container.insert(id.clone(), sym_ref);
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

#[derive(Debug, Eq, PartialEq)]
enum JavaOpN {
    Error(Cow<'static, str>),
    Nop(JavaValueRef),
    Neg(JavaValueRef),
    ConstShort(i16),
    ConstLong(i64),
    GetParam(JavaValueKind, Rc<String>),
    InvokeStatic(JavaSymbolRef, JavaValueKind, Rc<String>, Vec<JavaValueRef>),
    Create(JavaSymbolRef, Vec<JavaValueRef>),
    Add(JavaValueRef, JavaValueRef),
}

#[derive(Debug, Eq, PartialEq)]
enum JavaTerminatorOpN {
    Return,
    ReturnValue(JavaValueRef),
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum JavaValueKind {
    Never,
    Void,
    // Bool,
    // Long,
    // Short,
    // Double,
    // Int,
    // Type(JavaSymbolRef),
    JavaType(JavaType),
    Var(Rc<(String, JavaValueKind)>),
}

impl JavaValueKind {
    pub fn as_value_type(&self) -> &JavaValueKind {
        match self {
            JavaValueKind::Var(v) => &v.deref().1,
            o => o
        }
    }
}

impl Typed for JavaOpN {
    type ValueType = JavaValueKind;

    fn typ(&self) -> Self::ValueType {
        fn bin_op_typ<T: Typed<ValueType=JavaValueKind>>(l: &T, r: &T) -> JavaValueKind {
            let l = l.typ().as_value_type().clone();
            let r = r.typ();
            if l == *r.as_value_type() {
                l
            } else {
                JavaValueKind::Never
            }
        }

        match self {
            JavaOpN::Nop(v) => v.typ(),
            JavaOpN::Neg(v) => v.typ(),
            JavaOpN::ConstShort(_) => JavaValueKind::JavaType(JavaType::Short),
            JavaOpN::ConstLong(_) => JavaValueKind::JavaType(JavaType::Long),
            JavaOpN::GetParam(k, _) => k.clone(),
            JavaOpN::InvokeStatic(_, k, _, _) => k.clone(),
            JavaOpN::Create(s, _) => JavaValueKind::JavaType(JavaType::Class(s.clone())),
            JavaOpN::Add(l, r) => bin_op_typ(l, r),
            JavaOpN::Error(_) => JavaValueKind::Never
        }
    }
}

impl Typed for JavaTerminatorOpN {
    type ValueType = JavaValueKind;

    fn typ(&self) -> Self::ValueType {
        match self {
            JavaTerminatorOpN::Return => JavaValueKind::Void,
            JavaTerminatorOpN::ReturnValue(v) => v.typ()
        }
    }
}

impl Display for JavaValueKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            JavaValueKind::Never => write!(f, "???"),
            JavaValueKind::Void => write!(f, "void"),
            // JavaValueKind::Bool => write!(f, "boolean"),
            // JavaValueKind::Byte => write!(f, "byte"),
            // JavaValueKind::Short => write!(f, "short"),
            // JavaValueKind::Int => write!(f, "int"),
            // JavaValueKind::Long => write!(f, "long"),
            // JavaValueKind::Float => write!(f, "float"),
            // JavaValueKind::Double => write!(f, "double"),
            // JavaValueKind::Type((_, fqdn)) => Display::fmt(&*fqdn, f),
            JavaValueKind::JavaType(jt) => Display::fmt(jt, f),
            JavaValueKind::Var(v) => Display::fmt(&v.0, f),
        }
    }
}

impl From<JavaType> for JavaValueKind {
    fn from(value: JavaType) -> Self {
        match value {
            JavaType::Void => Self::Void,
            value => Self::JavaType(value)
        }
    }
}

type JavaValueRef = RuntimeValue<JavaValueKind>;

type JavaBody = Body0<JavaValueKind, JavaOpN, JavaTerminatorOpN>;
type JavaBlockBuilder<'a> = BlockBuilder<'a, SkValueKind, JavaOpN, JavaTerminatorOpN>;

impl<'a> TryFrom<(&'a SkBody, &'a JavaModuleBuilder<'_>, &'a JavaModule)> for JavaBody {
    type Error = Cow<'static, str>;

    fn try_from((pb, builder, module): (&'a SkBody, &'a JavaModuleBuilder, &'a JavaModule)) -> Result<Self, Self::Error> {
        let params = pb.params()
            .iter()
            .map(|x| JavaValueKind::Never) // TODO
            .collect::<Vec<_>>();
        Ok(Self::isolated(params.as_slice(), |b, args| {
            let mut mapping: BTreeMap<usize, JavaValueRef> = BTreeMap::new();
            for (id, op) in pb.ops().iter().enumerate() {
                let jop = match op {
                    SkOp::Label(v, n) => {
                        match v.id() {
                            RefId::Param(_) => {
                                let s = builder.convert_kind(&v.typ());
                                JavaOpN::GetParam(JavaValueKind::Var(Rc::new((n.clone(), s))), Rc::new(n.clone()))
                            }
                            RefId::Op(oi) => match mapping.get(&oi).cloned() {
                                None => JavaOpN::Error("no mapping found for SkOp::Add arg".into()),
                                Some(v) => JavaOpN::Nop(v)
                            }
                        }
                    }
                    SkOp::ConstUnit => JavaOpN::Error("SkOp::ConstUnit conversion not implemented".into()),
                    SkOp::ConstI64(n) if *n < i16::MAX as i64 => JavaOpN::ConstShort(n.clone() as i16),
                    SkOp::ConstI64(n) => JavaOpN::ConstLong(n.clone()),
                    SkOp::Block(_) => JavaOpN::Error("SkOp::Block conversion not implemented".into()), // TODO
                    SkOp::Neg(v) => match v.id() {
                        RefId::Param(_) => JavaOpN::Error("SkOp::Neg arg cannot be an RefId::Param".into()),
                        RefId::Op(ref v) => match mapping.get(v).cloned() {
                            None => JavaOpN::Error("no mapping found for SkOp::Neg arg".into()),
                            Some(v) => JavaOpN::Neg(v),
                        }
                    }
                    SkOp::Add(l, r) => match (l.id(), r.id()) {
                        (RefId::Param(_), _) => JavaOpN::Error("SkOp::Add left arg cannot be an RefId::Param".into()),
                        (_, RefId::Param(_)) => JavaOpN::Error("SkOp::Add right arg cannot be an RefId::Param".into()),
                        (RefId::Op(ref l), RefId::Op(ref r)) => match (mapping.get(l).cloned(), mapping.get(r).cloned()) {
                            (None, _) => JavaOpN::Error("no mapping found for SkOp::Add left arg".into()),
                            (_, None) => JavaOpN::Error("no mapping found for SkOp::Add right arg".into()),
                            (Some(l), Some(r)) => JavaOpN::Add(l, r),
                        }
                    }
                    SkOp::Mul(_, _) => JavaOpN::Error("SkOp::ConstUnit conversion not implemented".into()), // TODO
                    SkOp::Gt(_, _) => JavaOpN::Error("SkOp::Gt conversion not implemented".into()), // TODO
                    SkOp::If(_, _, _) => JavaOpN::Error("SkOp::If conversion not implemented".into()), // TODO
                    SkOp::Call(s, _, a) => {
                        let id = s.id();
                        match builder.get_fn_container(s) {
                            None => JavaOpN::Error(format!("function #{s} not found in SkOp::Call").into()), // TODO inline function?
                            Some(s) => match module.resolve(s.1.as_str()) {
                                None => JavaOpN::Error("container not found in SkOp::Call".into()),
                                Some(m) => match m.methods.get(&id) {
                                    None => JavaOpN::Error("container doesn't contain function in SkOp::Call".into()),
                                    Some(f) => {
                                        let a = a
                                            .iter()
                                            .enumerate()
                                            .map(|(i, v)| match v.id() {
                                                RefId::Param(_) => Err(format!("SkOp::Call arg {} cannot be an RefId::Param", i + 1).into()),
                                                RefId::Op(oi) => mapping
                                                    .get(&oi)
                                                    .cloned()
                                                    .map(Ok)
                                                    .unwrap_or_else(|| Err(format!("no mapping found for SkOp::Add arg {}", i + 1).into()))
                                            })
                                            .collect::<Result<Vec<_>, Self::Error>>()?;
                                        JavaOpN::InvokeStatic(
                                            s,
                                            f.ret_type.clone().into(),
                                            Rc::new(f.name.clone()),
                                            a,
                                        )
                                    }
                                }
                            }
                        }
                    }
                    SkOp::Create(s, p) => {
                        let p = p
                            .iter()
                            .map(|(n, v)| (n, v))
                            .collect::<BTreeMap<_, _>>();
                        let s = builder.convert_refs(s);
                        match module.resolve(s.1.as_str()) {
                            None => JavaOpN::Error("symbol not found in SkOp::Create".into()),
                            Some(m) => match m.kind {
                                JavaSymbolKind::Record(ref m) => m.fields
                                    .iter()
                                    .enumerate()
                                    .map(|(i, f)| {
                                        let Some(v) = p.get(&f.name) else {
                                            return Err(format!("missing parameter {} '{}' in SkOp::Create", i + 1, f.name))?;
                                        };
                                        match v.id() {
                                            RefId::Param(_) => return Err(format!("SkOp::Create arg {} cannot be an RefId::Param", i + 1))?,
                                            RefId::Op(oi) => match mapping.get(&oi) {
                                                None => return Err(format!("no mapping found for SkOp::Create arg {}", i + 1))?,
                                                Some(v) => Ok(v.clone())
                                            }
                                        }
                                    })
                                    .collect::<Result<Vec<_>, Self::Error>>()
                                    .map(|params| JavaOpN::Create(s.clone(), params))
                                    .unwrap_or_else(|err| JavaOpN::Error(err)),
                                _ => JavaOpN::Error("only Records can be constructed in SkOp::Create".into()),
                            },
                        }
                    }
                    SkOp::Match(_, _) => JavaOpN::Error("SkOp::Match conversion not implemented".into()), // TODO
                };
                let r = b.op(jop);
                mapping.insert(id, r);
            }
            if let SkTerminatorOp::Yield(ref v) = pb.terminator_op() {
                if let Some(v) = match v.id() {
                    RefId::Param(_) => None,
                    RefId::Op(oi) => Some(oi)
                }.and_then(|oi| mapping.get(&oi)).cloned() {
                    return Ok(JavaTerminatorOpN::ReturnValue(v));
                }
            };
            Ok(JavaTerminatorOpN::Return)
        })?)
    }
}

#[test]
fn it_generates_java() -> Result<(), Cow<'static, str>> {
    use crate::lexer::Token;

    let module = Module::parse_tokens("my_first_module", /*language=rust*/Token::parse_ascii(r#"
fn new_point(x: i16, y: i16) -> Point {
  Point {
    y: y + -1,
    x
  }
}

/// Doc is also supported *_*
fn new_rect(x: i16, y: i16, w: i16, h: i16) -> Rectangle {
  Rectangle {
    origin: new_point(x, y),
    size: Size {
      width: atimes2(w),
      height: h,
    },
  }
}

fn atimes2(v: i16) -> i16 {
  v + v
}

// declaration order shouldn't matter

/// A simple shape
struct Rectangle {
  origin: Point,

  /// Dimensions, in pixels
  size: Size,
}

pub struct Size {
  /// W
  width: i16,
  /// H
  height: i16,
}

/// A point with a X and Y coordinate
///
/// NOTE: coordinates are i16 since Skrull doesn't support isize primitive type yet.
struct Point {
  x: i16,
  y: i16
}

/// Shapes we are able to create
enum Shape {
  /// The simplest shape available
  Rect(Rectangle),
}
"#)?)?;

    let java = JavaModule::try_from(&module)?;

    assert_eq!(
        java.resolve("my_first_module.Shape").map(ToString::to_string),
        /*language=java*/Some(r#"package my_first_module;

/**
 * Shapes we are able to create
 */
sealed interface Shape {

/**
 * The simplest shape available
 */
record Rect(
  my_first_module.Rectangle _0
) implements my_first_module.Shape { }

}
"#.to_string())
    );

    assert_eq!(
        java.resolve("my_first_module.Size").map(ToString::to_string),
        /*language=java*/Some(r#"package my_first_module;

/**
 * @param width W
 * @param height H
 */
public record Size(
  short width,
  short height
) { }"#.to_string())
    );

    assert_eq!(
        java.resolve("my_first_module.Point").map(ToString::to_string),
        /*language=java*/Some(r#"package my_first_module;

/**
 * A point with a X and Y coordinate
 * <p>
 * NOTE: coordinates are i16 since Skrull doesn't support isize primitive type yet.
 */
record Point(
  short x,
  short y
) {

  static my_first_module.Point new_point(short x, short y) {
    final short _0 = -1;
    final short _1 = y + _0;
    final my_first_module.Point _2 = new my_first_module.Point(x, _1);
    return _2;
  }

}"#.to_string())
    );

    assert_eq!(
        java.resolve("my_first_module.Utils").map(ToString::to_string),
        /*language=java*/Some(r#"package my_first_module;

public final class Utils {

  static short atimes2(short v) {
    final short _0 = v + v;
    return _0;
  }

}"#.to_string())
    );

    assert_eq!(
        java.resolve("my_first_module.Rectangle").map(ToString::to_string),
        /*language=java*/Some(r#"package my_first_module;

/**
 * A simple shape
 *
 * @param size Dimensions, in pixels
 */
record Rectangle(
  my_first_module.Point origin,
  my_first_module.Size size
) {

  /**
   * Doc is also supported *_*
   */
  static my_first_module.Rectangle new_rect(short x, short y, short w, short h) {
    final my_first_module.Point _0 = my_first_module.Point.new_point(x, y);
    final short _1 = my_first_module.Utils.atimes2(w);
    final my_first_module.Size _2 = new my_first_module.Size(_1, h);
    final my_first_module.Rectangle _3 = new my_first_module.Rectangle(_0, _2);
    return _3;
  }

}"#.to_string())
    );

    assert_eq!(java.resolve("my_first_module.Size"), Some(&JavaSymbol {
        pkg: "my_first_module".to_string(),
        methods: Default::default(),
        kind: JavaSymbolKind::Record(JavaRecord {
            doc: None,
            visibility: JavaVisibility::Public,
            name: "Size".to_string(),
            implements: None,
            fields: vec![
                JavaField {
                    doc: Some(Rc::from(" W")),
                    name: "width".to_string(),
                    typ: JavaType::Short,
                },
                JavaField {
                    doc: Some(Rc::from(" H")),
                    name: "height".to_string(),
                    typ: JavaType::Short,
                },
            ],
        }),
    }));

    // assert_eq!(java.resolve("my_first_module.Point"), Some(&JavaSymbol {
    //     pkg: "my_first_module".to_string(),
    //     methods: vec![
    //         JavaFunction {
    //             visibility: JavaVisibility::PackagePrivate,
    //             name: "new_point".into(),
    //             parameters: vec![
    //                 JavaFunctionParameter {
    //                     name: "x".into(),
    //                     fqdn: JavaType::Short,
    //                 },
    //                 JavaFunctionParameter {
    //                     name: "y".into(),
    //                     fqdn: JavaType::Short,
    //                 },
    //             ],
    //             ret_type: JavaType::Class((0, Rc::new("my_first_module.Point".into()))),
    //             body: Default::default(),
    //         }
    //     ],
    //     kind: JavaSymbolKind::Record(JavaRecord {
    //         visibility: JavaVisibility::PackagePrivate,
    //         name: "Point".to_string(),
    //         implements: None,
    //         fields: vec![
    //             JavaField {
    //                 name: "x".to_string(),
    //                 typ: JavaType::Short,
    //             },
    //             JavaField {
    //                 name: "y".to_string(),
    //                 typ: JavaType::Short,
    //             },
    //         ],
    //     }),
    // }));

    // assert_eq!(java.resolve("my_first_module.Rectangle"), Some(&JavaSymbol {
    //     pkg: "my_first_module".to_string(),
    //     methods: vec![],
    //     kind: JavaSymbolKind::Record(JavaRecord {
    //         visibility: JavaVisibility::PackagePrivate,
    //         name: "Rectangle".to_string(),
    //         implements: None,
    //         fields: vec![
    //             JavaField {
    //                 name: "origin".to_string(),
    //                 typ: JavaType::Class((0, Rc::from("my_first_module.Point".to_string()))),
    //             },
    //             JavaField {
    //                 name: "size".to_string(),
    //                 typ: JavaType::Class((3, Rc::from("my_first_module.Size".to_string()))),
    //             },
    //         ],
    //     }),
    // }));

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
) {

  public static short is_priced(skull_test_transform_enum.Some maybe_price) {
    final ??? _0 = /* TODO SkOp::Match conversion not implemented */;
    return _0;
  }

}"#.to_string())
    );

    Ok(())
}
