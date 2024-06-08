use alloc::{format, vec};
use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{Display, Error, Formatter};
use core::ops::Deref;

use crate::bytecode::{SkBody, SkOp, SkTerminatorOp, SkValueKind};
use crate::lexer::Token;
use crate::mlir::ops::{BlockBuilder, Body as Body0, RefId, RuntimeValue, Typed};
use crate::scope::{Loc, Scope, Scopes};
use crate::types::{EnumDef, EnumVariantFields, FunctionDef, Module, PrimitiveType, Scope as ScopeNode, StructDef, StructFields, Symbol, SymbolRef, TypeRef, value_kind_to_type_ref};

mod match_op;

type JavaSymbolRef = (usize, Rc<String>, Option<String>);

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
    pub fn from_type_scope(value: ScopeNode) -> Self {
        match value {
            ScopeNode::Private => JavaVisibility::PackagePrivate,
            ScopeNode::Public => JavaVisibility::Public,
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
    Boolean,
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
            JavaType::Boolean => write!(f, "boolean"),
            JavaType::Byte => write!(f, "byte"),
            JavaType::Short => write!(f, "short"),
            JavaType::Int => write!(f, "int"),
            JavaType::Long => write!(f, "long"),
            JavaType::Float => write!(f, "float"),
            JavaType::Double => write!(f, "double"),
            JavaType::Class((_, fqdn, ..)) => write!(f, "{fqdn}"),
        }
    }
}

impl JavaType {
    pub fn from_type_ref(ctx: &(impl ToJavaResolver + ?Sized), value: &TypeRef) -> Self {
        match value {
            TypeRef::Primitive(p) => match p {
                PrimitiveType::Bool => JavaType::Boolean,
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
        let empty = BTreeMap::default();
        while let Some(variant) = permitted.next() {
            write!(f, "\n{}", WithMethods(variant, &empty))?;
        }
        if !methods.is_empty() {
            write!(f, "\n\n")?;
            for fun in methods.values() {
                write!(f, "{fun}")?;
            }
        };
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
    body: Option<JavaBody>,
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
        let mut params = parameters.iter();
        let mut is_first = true;
        while let Some(JavaFunctionParameter { name, fqdn }) = params.next() {
            if is_first {
                is_first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{fqdn} {name}")?;
        }

        let body = if let Some(body) = body {
            write!(f, ") {{\n")?;
            body
        } else {
            write!(f, ");\n")?;
            return Ok(());
        };

        fn get(f: &mut Formatter<'_>, body: &JavaBody, scope: &mut JavaPrintScope, v: &JavaValueRef) -> Result<Rc<String>, core::fmt::Error> {
            if let Some(known) = scope.get(v) {
                return Ok(known.clone());
            }
            let kind = v.typ();
            let op = body.op(v);
            let indent = "  ".repeat(2 + body.depth());
            if op.is_some() && op.unwrap().should_store_in_var() {
                let id = Loc::from(v);
                let expr = print_rec(f, body, scope, op.unwrap())?;
                let var_name = Rc::new(format!("_{}_{}{}", id.depth, if id.p { "p" } else { "" }, id.id));
                write!(f, "{indent}final {kind} {var_name} = {expr};\n")?;
                scope.bind(v, var_name.clone()).map_err(|_| core::fmt::Error)?;
                Ok(var_name)
            } else if let Some(JavaOpN::If(e, tb, fb)) = op {
                let e = get(f, body, scope, e)?;
                let id = Loc::from(v);
                let var_name = Rc::new(format!("_if_{}_{}{}", id.depth, if id.p { "p" } else { "" }, id.id));
                write!(f, "{indent}final {kind} {var_name};\n")?;
                write!(f, "{indent}if ({e}) {{\n")?;
                print_nested_body(f, &indent, tb, scope, &var_name)?;
                write!(f, "{indent}}} else {{\n")?;
                print_nested_body(f, &indent, fb, scope, &var_name)?;
                write!(f, "{indent}}}\n")?;
                scope.bind(v, var_name.clone()).map_err(|_| core::fmt::Error)?;
                Ok(var_name)
            } else if let Some(JavaOpN::VarDef(k)) = op {
                let id = Loc::from(v);
                let var_name = Rc::new(format!("_var_{}_{}{}", id.depth, if id.p { "p" } else { "" }, id.id));
                let def = k.default_value();
                write!(f, "{indent}{k} {var_name} = {def};\n")?;
                scope.bind(v, var_name.clone()).map_err(|_| core::fmt::Error)?;
                Ok(var_name)
            } else if let Some(JavaOpN::VarSet(var, val)) = op {
                let var_name = get(f, body, scope, var)?;
                let val = get(f, body, scope, val)?;
                write!(f, "{indent}{var_name} = {val};\n")?;
                let placeholder = Rc::new("()".to_string());
                scope.bind(v, placeholder.clone()).map_err(|_| core::fmt::Error)?;
                Ok(placeholder)
            } else if let Some(JavaOpN::Unreachable(msg)) = op {
                write!(f, "{indent}assert false : \"{msg}\"\n")?;
                let val = Rc::new("null".to_string());
                scope.bind(v, val.clone()).map_err(|_| core::fmt::Error)?;
                Ok(val)
            } else if op.is_some() {
                let val = print_rec(f, body, scope, op.unwrap())?;
                scope.bind(v, val.clone()).map_err(|_| core::fmt::Error)?;
                Ok(val)
            } else {
                panic!("no Op to write! Cannot resolve {v:?} in {scope:?}: {:?} (body: {body:#?})", scope.get(v));
            }
        }

        fn print_nested_body(f: &mut Formatter, indent: &str, body: &JavaBody, scope: &mut JavaPrintScope, var_name: &Rc<String>) -> Result<(), Error> {
            let scope = &mut scope.nested([]);
            for (v, _) in body.entries() {
                let _ = get(f, body, scope, &v)?;
            }
            match body.terminator_op() {
                JavaTerminatorOpN::Return => {} // nothing to print
                JavaTerminatorOpN::Unreachable(msg) => {
                    write!(f, "{indent}  throw new AssertionError(\"UNREACHABLE: {msg}\");\n")?;
                }
                JavaTerminatorOpN::ReturnValue(v) => {
                    let ne = get(f, body, scope, v)?;
                    write!(f, "{indent}  {var_name} = {ne};\n")?;
                }
            }
            Ok(())
        }

        fn print_rec(f: &mut Formatter<'_>, body: &JavaBody, scope: &mut JavaPrintScope, op: &JavaOpN) -> Result<Rc<String>, core::fmt::Error> {
            match op {
                JavaOpN::ConstNull => Ok(Rc::new("null".to_string())),
                JavaOpN::ConstString(v) => Ok(Rc::new(format!("\"{v}\""))),
                JavaOpN::ConstBool(v) => Ok(Rc::new(if *v { "true" } else { "false" }.to_string())),
                JavaOpN::ConstShort(n) => Ok(Rc::new(format!("{n}"))),
                JavaOpN::ConstLong(n) => Ok(Rc::new(format!("{n}"))),
                JavaOpN::Add(l, r) => {
                    let l = get(f, body, scope, l)?;
                    let r = get(f, body, scope, r)?;
                    Ok(Rc::new(format!("{l} + {r}")))
                }
                JavaOpN::Gt(l, r) => {
                    let l = get(f, body, scope, l)?;
                    let r = get(f, body, scope, r)?;
                    Ok(Rc::new(format!("{l} > {r}")))
                }
                JavaOpN::Eq(l, r) => {
                    let l = get(f, body, scope, l)?;
                    let r = get(f, body, scope, r)?;
                    Ok(Rc::new(format!("{l} == {r}")))
                }
                JavaOpN::And(l, r) => {
                    let l = get(f, body, scope, l)?;
                    let r = get(f, body, scope, r)?;
                    Ok(Rc::new(format!("{l} && {r}")))
                }
                JavaOpN::Or(l, r) => {
                    let l = get(f, body, scope, l)?;
                    let r = get(f, body, scope, r)?;
                    Ok(Rc::new(format!("{l} || {r}")))
                }
                JavaOpN::Create((_, s, ..), p) => {
                    let p = p
                        .iter()
                        .map(|v| get(f, body, scope, v).map(|s| s.to_string())) // FIXME
                        .collect::<Result<Vec<_>, core::fmt::Error>>()?
                        .join(", ");
                    Ok(Rc::new(format!("new {s}({p})")))
                }
                JavaOpN::Neg(v) => {
                    let v = get(f, body, scope, v)?;
                    Ok(Rc::new(format!("-{v}")))
                }
                JavaOpN::Nop(v) => get(f, body, scope, v),
                JavaOpN::GetParam(_, n) => Ok(n.clone()),
                JavaOpN::GetTupleField(e, _, fi) => {
                    let e = get(f, body, scope, e)?;
                    Ok(Rc::new(format!("{e}._{fi}()")))
                }
                JavaOpN::GetRecordField(e, _, fi) => {
                    let e = get(f, body, scope, e)?;
                    Ok(Rc::new(format!("{e}.{fi}()")))
                }
                JavaOpN::InvokeStatic((_, s, ..), _, n, a) => {
                    let a = a
                        .iter()
                        .map(|v| get(f, body, scope, v).map(|s| s.to_string())) // FIXME
                        .collect::<Result<Vec<_>, core::fmt::Error>>()?
                        .join(", ");
                    Ok(Rc::new(format!("{s}.{n}({a})")))
                }
                JavaOpN::Error(msg) => Ok(Rc::new(format!("/* TODO {msg} */"))),
                JavaOpN::Unreachable(msg) => Ok(Rc::new(format!("null /* UNREACHABLE {msg} */"))),
                JavaOpN::If(_, _, _) => Ok(Rc::new("()".into())),
                JavaOpN::InstanceOf(e, (_, s, ..)) => {
                    let e = get(f, body, scope, e)?;
                    Ok(Rc::new(format!("{e} instanceof {s}")))
                }
                JavaOpN::Cast(e, k) => {
                    let e = get(f, body, scope, e)?;
                    Ok(Rc::new(format!("(({}) {e})", k.1)))
                }
                JavaOpN::VarDef(..) => Ok(Rc::new("()".into())),
                JavaOpN::VarSet(..) => Ok(Rc::new("()".into())),
                JavaOpN::VarGet(v) => Ok(get(f, body, scope, v)?),
            }
        }

        match body.terminator_op() {
            JavaTerminatorOpN::Return => {}
            JavaTerminatorOpN::Unreachable(msg) => {
                write!(f, "    throw new AssertionError(\"UNREACHABLE: {msg}\");\n")?;
            }
            JavaTerminatorOpN::ReturnValue(v) => {
                let mut scopes = JavaPrintScopes::new();
                let params = parameters
                    .iter()
                    .map(|p| Rc::new(p.name.clone()))
                    .collect::<Vec<_>>();
                let mut scope = scopes.root(params);
                for (v, _) in body.entries() {
                    let _ = get(f, body, &mut scope, &v)?;
                }
                let expr = get(f, body, &mut scope, v)?;
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
                Rc::new(format!("{root_pkg}.{}", s.name())),
                None
            );
            builder.pending_symbols.push((java_ref.1.clone(), match s {
                Symbol::Struct(v) => v.into(),
                Symbol::Enum(v) => {
                    for v in v.variants() {
                        let variant_ref: JavaSymbolRef = (
                            java_ref.0.clone(),
                            Rc::new(format!("{}.{}", java_ref.1, v.name())),
                            Some(v.name().to_string())
                        );
                        builder.by_fqdn.insert(variant_ref.1.clone(), variant_ref);
                    }
                    v.into()
                }
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
            Rc::new(format!("{root_pkg}.Utils")),
            None
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
        let mut pending_bodies = vec![];
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
                body: None,
            };

            let sym_ref = match (&function.ret_type, function.parameters.as_slice()) {
                (JavaType::Class(sym), _) => sym,
                (_, [JavaFunctionParameter { fqdn: JavaType::Class(sym), .. }]) => sym,
                _ => &utils_sym
            }.clone();
            let Some((sid, sym)) = module.by_fqdn
                .get(sym_ref.1.as_str())
                .cloned()
                .and_then(|i| module.symbols.get_mut(i).map(|sr| (i, sr))) else { continue; };
            sym.methods.insert(id.id(), function);
            pending_bodies.push((sid, id.id(), body));
            builder.fn_to_container.insert(id.clone(), sym_ref);
        }
        for (sid, mid, body) in pending_bodies {
            let body = TryFrom::try_from((body, &builder, &module))?;
            module
                .symbols
                .get_mut(sid)
                .and_then(|s| s.methods.get_mut(&mid))
                .map(|m| m.body = Some(body))
                .ok_or("couldn't attach body to method")?
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

    pub fn resolve_enum_variant(&self, fqdn: &str, variant: &str) -> Option<JavaSymbolRef> {
        let id = self.by_fqdn.get(fqdn)?;
        match self.symbols.get(*id) {
            Some(JavaSymbol { kind: JavaSymbolKind::SealedInterface(si), .. }) if si.permitted.iter().find(|v| variant == v.name).is_some() => Some((
                *id,
                format!("{fqdn}.{variant}").into(),
                Some(variant.to_string())
            )),
            _ => None
        }
    }

    pub fn resolve_record_field(&self, sr: &JavaSymbolRef, name: &str) -> Option<&JavaField> {
        let sym = self.symbols.get(sr.0)?;
        match sym {
            JavaSymbol { kind: JavaSymbolKind::Record(rec), .. } => rec.fields.iter().find(|f| *name == *f.name),
            JavaSymbol { kind: JavaSymbolKind::SealedInterface(si), .. } => if let Some(vn) = &sr.2 {
                if let Some(v) = si.permitted.iter().find(|nr| *vn == *nr.name) {
                    v.fields.iter().find(|f| *name == *f.name)
                } else {
                    None
                }
            } else {
                None
            },
            _ => None
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum JavaOpN {
    Error(Cow<'static, str>),
    Unreachable(Cow<'static, str>),
    Nop(JavaValueRef),
    VarDef(JavaValueKind),
    VarSet(JavaValueRef, JavaValueRef),
    VarGet(JavaValueRef),
    Neg(JavaValueRef),
    ConstNull,
    ConstString(String),
    ConstBool(bool),
    ConstShort(i16),
    ConstLong(i64),
    Cast(JavaValueRef, JavaSymbolRef),
    GetParam(JavaValueKind, Rc<String>),
    GetTupleField(JavaValueRef, JavaValueKind, usize),
    GetRecordField(JavaValueRef, JavaValueKind, String),
    InvokeStatic(JavaSymbolRef, JavaValueKind, Rc<String>, Vec<JavaValueRef>),
    Create(JavaSymbolRef, Vec<JavaValueRef>),
    Add(JavaValueRef, JavaValueRef),
    Gt(JavaValueRef, JavaValueRef),
    Eq(JavaValueRef, JavaValueRef),
    And(JavaValueRef, JavaValueRef),
    Or(JavaValueRef, JavaValueRef),
    InstanceOf(JavaValueRef, JavaSymbolRef),
    If(JavaValueRef, JavaBody, JavaBody),
}

#[derive(Debug, Eq, PartialEq)]
enum JavaTerminatorOpN {
    Return,
    ReturnValue(JavaValueRef),
    Unreachable(Cow<'static, str>),
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
    OpaqueVar(Rc<JavaValueKind>),
}

impl JavaValueKind {
    pub fn as_value_type(&self) -> &JavaValueKind {
        match self {
            JavaValueKind::OpaqueVar(v) => v.deref(),
            o => o
        }
    }

    pub fn default_value(&self) -> &str {
        match self {
            JavaValueKind::Never => "null",
            JavaValueKind::Void => "null",
            JavaValueKind::JavaType(typ) => match typ {
                JavaType::Void => "null",
                JavaType::Boolean => "false",
                JavaType::Byte => "(byte) 0",
                JavaType::Short => "(short) 0",
                JavaType::Int => "0",
                JavaType::Long => "0l",
                JavaType::Float => "0f",
                JavaType::Double => "0d",
                JavaType::Class(_) => "null",
            }
            JavaValueKind::OpaqueVar(_) => "null /* VAR !? */",
        }
    }
}

impl JavaOpN {
    fn should_store_in_var(&self) -> bool {
        !matches!(self,
            JavaOpN::Nop(..) |
            JavaOpN::If(..) |
            JavaOpN::VarDef(..) |
            JavaOpN::VarSet(..) |
            JavaOpN::VarGet(..) |
            JavaOpN::Unreachable(..) |
            JavaOpN::GetParam(..) |
            JavaOpN::ConstLong(..) |
            JavaOpN::ConstShort(..) |
            JavaOpN::ConstBool(..) |
            JavaOpN::And(..) |
            JavaOpN::Or(..) |
            JavaOpN::Eq(..) |
            JavaOpN::InstanceOf(..)
        )
    }
}

impl Typed for JavaOpN {
    type ValueType = JavaValueKind;

    fn typ(&self) -> Self::ValueType {
        fn bin_op_typ<T: Typed<ValueType=JavaValueKind>>(l: &T, r: &T) -> JavaValueKind {
            let l = l.typ().as_value_type().clone();
            let r = r.typ();
            if matches!(l, JavaValueKind::Never) {
                r
            } else if matches!(r, JavaValueKind::Never) {
                l
            } else if l == *r.as_value_type() {
                l
            } else {
                let (l, r) = match (l, r) {
                    (JavaValueKind::JavaType(l), JavaValueKind::JavaType(r)) => (l, r),
                    _ => return JavaValueKind::Never
                };
                JavaValueKind::JavaType(match (l, r) {
                    (JavaType::Byte, JavaType::Short) | (JavaType::Short, JavaType::Byte) => JavaType::Short,
                    (JavaType::Byte, JavaType::Int) | (JavaType::Int, JavaType::Byte) => JavaType::Int,
                    (JavaType::Byte, JavaType::Long) | (JavaType::Long, JavaType::Byte) => JavaType::Long,
                    (JavaType::Byte, JavaType::Float) | (JavaType::Float, JavaType::Byte) => JavaType::Float,
                    (JavaType::Byte, JavaType::Double) | (JavaType::Double, JavaType::Byte) => JavaType::Double,

                    (JavaType::Short, JavaType::Int) | (JavaType::Int, JavaType::Short) => JavaType::Int,
                    (JavaType::Short, JavaType::Long) | (JavaType::Long, JavaType::Short) => JavaType::Long,
                    (JavaType::Short, JavaType::Float) | (JavaType::Float, JavaType::Short) => JavaType::Float,
                    (JavaType::Short, JavaType::Double) | (JavaType::Double, JavaType::Short) => JavaType::Double,

                    (JavaType::Int, JavaType::Long) | (JavaType::Long, JavaType::Int) => JavaType::Long,
                    (JavaType::Int, JavaType::Double) | (JavaType::Double, JavaType::Int) => JavaType::Double,

                    (JavaType::Long, JavaType::Double) | (JavaType::Double, JavaType::Long) => JavaType::Double,

                    _ => return JavaValueKind::Never
                })
            }
        }

        match self {
            JavaOpN::Add(l, r) => bin_op_typ(l, r),
            JavaOpN::Gt(_, _) => JavaValueKind::JavaType(JavaType::Boolean),
            JavaOpN::Eq(_, _) => JavaValueKind::JavaType(JavaType::Boolean),
            JavaOpN::And(_, _) => JavaValueKind::JavaType(JavaType::Boolean),
            JavaOpN::Or(_, _) => JavaValueKind::JavaType(JavaType::Boolean),
            JavaOpN::InstanceOf(_, _) => JavaValueKind::JavaType(JavaType::Boolean),
            JavaOpN::ConstNull => JavaValueKind::Void,
            JavaOpN::ConstString(_) => JavaValueKind::JavaType(JavaType::Class((usize::MAX, "java.lang.String".to_string().into(), None))),
            JavaOpN::ConstBool(_) => JavaValueKind::JavaType(JavaType::Boolean),
            JavaOpN::ConstLong(_) => JavaValueKind::JavaType(JavaType::Long),
            JavaOpN::ConstShort(_) => JavaValueKind::JavaType(JavaType::Short),
            JavaOpN::Create(s, _) => JavaValueKind::JavaType(JavaType::Class(s.clone())),
            JavaOpN::Cast(_, s) => JavaValueKind::JavaType(JavaType::Class(s.clone())),
            JavaOpN::Error(_) => JavaValueKind::Never,
            JavaOpN::Unreachable(_) => JavaValueKind::Never,
            JavaOpN::GetParam(k, _) => k.clone(),
            JavaOpN::GetTupleField(_, k, _) => k.clone(),
            JavaOpN::GetRecordField(_, k, _) => k.clone(),
            JavaOpN::If(_, t, f) => bin_op_typ(t, f),
            JavaOpN::InvokeStatic(_, k, _, _) => k.clone(),
            JavaOpN::Neg(v) => v.typ(),
            JavaOpN::Nop(v) => v.typ(),
            JavaOpN::VarDef(k) => JavaValueKind::OpaqueVar(Rc::from(k.clone())),
            JavaOpN::VarSet(_, _) => JavaValueKind::Void,
            JavaOpN::VarGet(vr) => vr.typ().as_value_type().clone(),
        }
    }
}

impl Typed for JavaTerminatorOpN {
    type ValueType = JavaValueKind;

    fn typ(&self) -> Self::ValueType {
        match self {
            JavaTerminatorOpN::Return => JavaValueKind::Void,
            JavaTerminatorOpN::ReturnValue(v) => v.typ(),
            JavaTerminatorOpN::Unreachable(_) => JavaValueKind::Never
        }
    }
}

impl Display for JavaValueKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            JavaValueKind::Never => write!(f, "var /* ??? */"),
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
            JavaValueKind::OpaqueVar(v) => Display::fmt(v, f),
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

type JavaScopes<'a> = Scopes<JavaValueRef>;
type JavaScope<'a> = Scope<'a, JavaValueRef>;

type JavaPrintScopes<'a> = Scopes<Rc<String>>;
type JavaPrintScope<'a> = Scope<'a, Rc<String>>;

impl<'a> TryFrom<(&'a SkBody, &'a JavaModuleBuilder<'_>, &'a JavaModule)> for JavaBody {
    type Error = Cow<'static, str>;

    fn try_from((pb, builder, module): (&'a SkBody, &'a JavaModuleBuilder, &'a JavaModule)) -> Result<Self, Self::Error> {
        let params = pb.params()
            .iter()
            .map(|x| builder.convert_kind(x)) // TODO
            .collect::<Vec<_>>();
        Ok(Self::isolated(params.as_slice(), |b, args| {
            let mut scopes = JavaScopes::new();
            let scope = scopes.root(args.as_slice());
            Self::visit_body(pb, builder, module, b, scope)
        })?)
    }
}

impl JavaBody {
    fn visit_body(
        pb: &SkBody,
        builder: &JavaModuleBuilder,
        module: &JavaModule,
        b: &mut BlockBuilder<JavaValueKind, JavaOpN, JavaTerminatorOpN>,
        mut scope: JavaScope,
    ) -> Result<JavaTerminatorOpN, Cow<'static, str>> {
        for (rv, op) in pb.entries() {
            let jop = Self::visit_op(builder, module, b, &mut scope, op)?;
            let r = b.op(jop);
            scope.bind(&rv, r)?;
        }
        Ok(match pb.terminator_op() {
            SkTerminatorOp::Yield(v) if matches!(v.typ(), SkValueKind::Unit) => JavaTerminatorOpN::Return,
            SkTerminatorOp::Yield(v) => JavaTerminatorOpN::ReturnValue(match scope.get(v) {
                None => return Err("no mapping found for SkTerminatorOp::Yield arg")?,
                Some(v) => v.clone()
            }),
        })
    }

    fn visit_nested_body(
        builder: &JavaModuleBuilder,
        module: &JavaModule,
        b: &mut BlockBuilder<JavaValueKind, JavaOpN, JavaTerminatorOpN>,
        scope: &mut JavaScope,
        tb: &SkBody,
        parameters: impl Into<Vec<JavaValueKind>>,
    ) -> Result<JavaBody, Cow<'static, str>> {
        b.body(parameters, |bb, params| {
            Self::visit_body(tb, builder, module, bb, scope.nested(params.as_slice()))
        })
    }

    fn visit_op(
        builder: &JavaModuleBuilder,
        module: &JavaModule,
        b: &mut BlockBuilder<JavaValueKind, JavaOpN, JavaTerminatorOpN>,
        mut scope: &mut JavaScope,
        op: &SkOp,
    ) -> Result<JavaOpN, Cow<'static, str>> {
        Ok(match op {
            SkOp::Label(v, _) => match scope.get(v).cloned() {
                None => JavaOpN::Error("no mapping found for SkOp::Label arg".into()),
                // None => panic!("no mapping found for SkOp::Label arg: {v:?} in {scope:?}"),
                Some(v) => JavaOpN::Nop(v.clone())
            },
            SkOp::ConstUnit => JavaOpN::Error("SkOp::ConstUnit conversion not implemented".into()),
            SkOp::ConstI64(n) if *n < i16::MAX as i64 => JavaOpN::ConstShort(n.clone() as i16),
            SkOp::ConstBool(v) => JavaOpN::ConstBool(*v),
            SkOp::ConstI64(n) => JavaOpN::ConstLong(n.clone()),
            SkOp::Block(_) => JavaOpN::Error("SkOp::Block conversion not implemented".into()), // TODO
            SkOp::Neg(v) => match v.id() {
                RefId::Param(_) => JavaOpN::Error("SkOp::Neg arg cannot be an RefId::Param".into()),
                RefId::Op(_) => match scope.get(v).cloned() {
                    None => JavaOpN::Error("no mapping found for SkOp::Neg arg".into()),
                    Some(v) => JavaOpN::Neg(v),
                }
            }
            SkOp::Add(l, r) => match (l.id(), r.id()) {
                (RefId::Param(_), _) => JavaOpN::Error("SkOp::Add left arg cannot be an RefId::Param".into()),
                (_, RefId::Param(_)) => JavaOpN::Error("SkOp::Add right arg cannot be an RefId::Param".into()),
                (RefId::Op(_), RefId::Op(_)) => match (scope.get(l).cloned(), scope.get(r).cloned()) {
                    (None, _) => JavaOpN::Error("no mapping found for SkOp::Add left arg".into()),
                    (_, None) => JavaOpN::Error("no mapping found for SkOp::Add right arg".into()),
                    (Some(l), Some(r)) => JavaOpN::Add(l, r),
                }
            }
            SkOp::Mul(_, _) => JavaOpN::Error("SkOp::ConstUnit conversion not implemented".into()), // TODO
            SkOp::Gt(l, r) => match (l.id(), r.id()) {
                (RefId::Param(_), _) => JavaOpN::Error("SkOp::Gt left arg cannot be an RefId::Param".into()),
                (_, RefId::Param(_)) => JavaOpN::Error("SkOp::Gt right arg cannot be an RefId::Param".into()),
                (RefId::Op(_), RefId::Op(_)) => match (scope.get(l).cloned(), scope.get(r).cloned()) {
                    (None, _) => JavaOpN::Error("no mapping found for SkOp::Gt left arg".into()),
                    (_, None) => JavaOpN::Error("no mapping found for SkOp::Gt right arg".into()),
                    (Some(l), Some(r)) => JavaOpN::Gt(l, r),
                }
            }
            SkOp::If(e, tb, fb) => match e.id() {
                RefId::Param(_) => JavaOpN::Error("SkOp::If condition arg cannot be an RefId::Param".into()),
                RefId::Op(_) => match scope.get(e).cloned() {
                    None => JavaOpN::Error("no mapping found for SkOp::If condition arg".into()),
                    Some(e) => {
                        let tb: JavaBody = Self::visit_nested_body(builder, module, b, scope, tb, [])?;
                        let fb: JavaBody = Self::visit_nested_body(builder, module, b, scope, fb, [])?;
                        JavaOpN::If(e, tb, fb)
                    }
                }
            },
            SkOp::Call(s, _, a) => {
                let id = s.id();
                match builder.get_fn_container(s) {
                    None => JavaOpN::Error(format!("function #{s} not found in SkOp::Call").into()),
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
                                        RefId::Op(_) => scope
                                            .get(v)
                                            .cloned()
                                            .map(Ok)
                                            .unwrap_or_else(|| Err(format!("no mapping found for SkOp::Call arg {}", i + 1).into()))
                                    })
                                    .collect::<Result<Vec<_>, Cow<'static, str>>>()?;
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
                                    RefId::Op(_) => match scope.get(v) {
                                        None => return Err(format!("no mapping found for SkOp::Create arg {}", i + 1))?,
                                        Some(v) => Ok(v.clone())
                                    }
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map(|params| JavaOpN::Create(s.clone(), params))
                            .unwrap_or_else(|err| JavaOpN::Error(err)),
                        _ => JavaOpN::Error("only Records can be constructed in SkOp::Create".into()),
                    },
                }
            }
            SkOp::Match(e, p) => match e.id() {
                RefId::Param(_) => JavaOpN::Error("SkOp::Match expression arg cannot be an RefId::Param".into()),
                RefId::Op(_) => match scope.get(e).cloned() {
                    None => JavaOpN::Error("no mapping found for SkOp::Match expression arg".into()),
                    Some(e) => {
                        let mut iter = p.iter();
                        if let Some(first) = iter.next() {
                            Self::visit_match_case(builder, module, b, scope, first, iter, e)?
                        } else {
                            JavaOpN::Error("No case in SkOp::Match!".into())
                        }
                    }
                }
            },
        })
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
      width: times2(w), // FIXME incompatible cast should be detected
      height: h,
    },
  }
}

fn times2(v: i64) -> i64 {
  v + if v > 0 { v + 1000000 } else { 4000000 + 600000 }
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
    final short _0_3 = -1;
    final short _0_4 = y + _0_3;
    final my_first_module.Point _0_5 = new my_first_module.Point(x, _0_4);
    return _0_5;
  }

}"#.to_string())
    );

    assert_eq!(
        java.resolve("my_first_module.Utils").map(ToString::to_string),
        /*language=java*/Some(r#"package my_first_module;

public final class Utils {

  static long times2(long v) {
    final boolean _0_2 = v > 0;
    final long _if_0_3;
    if (_0_2) {
      final long _1_1 = v + 1000000;
      _if_0_3 = _1_1;
    } else {
      final long _1_2 = 4000000 + 600000;
      _if_0_3 = _1_2;
    }
    final long _0_4 = v + _if_0_3;
    return _0_4;
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
    final my_first_module.Point _0_4 = my_first_module.Point.new_point(x, y);
    final long _0_5 = my_first_module.Utils.times2(w);
    final my_first_module.Size _0_6 = new my_first_module.Size(_0_5, h);
    final my_first_module.Rectangle _0_7 = new my_first_module.Rectangle(_0_4, _0_6);
    return _0_7;
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

pub fn is_priced(maybe_price: Some) -> bool {
  match maybe_price {
    Some(Price::Limit | Price::StopLimit) => true,
    _ => false
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

  public static boolean is_priced(skull_test_transform_enum.Some maybe_price) {
    final boolean _if_0_2;
    if (maybe_price instanceof skull_test_transform_enum.Some) {
      final skull_test_transform_enum.Some _1_0 = ((skull_test_transform_enum.Some) maybe_price);
      final skull_test_transform_enum.Price _1_1 = _1_0._0();
      _if_0_2 = _1_1 instanceof skull_test_transform_enum.Price.Limit || _1_1 instanceof skull_test_transform_enum.Price.StopLimit;
    } else {
      _if_0_2 = false;
    }
    final boolean _if_0_3;
    if (_if_0_2) {
      _if_0_3 = true;
    } else {
      _if_0_3 = false;
    }
    return _if_0_3;
  }

}"#.to_string())
    );

    Ok(())
}
