use alloc::{format, vec};
use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{Display, Formatter};
use core::ops::Deref;

use crate::bytecode::{AnyBody, AnyValueRef, Body, BodyN, Op, OpN, TerminatorOp, TerminatorOpN, ValueKind};
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

    fn convert_kind(&self, typ: &ValueKind) -> JavaValueKind {
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
struct JavaFunction {
    doc: Option<Rc<str>>,
    visibility: JavaVisibility,
    name: String,
    parameters: Vec<JavaFunctionParameter>,
    ret_type: JavaType,
    body: JavaBodyN,
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
        if let op = body.terminator_op() {
            fn get(f: &mut Formatter<'_>, loc: &mut BTreeMap<usize, Rc<String>>, body: &JavaBodyN, v: &JavaValueRefN) -> Result<Rc<String>, core::fmt::Error> {
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

            fn print_rec(f: &mut Formatter<'_>, loc: &mut BTreeMap<usize, Rc<String>>, body: &JavaBodyN, op: &JavaOpN, kind: &JavaValueKind) -> Result<Rc<String>, core::fmt::Error> {
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
                }
            }

            match op {
                JavaTerminatorOpN::Return => {}
                JavaTerminatorOpN::ReturnValue(v) => {
                    let mut loc = BTreeMap::new();
                    let expr = get(f, &mut loc, body, v)?;
                    write!(f, "    return {expr};\n")?;
                }
            }
        } else {
            write!(f, "    return null; // TODO: functions body\n")?;
        }
        write!(f, "  }}\n")?;
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
enum JavaSymbolKind {
    Record(JavaRecord),
    SealedInterface(JavaSealedInterface),
}

impl<'a> Display for WithMethods<'a, JavaSymbolKind> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self(sym, methods) = self;
        match sym {
            JavaSymbolKind::Record(r) => Display::fmt(&WithMethods(r, methods), f),
            JavaSymbolKind::SealedInterface(si) => Display::fmt(&WithMethods(si, methods), f),
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
                Rc::new(format!("{}.{}", root_pkg, s.name()))
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
        for (id, def) in &builder.pending_functions {
            let Some(body) = def.body_n() else { continue; };

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

            let JavaType::Class(sym_ref) = ret_type else { continue; };
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
enum JavaOp {
    Nop(Rc<JavaValueRef>),
    Neg(Rc<JavaValueRef>),
    ConstShort(i16),
    ConstLong(i64),
    GetParam(Rc<String>),
    InvokeStatic(JavaSymbolRef, Rc<String>, Vec<Rc<JavaValueRef>>),
    Create(JavaSymbolRef, Vec<Rc<JavaValueRef>>),
    Add(Rc<JavaValueRef>, Rc<JavaValueRef>),
}

#[derive(Debug, Eq, PartialEq)]
enum JavaTerminatorOp {
    Return,
    ReturnValue(Rc<JavaValueRef>),
}

#[derive(Debug, Eq, PartialEq)]
enum JavaOpN {
    Nop(JavaValueRefN),
    Neg(JavaValueRefN),
    ConstShort(i16),
    ConstLong(i64),
    GetParam(JavaValueKind, Rc<String>),
    InvokeStatic(JavaSymbolRef, JavaValueKind, Rc<String>, Vec<JavaValueRefN>),
    Create(JavaSymbolRef, Vec<JavaValueRefN>),
    Add(JavaValueRefN, JavaValueRefN),
}

#[derive(Debug, Eq, PartialEq)]
enum JavaTerminatorOpN {
    Return,
    ReturnValue(JavaValueRefN),
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
                panic!("WOT: {l:?} != {r:?}")
                // JavaValueKind::Void
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
            JavaOpN::Add(l, r) => bin_op_typ(l, r)
            // JavaOpN::Add(l, r) => l.typ()
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

type JavaValueRef = AnyValueRef<JavaValueKind>;
type JavaValueRefN = RuntimeValue<JavaValueKind>;
type JavaBody = AnyBody<JavaOp, JavaTerminatorOp, JavaValueKind>;

type JavaBodyN = Body0<JavaValueKind, JavaOpN, JavaTerminatorOpN>;
type JavaBlockBuilderN<'a> = BlockBuilder<'a, ValueKind, JavaOpN, JavaTerminatorOpN>;

impl<'a> TryFrom<(&'a Body, &'a JavaModuleBuilder<'_>, &'a JavaModule)> for JavaBody {
    type Error = Cow<'static, str>;

    fn try_from((pb, builder, module): (&'a Body, &'a JavaModuleBuilder, &'a JavaModule)) -> Result<Self, Self::Error> {
        let params = pb.params
            .iter()
            .map(|x| JavaValueKind::Never)
            .collect::<Vec<_>>();
        Ok(Self::new_with_dyn_params(params.as_slice(), |b, args| {
            let mut mapping: BTreeMap<usize, Rc<JavaValueRef>> = BTreeMap::new();
            let def = b.push(JavaOp::ConstLong(-1), JavaType::Long.into())?;
            for (id, op) in pb.ops.iter().enumerate() {
                let (jop, kind) = match op {
                    Op::Label(v, n) => {
                        if v.is_param() {
                            let s = builder.convert_kind(v.kind());
                            (JavaOp::GetParam(Rc::new(n.clone())), JavaValueKind::Var(Rc::new((n.clone(), s))))
                        } else {
                            let Some(v) = mapping.get(&v.id()).cloned() else { continue; };
                            let k = v.kind().clone();
                            (JavaOp::Nop(v), k)
                        }
                    }
                    Op::ConstUnit => continue,
                    Op::ConstI64(n) if *n < i16::MAX as i64 => (JavaOp::ConstShort(n.clone() as i16), JavaType::Short.into()),
                    Op::ConstI64(n) => (JavaOp::ConstLong(n.clone()), JavaType::Long.into()),
                    Op::Block(_) => continue,
                    Op::Neg(v) => {
                        let Some(v) = mapping.get(&v.id()).cloned() else { continue; };
                        let k = v.kind().clone();
                        (JavaOp::Neg(v), k)
                    }
                    Op::Add(l, r) => {
                        let Some(l) = mapping.get(&l.id()).cloned() else { continue; };
                        let Some(r) = mapping.get(&r.id()).cloned() else { continue; };
                        let k = r.kind().clone();
                        (JavaOp::Add(l, r), k)
                    }
                    Op::Mul(_, _) => continue,
                    Op::Gt(_, _) => continue,
                    Op::If(_, _, _) => continue,
                    Op::Call(s, a) => {
                        let id = s.id();
                        let s = builder.get_fn_container(s).ok_or("function not found")?;
                        let m = module.resolve(s.1.as_str()).ok_or("container not found")?;
                        let f = m.methods.get(&id).ok_or("container doesn't contain function")?;
                        let a = a
                            .iter()
                            .map(|v| mapping.get(&v.id())
                                .cloned()
                                .unwrap_or_else(|| def.clone()))
                            .collect();
                        (JavaOp::InvokeStatic(s, Rc::new(f.name.clone()), a), f.ret_type.clone().into())
                    }
                    Op::Create(s, p) => {
                        let p = p.iter().map(|(n, v)| (n, v)).collect::<BTreeMap<_, _>>();
                        let s = builder.convert_refs(s);
                        let m = module.resolve(s.1.as_str()).ok_or("symbol not found")?;
                        let m = match m.kind {
                            JavaSymbolKind::Record(ref m) => m,
                            _ => return Err("only Records can be constructed")?
                        };
                        let mut params = vec![];
                        for f in &m.fields {
                            let Some(v) = p.get(&f.name) else { return Err("missing parameter")?; };
                            let v = mapping.get(&v.id()).cloned().unwrap_or_else(|| def.clone());
                            params.push(v);
                        }
                        (JavaOp::Create(s.clone(), params), JavaType::Class(s).into())
                    }
                    Op::Match(_, _) => continue,
                };
                let r = b.push(jop, kind)?;
                mapping.insert(id, r);
            }
            if let Some(TerminatorOp::Yield(ref v)) = pb.terminator_op {
                if let Some(v) = mapping.get(&v.id()).cloned() {
                    return Ok(JavaTerminatorOp::ReturnValue(v));
                }
            };
            Ok(JavaTerminatorOp::Return)
        })?)
    }
}

impl<'a> TryFrom<(&'a BodyN, &'a JavaModuleBuilder<'_>, &'a JavaModule)> for JavaBodyN {
    type Error = Cow<'static, str>;

    fn try_from((pb, builder, module): (&'a BodyN, &'a JavaModuleBuilder, &'a JavaModule)) -> Result<Self, Self::Error> {
        let params = pb.params()
            .iter()
            .map(|x| JavaValueKind::Never) // TODO
            .collect::<Vec<_>>();
        Ok(Self::isolated(params.as_slice(), |b, args| {
            let mut mapping: BTreeMap<usize, JavaValueRefN> = BTreeMap::new();
            let def = b.op(JavaOpN::ConstLong(-1));
            for (id, op) in pb.ops().iter().enumerate() {
                let jop = match op {
                    OpN::Label(v, n) => {
                        match v.id() {
                            RefId::Param(_) => {
                                let s = builder.convert_kind(&v.typ());
                                JavaOpN::GetParam(JavaValueKind::Var(Rc::new((n.clone(), s))), Rc::new(n.clone()))
                            }
                            RefId::Op(oi) => {
                                let Some(v) = mapping.get(&oi).cloned() else { return Err("MISSING VALUE")?; }; // FIXME
                                JavaOpN::Nop(v)
                            }
                        }
                    }
                    OpN::ConstUnit => continue,
                    OpN::ConstI64(n) if *n < i16::MAX as i64 => JavaOpN::ConstShort(n.clone() as i16),
                    OpN::ConstI64(n) => JavaOpN::ConstLong(n.clone()),
                    OpN::Block(_) => continue, // TODO
                    OpN::Neg(v) => {
                        let Some(v) = mapping.get(&match v.id() {
                            RefId::Param(_) => continue,
                            RefId::Op(oi) => oi
                        }).cloned() else { return Err("MISSING VALUE")?; }; // FIXME
                        JavaOpN::Neg(v)
                    }
                    OpN::Add(l, r) => {
                        let Some(l) = mapping.get(&match l.id() {
                            RefId::Param(_) => continue,
                            RefId::Op(oi) => oi
                        }).cloned() else { return Err("MISSING VALUE")?; }; // FIXME
                        let Some(r) = mapping.get(&match r.id() {
                            RefId::Param(_) => continue,
                            RefId::Op(oi) => oi
                        }).cloned() else { return Err("MISSING VALUE")?; }; // FIXME
                        JavaOpN::Add(l, r)
                    }
                    OpN::Mul(_, _) => continue, // TODO
                    OpN::Gt(_, _) => continue, // TODO
                    OpN::If(_, _, _) => continue, // TODO
                    OpN::Call(s, _, a) => {
                        let id = s.id();
                        let s = builder.get_fn_container(s).ok_or("function not found")?;
                        let m = module.resolve(s.1.as_str()).ok_or("container not found")?;
                        let f = m.methods.get(&id).ok_or("container doesn't contain function")?;
                        let a = a
                            .iter()
                            .map(|v| match v.id() {
                                RefId::Param(_) => None,
                                RefId::Op(oi) => Some(oi)
                            }.and_then(|oi| mapping.get(&oi))
                                .cloned()
                                .unwrap_or_else(|| def.clone()))
                            .collect();
                        JavaOpN::InvokeStatic(
                            s,
                            f.ret_type.clone().into(),
                            Rc::new(f.name.clone()),
                            a,
                        )
                    }
                    OpN::Create(s, p) => {
                        let p = p.iter().map(|(n, v)| (n, v)).collect::<BTreeMap<_, _>>();
                        let s = builder.convert_refs(s);
                        let m = module.resolve(s.1.as_str()).ok_or("symbol not found")?;
                        let m = match m.kind {
                            JavaSymbolKind::Record(ref m) => m,
                            _ => return Err("only Records can be constructed")?
                        };
                        let mut params = vec![];
                        for f in &m.fields {
                            let Some(v) = p.get(&f.name) else { return Err("missing parameter")?; };
                            let v = match v.id() {
                                RefId::Param(_) => None,
                                RefId::Op(oi) => Some(oi)
                            }.and_then(|oi| mapping.get(&oi)).cloned().unwrap_or_else(|| def.clone());
                            params.push(v);
                        }
                        JavaOpN::Create(s.clone(), params)
                    }
                    OpN::Match(_, _) => continue, // TODO
                };
                let r = b.op(jop);
                mapping.insert(id, r);
            }
            if let TerminatorOpN::Yield(ref v) = pb.terminator_op() {
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
      width: w,
      height: h,
    },
  }
}

fn times2(v: i16) -> i16 {
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
    final my_first_module.Size _1 = new my_first_module.Size(w, h);
    final my_first_module.Rectangle _2 = new my_first_module.Rectangle(_0, _1);
    return _2;
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
) { }"#.to_string())
    );

    Ok(())
}
