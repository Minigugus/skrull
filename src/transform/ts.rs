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

type TsSymbolRef = (usize, Rc<String>, Option<String>);

trait ToTsResolver {
    fn convert_refs(&self, r: &SymbolRef) -> TsSymbolRef;
    fn get_fn_container(&self, r: &SymbolRef) -> Option<TsSymbolRef>;

    fn convert_types(&self, typ: &TypeRef) -> TsType {
        TsType::from_type_ref(self, typ)
    }

    fn convert_kind(&self, typ: &SkValueKind) -> TsValueKind {
        let kind = value_kind_to_type_ref(typ);
        match kind {
            Some(typ) => Self::convert_types(self, &typ).into(),
            None => TsValueKind::Never
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum TsVisibility {
    // Public,
    // Protected,
    // PackagePrivate,
    // Private,
    Exported,
    Local,
}

impl TsVisibility {
    pub fn from_type_scope(value: ScopeNode) -> Self {
        match value {
            ScopeNode::Private => TsVisibility::Local,
            ScopeNode::Public => TsVisibility::Exported,
        }
    }
}

impl Display for TsVisibility {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            TsVisibility::Exported => write!(f, "export "),
            TsVisibility::Local => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TsType {
    Undefined,
    Boolean,
    Number,
    String,
    Class(TsSymbolRef),
}

impl Display for TsType {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            TsType::Undefined => write!(f, "undefined"),
            TsType::Boolean => write!(f, "boolean"),
            TsType::Number => write!(f, "number"),
            TsType::String => write!(f, "string"),
            TsType::Class((_, fqdn, ..)) => write!(f, "{fqdn}"),
        }
    }
}

impl TsType {
    pub fn from_type_ref(ctx: &(impl ToTsResolver + ?Sized), value: &TypeRef) -> Self {
        match value {
            TypeRef::Primitive(p) => match p {
                PrimitiveType::Bool => TsType::Boolean,
                PrimitiveType::I16 => TsType::Number,
                PrimitiveType::U32 => TsType::Number,
                PrimitiveType::I64 => TsType::Number,
                PrimitiveType::F64 => TsType::Number,
                PrimitiveType::Unit => TsType::Undefined,
                PrimitiveType::Usize => TsType::Number,
            }
            TypeRef::Ref(r) => TsType::Class(ctx.convert_refs(r))
        }
    }
}

fn print_as_ts_doc(f: &mut Formatter, indent: &str, doc: &Option<Rc<str>>, params: Option<&[TsField]>) -> Result<(), core::fmt::Error> {
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
struct TsField {
    doc: Option<Rc<str>>,
    name: String,
    typ: TsType,
}

impl Display for TsField {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self {
            typ,
            name,
            ..
        } = self;
        write!(f, "{name}: {typ}")
    }
}

#[derive(Debug, Eq, PartialEq)]
struct TsInterface {
    doc: Option<Rc<str>>,
    visibility: TsVisibility,
    name: String,
    parent: Option<TsSymbolRef>,
    fields: Vec<TsField>,
}

struct WithMethods<'a, T>(&'a T, &'a BTreeMap<usize, TsFunction>);

impl<'a> Display for WithMethods<'a, TsInterface> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self(
            TsInterface {
                doc,
                visibility,
                name,
                parent,
                fields
            },
            methods
        ) = self;
        print_as_ts_doc(f, "", doc, None)?;
        write!(f, "{visibility}class ")?;
        if let Some((_, parent, _)) = parent {
            write!(f, "{parent}_")?;
        }
        write!(f, "{name} {{\n")?;
        print_as_ts_doc(f, "  ", &None, Some(fields.as_slice()))?;
        write!(f, "  public constructor(")?;
        let mut fields = fields.iter();
        if let Some(field) = fields.next() {
            write!(f, "\n    public readonly {field}")?;
            while let Some(field) = fields.next() {
                write!(f, ",\n    public readonly {field}")?;
            }
            write!(f, "\n  ) {{ }}")?;
        } else {
            write!(f, ") {{ }}")?;
        }
        write!(f, "\n}}\n\n")?;
        for fun in methods.values() {
            write!(f, "{fun}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
struct TsUnionType {
    doc: Option<Rc<str>>,
    visibility: TsVisibility,
    name: String,
    permitted: Vec<TsInterface>,
}

impl<'a> Display for WithMethods<'a, TsUnionType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self(
            TsUnionType {
                doc,
                visibility,
                name,
                permitted
            },
            methods
        ) = self;
        let empty = Default::default();
        let mut variants = permitted.iter();
        while let Some(variant) = variants.next() {
            write!(f, "{}", WithMethods(variant, &empty))?;
        }
        print_as_ts_doc(f, "", doc, None)?;
        write!(f, "{visibility}type {name} =")?;
        let mut variants = permitted.iter();
        while let Some(variant) = variants.next() {
            write!(f, "\n  | {name}_{}", variant.name)?;
        }
        write!(f, ";\n\n")?;
        if !methods.is_empty() {
            for fun in methods.values() {
                write!(f, "{fun}")?;
            }
        };
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
struct TsUtilClass {
    visibility: TsVisibility,
    name: String,
}

impl<'a> Display for WithMethods<'a, TsUtilClass> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self(
            _,
            // TsUtilClass {
            //     visibility,
            //     name,
            // },
            methods
        ) = self;
        // write!(f, "{visibility}class {name} {{\n\n")?;
        for fun in methods.values() {
            write!(f, "{fun}")?;
        }
        // write!(f, "\n}}")?;
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
struct TsFunction {
    doc: Option<Rc<str>>,
    visibility: TsVisibility,
    name: String,
    parameters: Vec<TsFunctionParameter>,
    ret_type: TsType,
    body: Option<TsBody>,
}

#[derive(Debug, Eq, PartialEq)]
struct TsFunctionParameter {
    name: String,
    fqdn: TsType,
}

impl Display for TsFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self {
            doc,
            visibility,
            name,
            parameters,
            ret_type,
            body
        } = self;
        print_as_ts_doc(f, "", doc, None)?;
        write!(f, "{visibility}function {name}(")?;
        let mut params = parameters.iter();
        let mut is_first = true;
        while let Some(TsFunctionParameter { name, fqdn }) = params.next() {
            if is_first {
                is_first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{name}: {fqdn}")?;
        }

        let body = if let Some(body) = body {
            write!(f, "): {ret_type} {{\n")?;
            body
        } else {
            write!(f, "): {ret_type};\n")?;
            return Ok(());
        };

        fn get(f: &mut Formatter<'_>, body: &TsBody, scope: &mut TsPrintScope, v: &TsValueRef) -> Result<Rc<String>, core::fmt::Error> {
            if let Some(known) = scope.get(v) {
                return Ok(known.clone());
            }
            let kind = v.typ();
            let op = body.op(v);
            let indent = "  ".repeat(1 + body.depth());
            if op.is_some() && op.unwrap().should_store_in_var() {
                let id = Loc::from(v);
                let expr = print_rec(f, body, scope, op.unwrap())?;
                let var_name = Rc::new(format!("_{}_{}{}", id.depth, if id.p { "p" } else { "" }, id.id));
                write!(f, "{indent}const {var_name}: {kind} = {expr};\n")?;
                scope.bind(v, var_name.clone()).map_err(|_| core::fmt::Error)?;
                Ok(var_name)
            } else if let Some(TsOp::If(e, tb, fb)) = op {
                let e = get(f, body, scope, e)?;
                let id = Loc::from(v);
                let var_name = Rc::new(format!("_if_{}_{}{}", id.depth, if id.p { "p" } else { "" }, id.id));
                write!(f, "{indent}let {var_name}: {kind};\n")?;
                write!(f, "{indent}if ({e}) {{\n")?;
                print_nested_body(f, &indent, tb, scope, &var_name)?;
                write!(f, "{indent}}} else {{\n")?;
                print_nested_body(f, &indent, fb, scope, &var_name)?;
                write!(f, "{indent}}}\n")?;
                scope.bind(v, var_name.clone()).map_err(|_| core::fmt::Error)?;
                Ok(var_name)
            } else if let Some(TsOp::VarDef(_)) = op {
                let id = Loc::from(v);
                let var_name = Rc::new(format!("_var_{}_{}{}", id.depth, if id.p { "p" } else { "" }, id.id));
                write!(f, "{indent}let {var_name}: {kind} = null!;\n")?;
                scope.bind(v, var_name.clone()).map_err(|_| core::fmt::Error)?;
                Ok(var_name)
            } else if let Some(TsOp::VarSet(var, val)) = op {
                let var_name = get(f, body, scope, var)?;
                let val = get(f, body, scope, val)?;
                write!(f, "{indent}{var_name} = {val};\n")?;
                let placeholder = Rc::new("()".to_string());
                scope.bind(v, placeholder.clone()).map_err(|_| core::fmt::Error)?;
                Ok(placeholder)
            } else if let Some(TsOp::Unreachable(msg)) = op {
                write!(f, "{indent}throw new Error(\"{msg}\");\n")?;
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

        fn print_nested_body(f: &mut Formatter, indent: &str, body: &TsBody, scope: &mut TsPrintScope, var_name: &Rc<String>) -> Result<(), Error> {
            let scope = &mut scope.nested([]);
            for (v, _) in body.entries() {
                let _ = get(f, body, scope, &v)?;
            }
            match body.terminator_op() {
                TsTerminatorOp::Return => {} // nothing to print
                TsTerminatorOp::Unreachable(msg) => {
                    write!(f, "{indent}  throw new Error(\"UNREACHABLE: {msg}\");\n")?;
                }
                TsTerminatorOp::ReturnValue(v) => {
                    let ne = get(f, body, scope, v)?;
                    write!(f, "{indent}  {var_name} = {ne};\n")?;
                }
            }
            Ok(())
        }

        fn print_rec(f: &mut Formatter<'_>, body: &TsBody, scope: &mut TsPrintScope, op: &TsOp) -> Result<Rc<String>, core::fmt::Error> {
            match op {
                TsOp::ConstNull => Ok(Rc::new("null".to_string())),
                TsOp::ConstString(v) => Ok(Rc::new(format!("\"{v}\""))),
                TsOp::ConstBool(v) => Ok(Rc::new(if *v { "true" } else { "false" }.to_string())),
                TsOp::ConstNumber(n) => Ok(Rc::new(format!("{n}"))),
                TsOp::Add(l, r) => {
                    let l = get(f, body, scope, l)?;
                    let r = get(f, body, scope, r)?;
                    Ok(Rc::new(format!("{l} + {r}")))
                }
                TsOp::Gt(l, r) => {
                    let l = get(f, body, scope, l)?;
                    let r = get(f, body, scope, r)?;
                    Ok(Rc::new(format!("{l} > {r}")))
                }
                TsOp::Eq(l, r) => {
                    let l = get(f, body, scope, l)?;
                    let r = get(f, body, scope, r)?;
                    Ok(Rc::new(format!("{l} == {r}")))
                }
                TsOp::And(l, r) => {
                    let l = get(f, body, scope, l)?;
                    let r = get(f, body, scope, r)?;
                    Ok(Rc::new(format!("{l} && {r}")))
                }
                TsOp::Or(l, r) => {
                    let l = get(f, body, scope, l)?;
                    let r = get(f, body, scope, r)?;
                    Ok(Rc::new(format!("{l} || {r}")))
                }
                TsOp::Create((_, s, ..), p) => {
                    let p = p
                        .iter()
                        .map(|v| get(f, body, scope, v).map(|s| s.to_string())) // FIXME
                        .collect::<Result<Vec<_>, core::fmt::Error>>()?
                        .join(", ");
                    Ok(Rc::new(format!("new {s}({p})")))
                }
                TsOp::Neg(v) => {
                    let v = get(f, body, scope, v)?;
                    Ok(Rc::new(format!("-{v}")))
                }
                TsOp::Nop(v) => get(f, body, scope, v),
                TsOp::GetParam(_, n) => Ok(n.clone()),
                TsOp::GetTupleField(e, _, fi) => {
                    let e = get(f, body, scope, e)?;
                    Ok(Rc::new(format!("{e}._{fi}")))
                }
                TsOp::GetRecordField(e, _, fi) => {
                    let e = get(f, body, scope, e)?;
                    Ok(Rc::new(format!("{e}.{fi}")))
                }
                TsOp::InvokeStatic((_, s, ..), _, n, a) => {
                    let a = a
                        .iter()
                        .map(|v| get(f, body, scope, v).map(|s| s.to_string())) // FIXME
                        .collect::<Result<Vec<_>, core::fmt::Error>>()?
                        .join(", ");
                    Ok(Rc::new(format!("{n}({a})")))
                }
                TsOp::Error(msg) => Ok(Rc::new(format!("/* TODO {msg} */"))),
                TsOp::Unreachable(msg) => Ok(Rc::new(format!("null /* UNREACHABLE {msg} */"))),
                TsOp::If(_, _, _) => Ok(Rc::new("()".into())),
                TsOp::InstanceOf(e, (_, s, ..)) => {
                    let e = get(f, body, scope, e)?;
                    Ok(Rc::new(format!("{e} instanceof {s}")))
                }
                TsOp::Cast(e, k) => {
                    let e = get(f, body, scope, e)?;
                    Ok(Rc::new(format!("({e}) as ({})", k.1)))
                }
                TsOp::VarDef(..) => Ok(Rc::new("()".into())),
                TsOp::VarSet(..) => Ok(Rc::new("()".into())),
                TsOp::VarGet(v) => Ok(get(f, body, scope, v)?),
            }
        }

        match body.terminator_op() {
            TsTerminatorOp::Return => {}
            TsTerminatorOp::Unreachable(msg) => {
                write!(f, "  throw new Error(\"UNREACHABLE: {msg}\");\n")?;
            }
            TsTerminatorOp::ReturnValue(v) => {
                let mut scopes = TsPrintScopes::new();
                let params = parameters
                    .iter()
                    .map(|p| Rc::new(p.name.clone()))
                    .collect::<Vec<_>>();
                let mut scope = scopes.root(params);
                for (v, _) in body.entries() {
                    let _ = get(f, body, &mut scope, &v)?;
                }
                let expr = get(f, body, &mut scope, v)?;
                write!(f, "  return {expr};\n")?;
            }
        }
        write!(f, "}}\n\n")?;
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
enum TsSymbolKind {
    Interface(TsInterface),
    UnionType(TsUnionType),
    UtilClass(TsUtilClass),
}

impl<'a> Display for WithMethods<'a, TsSymbolKind> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self(sym, methods) = self;
        match sym {
            TsSymbolKind::Interface(r) => Display::fmt(&WithMethods(r, methods), f),
            TsSymbolKind::UnionType(si) => Display::fmt(&WithMethods(si, methods), f),
            TsSymbolKind::UtilClass(uc) => Display::fmt(&WithMethods(uc, methods), f),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct TsSymbol {
    kind: TsSymbolKind,
    methods: BTreeMap<usize, TsFunction>,
}

impl Display for TsSymbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self {
            kind,
            methods,
            ..
        } = self;
        // write!(f, "package {pkg};\n\n{}", WithMethods(kind, methods))
        write!(f, "{}", WithMethods(kind, methods))
    }
}

enum TsPendingSymbol<'a> {
    Interface(&'a StructDef),
    UnionType(&'a EnumDef),
}

impl<'a> From<&'a StructDef> for TsPendingSymbol<'a> {
    fn from(value: &'a StructDef) -> Self {
        Self::Interface(value)
    }
}

impl<'a> From<&'a EnumDef> for TsPendingSymbol<'a> {
    fn from(value: &'a EnumDef) -> Self {
        Self::UnionType(value)
    }
}

struct TsModuleBuilder<'a> {
    fqdn: String,
    pending_symbols: Vec<(Rc<String>, TsPendingSymbol<'a>)>,
    pending_functions: Vec<(SymbolRef, &'a FunctionDef)>,
    by_ref: BTreeMap<SymbolRef, TsSymbolRef>,
    by_fqdn: BTreeMap<Rc<String>, TsSymbolRef>,
    fn_to_container: BTreeMap<SymbolRef, TsSymbolRef>,
}

impl<'a> ToTsResolver for TsModuleBuilder<'a> {
    fn convert_refs(&self, r: &SymbolRef) -> TsSymbolRef {
        self.by_ref
            .get(r)
            .cloned()
            .expect("SymbolRef should exist")
    }

    fn get_fn_container(&self, r: &SymbolRef) -> Option<TsSymbolRef> {
        self.fn_to_container
            .get(r)
            .cloned()
    }
}

impl<'a> TryFrom<&'a Module> for TsModule {
    type Error = Cow<'static, str>;

    fn try_from(src: &'a Module) -> Result<Self, Self::Error> {
        let mut builder = TsModuleBuilder {
            fqdn: "Root".into(),
            pending_symbols: Default::default(),
            pending_functions: Default::default(),
            by_ref: Default::default(),
            by_fqdn: Default::default(),
            fn_to_container: Default::default(),
        };
        for (id, s) in src.symbols_ref() {
            let ts_ref: TsSymbolRef = (
                builder.pending_symbols.len(),
                Rc::new(s.name().into()),
                None
            );
            builder.pending_symbols.push((ts_ref.1.clone(), match s {
                Symbol::Struct(v) => v.into(),
                Symbol::Enum(v) => {
                    for v in v.variants() {
                        let variant_ref: TsSymbolRef = (
                            ts_ref.0.clone(),
                            Rc::new(format!("{}_{}", ts_ref.1, v.name())),
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
            builder.by_ref.insert(id, ts_ref.clone());
            builder.by_fqdn.insert(ts_ref.1.clone(), ts_ref);
        }
        let mut module = Self {
            symbols: vec![],
            by_fqdn: Default::default(),
        };
        for (fqdn, sym) in &builder.pending_symbols {
            let id = module.symbols.len();
            let sym = match sym {
                TsPendingSymbol::Interface(s) => TsSymbol {
                    methods: Default::default(),
                    kind: TsSymbolKind::Interface(TsInterface {
                        doc: s.doc().cloned(),
                        visibility: TsVisibility::from_type_scope(s.scope()),
                        name: s.name().to_string(),
                        parent: None,
                        fields: match s.fields() {
                            StructFields::Named(s) => s
                                .iter()
                                .filter_map(|f| Some(TsField {
                                    doc: f.doc().cloned(),
                                    name: f.name().to_string(),
                                    typ: match builder.convert_types(f.r#type()) {
                                        TsType::Undefined => return None,
                                        t => t
                                    },
                                }))
                                .collect(),
                            StructFields::Tuple(s) => s
                                .iter()
                                .filter_map(|f| Some(TsField {
                                    doc: None,
                                    name: format!("_{}", f.offset()),
                                    typ: match builder.convert_types(f.r#type()) {
                                        TsType::Undefined => return None,
                                        t => t
                                    },
                                }))
                                .collect(),
                            StructFields::Unit => Vec::new()
                        },
                    }),
                },
                TsPendingSymbol::UnionType(e) => TsSymbol {
                    methods: Default::default(),
                    kind: TsSymbolKind::UnionType(TsUnionType {
                        doc: e.doc().cloned(),
                        visibility: TsVisibility::from_type_scope(e.scope()),
                        name: e.name().to_string(),
                        permitted: e.variants()
                            .iter()
                            .map(|v| TsInterface {
                                doc: v.doc().cloned(),
                                visibility: TsVisibility::Local,
                                name: v.name().to_string(),
                                parent: Some(builder.by_fqdn
                                    .get(&*fqdn)
                                    .cloned()
                                    .expect("a parent interface")),
                                fields: match v.fields() {
                                    EnumVariantFields::Named(f) => f
                                        .iter()
                                        .filter_map(|f| Some(TsField {
                                            doc: f.doc().cloned(),
                                            name: f.name().to_string(),
                                            typ: match builder.convert_types(f.r#type()) {
                                                TsType::Undefined => return None,
                                                t => t
                                            },
                                        }))
                                        .collect(),
                                    EnumVariantFields::Tuple(f) => f
                                        .iter()
                                        .filter_map(|f| Some(TsField {
                                            doc: None,
                                            name: format!("_{}", f.offset()),
                                            typ: match builder.convert_types(f.r#type()) {
                                                TsType::Undefined => return None,
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
        let utils_sym: TsSymbolRef = (
            builder.pending_symbols.len(),
            Rc::new(format!("Utils")),
            None
        );
        module.by_fqdn.insert(utils_sym.1.to_string(), utils_sym.0);
        module.symbols.push(TsSymbol {
            kind: TsSymbolKind::UtilClass(TsUtilClass {
                visibility: TsVisibility::Exported,
                name: "Utils".to_string(),
            }),
            methods: Default::default(),
        });

        let mut next_local_fn_id = 0usize;

        let mut pending_bodies = vec![];
        for (id, def) in &builder.pending_functions {
            let Some(body) = def.body() else { continue; };

            let ret_type = builder.convert_types(def.ret_type());

            let function = TsFunction {
                doc: def.doc().cloned(),
                visibility: TsVisibility::from_type_scope(def.scope()),
                name: match def.scope() {
                    crate::types::Scope::Public => def.name().to_string(),
                    crate::types::Scope::Private => {
                        next_local_fn_id += 1;
                        format!("_{}_{}", next_local_fn_id, def.name())
                    }
                },
                parameters: def.params()
                    .iter()
                    .map(|p| TsFunctionParameter {
                        name: p.name().to_string(),
                        fqdn: builder.convert_types(p.typ()),
                    }).collect(),
                ret_type: ret_type.clone(),
                body: None,
            };

            let sym_ref = match (&function.ret_type, function.parameters.as_slice()) {
                (TsType::Class(sym), _) => sym,
                (_, [TsFunctionParameter { fqdn: TsType::Class(sym), .. }]) => sym,
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

struct TsModule {
    symbols: Vec<TsSymbol>,
    by_fqdn: BTreeMap<String, usize>,
}

impl TsModule {
    pub fn resolve(&self, fqdn: &str) -> Option<&TsSymbol> {
        self.by_fqdn
            .get(fqdn)
            .and_then(|id| self.symbols.get(*id))
    }

    pub fn resolve_enum_variant(&self, fqdn: &str, variant: &str) -> Option<TsSymbolRef> {
        let id = self.by_fqdn.get(fqdn)?;
        match self.symbols.get(*id) {
            Some(TsSymbol { kind: TsSymbolKind::UnionType(si), .. }) if si.permitted.iter().find(|v| variant == v.name).is_some() => Some((
                *id,
                format!("{fqdn}_{variant}").into(),
                Some(variant.to_string())
            )),
            _ => None
        }
    }

    pub fn resolve_record_field(&self, sr: &TsSymbolRef, name: &str) -> Option<&TsField> {
        let sym = self.symbols.get(sr.0)?;
        match sym {
            TsSymbol { kind: TsSymbolKind::Interface(rec), .. } => rec.fields.iter().find(|f| *name == *f.name),
            TsSymbol { kind: TsSymbolKind::UnionType(si), .. } => if let Some(vn) = &sr.2 {
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

impl Display for TsModule {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        for sym in &self.symbols {
            Display::fmt(sym, f)?;
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
struct TsNumber(f64);

impl From<f64> for TsNumber {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl From<i64> for TsNumber {
    fn from(value: i64) -> Self {
        Self(value as f64)
    }
}

impl Display for TsNumber {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Eq for TsNumber {}

#[derive(Debug, Eq, PartialEq)]
enum TsOp {
    Error(Cow<'static, str>),
    Unreachable(Cow<'static, str>),
    Nop(TsValueRef),
    VarDef(TsValueKind),
    VarSet(TsValueRef, TsValueRef),
    VarGet(TsValueRef),
    Neg(TsValueRef),
    ConstNull,
    ConstString(String),
    ConstBool(bool),
    ConstNumber(TsNumber),
    Cast(TsValueRef, TsSymbolRef),
    GetParam(TsValueKind, Rc<String>),
    GetTupleField(TsValueRef, TsValueKind, usize),
    GetRecordField(TsValueRef, TsValueKind, String),
    InvokeStatic(TsSymbolRef, TsValueKind, Rc<String>, Vec<TsValueRef>),
    Create(TsSymbolRef, Vec<TsValueRef>),
    Add(TsValueRef, TsValueRef),
    Gt(TsValueRef, TsValueRef),
    Eq(TsValueRef, TsValueRef),
    And(TsValueRef, TsValueRef),
    Or(TsValueRef, TsValueRef),
    InstanceOf(TsValueRef, TsSymbolRef),
    If(TsValueRef, TsBody, TsBody),
}

#[derive(Debug, Eq, PartialEq)]
enum TsTerminatorOp {
    Return,
    ReturnValue(TsValueRef),
    Unreachable(Cow<'static, str>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum TsValueKind {
    Never,
    Void,
    TsType(TsType),
    OpaqueVar(Rc<TsValueKind>),
}

impl TsValueKind {
    pub fn as_value_type(&self) -> &TsValueKind {
        match self {
            TsValueKind::OpaqueVar(v) => v.deref(),
            o => o
        }
    }

    pub fn default_value(&self) -> &str {
        match self {
            TsValueKind::Never => "null",
            TsValueKind::Void => "null",
            TsValueKind::TsType(typ) => match typ {
                TsType::Undefined => "null",
                TsType::Boolean => "false",
                TsType::String => "\"\"",
                TsType::Number => "0",
                TsType::Class(_) => "null",
            }
            TsValueKind::OpaqueVar(_) => "null /* VAR !? */",
        }
    }
}

impl TsOp {
    fn should_store_in_var(&self) -> bool {
        !matches!(self,
            TsOp::Nop(..) |
            TsOp::If(..) |
            TsOp::VarDef(..) |
            TsOp::VarSet(..) |
            TsOp::VarGet(..) |
            TsOp::Unreachable(..) |
            TsOp::GetParam(..) |
            TsOp::ConstNumber(..) |
            TsOp::ConstBool(..) |
            TsOp::And(..) |
            TsOp::Or(..) |
            TsOp::Eq(..) |
            TsOp::InstanceOf(..)
        )
    }
}

impl Typed for TsOp {
    type ValueType = TsValueKind;

    fn typ(&self) -> Self::ValueType {
        fn bin_op_typ<T: Typed<ValueType=TsValueKind>>(l: &T, r: &T) -> TsValueKind {
            let l = l.typ().as_value_type().clone();
            let r = r.typ();
            if matches!(l, TsValueKind::Never) {
                r
            } else if matches!(r, TsValueKind::Never) {
                l
            } else if l == *r.as_value_type() {
                l
            } else {
                let (l, r) = match (l, r) {
                    (TsValueKind::TsType(l), TsValueKind::TsType(r)) => (l, r),
                    _ => return TsValueKind::Never
                };
                TsValueKind::TsType(match (l, r) {
                    // (TsType::Number, TsType::Number) => TsType::Number,

                    _ => return TsValueKind::Never
                })
            }
        }

        match self {
            TsOp::Add(l, r) => bin_op_typ(l, r),
            TsOp::Gt(_, _) => TsValueKind::TsType(TsType::Boolean),
            TsOp::Eq(_, _) => TsValueKind::TsType(TsType::Boolean),
            TsOp::And(_, _) => TsValueKind::TsType(TsType::Boolean),
            TsOp::Or(_, _) => TsValueKind::TsType(TsType::Boolean),
            TsOp::InstanceOf(_, _) => TsValueKind::TsType(TsType::Boolean),
            TsOp::ConstNull => TsValueKind::Void,
            TsOp::ConstString(_) => TsValueKind::TsType(TsType::String),
            TsOp::ConstBool(_) => TsValueKind::TsType(TsType::Boolean),
            TsOp::ConstNumber(_) => TsValueKind::TsType(TsType::Number),
            TsOp::Create(s, _) => TsValueKind::TsType(TsType::Class(s.clone())),
            TsOp::Cast(_, s) => TsValueKind::TsType(TsType::Class(s.clone())),
            TsOp::Error(_) => TsValueKind::Never,
            TsOp::Unreachable(_) => TsValueKind::Never,
            TsOp::GetParam(k, _) => k.clone(),
            TsOp::GetTupleField(_, k, _) => k.clone(),
            TsOp::GetRecordField(_, k, _) => k.clone(),
            TsOp::If(_, t, f) => bin_op_typ(t, f),
            TsOp::InvokeStatic(_, k, _, _) => k.clone(),
            TsOp::Neg(v) => v.typ(),
            TsOp::Nop(v) => v.typ(),
            TsOp::VarDef(k) => TsValueKind::OpaqueVar(Rc::from(k.clone())),
            TsOp::VarSet(_, _) => TsValueKind::Void,
            TsOp::VarGet(vr) => vr.typ().as_value_type().clone(),
        }
    }
}

impl Typed for TsTerminatorOp {
    type ValueType = TsValueKind;

    fn typ(&self) -> Self::ValueType {
        match self {
            TsTerminatorOp::Return => TsValueKind::Void,
            TsTerminatorOp::ReturnValue(v) => v.typ(),
            TsTerminatorOp::Unreachable(_) => TsValueKind::Never
        }
    }
}

impl Display for TsValueKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            TsValueKind::Never => write!(f, "var /* ??? */"),
            TsValueKind::Void => write!(f, "void"),
            TsValueKind::TsType(tst) => Display::fmt(tst, f),
            TsValueKind::OpaqueVar(v) => Display::fmt(v, f),
        }
    }
}

impl From<TsType> for TsValueKind {
    fn from(value: TsType) -> Self {
        match value {
            TsType::Undefined => Self::Void,
            value => Self::TsType(value)
        }
    }
}

type TsValueRef = RuntimeValue<TsValueKind>;

type TsBody = Body0<TsValueKind, TsOp, TsTerminatorOp>;
type TsBlockBuilder<'a> = BlockBuilder<'a, SkValueKind, TsOp, TsTerminatorOp>;

type TsScopes<'a> = Scopes<TsValueRef>;
type TsScope<'a> = Scope<'a, TsValueRef>;

type TsPrintScopes<'a> = Scopes<Rc<String>>;
type TsPrintScope<'a> = Scope<'a, Rc<String>>;

impl<'a> TryFrom<(&'a SkBody, &'a TsModuleBuilder<'_>, &'a TsModule)> for TsBody {
    type Error = Cow<'static, str>;

    fn try_from((pb, builder, module): (&'a SkBody, &'a TsModuleBuilder, &'a TsModule)) -> Result<Self, Self::Error> {
        let params = pb.params()
            .iter()
            .map(|x| builder.convert_kind(x)) // TODO
            .collect::<Vec<_>>();
        Ok(Self::isolated(params.as_slice(), |b, args| {
            let mut scopes = TsScopes::new();
            let scope = scopes.root(args.as_slice());
            Self::visit_body(pb, builder, module, b, scope)
        })?)
    }
}

impl TsBody {
    fn visit_body(
        pb: &SkBody,
        builder: &TsModuleBuilder,
        module: &TsModule,
        b: &mut BlockBuilder<TsValueKind, TsOp, TsTerminatorOp>,
        mut scope: TsScope,
    ) -> Result<TsTerminatorOp, Cow<'static, str>> {
        for (rv, op) in pb.entries() {
            let jop = Self::visit_op(builder, module, b, &mut scope, op)?;
            let r = b.op(jop);
            scope.bind(&rv, r)?;
        }
        Ok(match pb.terminator_op() {
            SkTerminatorOp::Yield(v) if matches!(v.typ(), SkValueKind::Unit) => TsTerminatorOp::Return,
            SkTerminatorOp::Yield(v) => TsTerminatorOp::ReturnValue(match scope.get(v) {
                None => return Err("no mapping found for SkTerminatorOp::Yield arg")?,
                Some(v) => v.clone()
            }),
        })
    }

    fn visit_nested_body(
        builder: &TsModuleBuilder,
        module: &TsModule,
        b: &mut BlockBuilder<TsValueKind, TsOp, TsTerminatorOp>,
        scope: &mut TsScope,
        tb: &SkBody,
        parameters: impl Into<Vec<TsValueKind>>,
    ) -> Result<TsBody, Cow<'static, str>> {
        b.body(parameters, |bb, params| {
            Self::visit_body(tb, builder, module, bb, scope.nested(params.as_slice()))
        })
    }

    fn visit_op(
        builder: &TsModuleBuilder,
        module: &TsModule,
        b: &mut BlockBuilder<TsValueKind, TsOp, TsTerminatorOp>,
        mut scope: &mut TsScope,
        op: &SkOp,
    ) -> Result<TsOp, Cow<'static, str>> {
        Ok(match op {
            SkOp::Label(v, _) => match scope.get(v).cloned() {
                None => TsOp::Error("no mapping found for SkOp::Label arg".into()),
                // None => panic!("no mapping found for SkOp::Label arg: {v:?} in {scope:?}"),
                Some(v) => TsOp::Nop(v.clone())
            },
            SkOp::ConstUnit => TsOp::Error("SkOp::ConstUnit conversion not implemented".into()),
            SkOp::ConstBool(v) => TsOp::ConstBool(*v),
            SkOp::ConstI64(n) => TsOp::ConstNumber(n.clone().into()),
            SkOp::Block(_) => TsOp::Error("SkOp::Block conversion not implemented".into()), // TODO
            SkOp::Neg(v) => match v.id() {
                RefId::Param(_) => TsOp::Error("SkOp::Neg arg cannot be an RefId::Param".into()),
                RefId::Op(_) => match scope.get(v).cloned() {
                    None => TsOp::Error("no mapping found for SkOp::Neg arg".into()),
                    Some(v) => TsOp::Neg(v),
                }
            }
            SkOp::Add(l, r) => match (l.id(), r.id()) {
                (RefId::Param(_), _) => TsOp::Error("SkOp::Add left arg cannot be an RefId::Param".into()),
                (_, RefId::Param(_)) => TsOp::Error("SkOp::Add right arg cannot be an RefId::Param".into()),
                (RefId::Op(_), RefId::Op(_)) => match (scope.get(l).cloned(), scope.get(r).cloned()) {
                    (None, _) => TsOp::Error("no mapping found for SkOp::Add left arg".into()),
                    (_, None) => TsOp::Error("no mapping found for SkOp::Add right arg".into()),
                    (Some(l), Some(r)) => TsOp::Add(l, r),
                }
            }
            SkOp::Mul(_, _) => TsOp::Error("SkOp::ConstUnit conversion not implemented".into()), // TODO
            SkOp::Gt(l, r) => match (l.id(), r.id()) {
                (RefId::Param(_), _) => TsOp::Error("SkOp::Gt left arg cannot be an RefId::Param".into()),
                (_, RefId::Param(_)) => TsOp::Error("SkOp::Gt right arg cannot be an RefId::Param".into()),
                (RefId::Op(_), RefId::Op(_)) => match (scope.get(l).cloned(), scope.get(r).cloned()) {
                    (None, _) => TsOp::Error("no mapping found for SkOp::Gt left arg".into()),
                    (_, None) => TsOp::Error("no mapping found for SkOp::Gt right arg".into()),
                    (Some(l), Some(r)) => TsOp::Gt(l, r),
                }
            }
            SkOp::If(e, tb, fb) => match e.id() {
                RefId::Param(_) => TsOp::Error("SkOp::If condition arg cannot be an RefId::Param".into()),
                RefId::Op(_) => match scope.get(e).cloned() {
                    None => TsOp::Error("no mapping found for SkOp::If condition arg".into()),
                    Some(e) => {
                        let tb: TsBody = Self::visit_nested_body(builder, module, b, scope, tb, [])?;
                        let fb: TsBody = Self::visit_nested_body(builder, module, b, scope, fb, [])?;
                        TsOp::If(e, tb, fb)
                    }
                }
            },
            SkOp::Call(s, _, a) => {
                let id = s.id();
                match builder.get_fn_container(s) {
                    None => TsOp::Error(format!("function #{s} not found in SkOp::Call").into()),
                    Some(s) => match module.resolve(s.1.as_str()) {
                        None => TsOp::Error("container not found in SkOp::Call".into()),
                        Some(m) => match m.methods.get(&id) {
                            None => TsOp::Error("container doesn't contain function in SkOp::Call".into()),
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
                                TsOp::InvokeStatic(
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
                    None => TsOp::Error("symbol not found in SkOp::Create".into()),
                    Some(m) => match m.kind {
                        TsSymbolKind::Interface(ref m) => m.fields
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
                            .map(|params| TsOp::Create(s.clone(), params))
                            .unwrap_or_else(|err| TsOp::Error(err)),
                        _ => TsOp::Error("only Records can be constructed in SkOp::Create".into()),
                    },
                }
            }
            SkOp::Match(e, p) => match e.id() {
                RefId::Param(_) => TsOp::Error("SkOp::Match expression arg cannot be an RefId::Param".into()),
                RefId::Op(_) => match scope.get(e).cloned() {
                    None => TsOp::Error("no mapping found for SkOp::Match expression arg".into()),
                    Some(e) => {
                        let mut iter = p.iter();
                        if let Some(first) = iter.next() {
                            Self::visit_match_case(builder, module, b, scope, first, iter, e)?
                        } else {
                            TsOp::Error("No case in SkOp::Match!".into())
                        }
                    }
                }
            },
        })
    }
}

#[test]
fn it_generates_typescript() -> Result<(), Cow<'static, str>> {
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

    let ts = TsModule::try_from(&module)?;

    assert_eq!(
        format!("{ts}"),
        /*language=ts*/(r#"/**
 * A point with a X and Y coordinate
 * <p>
 * NOTE: coordinates are i16 since Skrull doesn't support isize primitive type yet.
 */
class Point {
  public constructor(
    public readonly x: number,
    public readonly y: number
  ) { }
}

function _1_new_point(x: number, y: number): Point {
  const _0_3: number = -1;
  const _0_4: number = y + _0_3;
  const _0_5: Point = new Point(x, _0_4);
  return _0_5;
}

/**
 * A simple shape
 */
class Rectangle {
  /**
   * @param size Dimensions, in pixels
   */
  public constructor(
    public readonly origin: Point,
    public readonly size: Size
  ) { }
}

/**
 * Doc is also supported *_*
 */
function _2_new_rect(x: number, y: number, w: number, h: number): Rectangle {
  const _0_4: Point = _1_new_point(x, y);
  const _0_5: number = _3_times2(w);
  const _0_6: Size = new Size(_0_5, h);
  const _0_7: Rectangle = new Rectangle(_0_4, _0_6);
  return _0_7;
}

/**
 * The simplest shape available
 */
class Shape_Rect {
  public constructor(
    public readonly _0: Rectangle
  ) { }
}

/**
 * Shapes we are able to create
 */
type Shape =
  | Shape_Rect;

export class Size {
  /**
   * @param width W
   * @param height H
   */
  public constructor(
    public readonly width: number,
    public readonly height: number
  ) { }
}

function _3_times2(v: number): number {
  const _0_2: boolean = v > 0;
  let _if_0_3: number;
  if (_0_2) {
    const _1_1: number = v + 1000000;
    _if_0_3 = _1_1;
  } else {
    const _1_2: number = 4000000 + 600000;
    _if_0_3 = _1_2;
  }
  const _0_4: number = v + _if_0_3;
  return _0_4;
}

"#.to_string())
    );

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
    let ts = TsModule::try_from(&module)?;

    assert_eq!(
        format!("{ts}"),
        /*language=ts*/(r#"class Price_Limit {
  public constructor() { }
}

class Price_Market {
  public constructor() { }
}

class Price_StopLimit {
  public constructor(
    public readonly stop_price: number
  ) { }
}

export type Price =
  | Price_Limit
  | Price_Market
  | Price_StopLimit;

class Some {
  public constructor(
    public readonly _0: Price
  ) { }
}

export function is_priced(maybe_price: Some): boolean {
  let _if_0_2: boolean;
  if (maybe_price instanceof Some) {
    const _1_0: Some = (maybe_price) as (Some);
    const _1_1: Price = _1_0._0;
    _if_0_2 = _1_1 instanceof Price_Limit || _1_1 instanceof Price_StopLimit;
  } else {
    _if_0_2 = false;
  }
  let _if_0_3: boolean;
  if (_if_0_2) {
    _if_0_3 = true;
  } else {
    _if_0_3 = false;
  }
  return _if_0_3;
}

"#.to_string())
    );

    Ok(())
}
