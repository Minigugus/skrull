use alloc::{format, vec};
use alloc::borrow::{Cow, ToOwned};
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use core::cmp::Ordering;
use core::fmt::{Debug, Display, Formatter};
use core::mem::take;

use crate::bytecode::{Body, MatchCaseOp, MatchPatternOp, SymbolRefOrEnum, TerminatorOp, ValueKind, ValueRef};
use crate::lexer::Token;
use crate::parser::{BlockExpression, Declaration, DocBlock, Enum, Expression, Fields, FunctionDeclaration, FunctionPrototype, Identifier, MatchExpression, MatchPattern, parse_declaration, Qualifier, Struct, Type, Visibility};

type Result<T> = core::result::Result<T, Cow<'static, str>>;

trait SymbolLoader {
    type Symbol;

    fn load(self, ctx: &impl LoaderContext) -> Result<Self::Symbol>;
}

trait LoaderContext {
    fn get_symbol_from_ref(&self, r: &SymbolRef) -> Result<&Symbol>;
    fn get_symbol_ref_from_name(&self, name: &(impl AsRef<str> + ?Sized)) -> Result<SymbolRef>;
    fn get_symbol_ref_from_qualifier(&self, name: &Qualifier) -> Result<SymbolRef>;

    fn get_symbol_from_name(&self, name: &(impl AsRef<str> + ?Sized)) -> Result<&Symbol> {
        self.get_symbol_from_ref(&self.get_symbol_ref_from_name(name)?)
    }
}

pub trait ResolverContext {
    fn get_symbol(&self, r: &SymbolRef) -> Result<&Symbol>;
}

impl ResolverContext for () {
    fn get_symbol(&self, _: &SymbolRef) -> Result<&Symbol> {
        Err("symbols not supported by this resolver".into())
    }
}

impl ValueKind {
    fn to_string(&self, ctx: &impl LoaderContext) -> String {
        match self {
            ValueKind::Never => "!".into(),
            ValueKind::Unit => "()".into(),
            ValueKind::Bool => "bool".into(),
            ValueKind::I64 => "i64".into(),
            ValueKind::I16 => "i16".into(),
            ValueKind::F64 => "f64".into(),
            ValueKind::Usize => "usize".into(),
            ValueKind::Type(sr) => {
                match ctx.get_symbol_from_ref(sr) {
                    Ok(found) => found.name().into(),
                    Err(e) => format!("<unresolved: {e}>")
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Scope {
    Private,
    Public,
}

impl From<Visibility> for Scope {
    fn from(value: Visibility) -> Self {
        match value {
            Visibility::Pub => Self::Public,
            Visibility::Default => Self::Private
        }
    }
}

#[derive(Debug, Clone)]
#[cfg(not(test))]
pub struct SymbolRef(u32, SymbolType);

#[derive(Debug, Clone)]
#[cfg(test)]
pub struct SymbolRef(pub u32, pub SymbolType);

impl Eq for SymbolRef {}

impl PartialOrd for SymbolRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.0, &other.0)
    }
}

impl Ord for SymbolRef {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&self.0, &other.0)
    }
}

impl PartialEq for SymbolRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl From<(u32, SymbolType)> for SymbolRef {
    fn from((id, typ): (u32, SymbolType)) -> Self {
        Self(id, typ)
    }
}

impl Display for SymbolRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl SymbolRef {
    pub fn id(&self) -> usize {
        self.0 as usize
    }

    pub fn typ(&self) -> &SymbolType {
        &self.1
    }
}

#[derive(Debug)]
pub enum Symbol {
    Struct(StructDef),
    Enum(EnumDef),
    Function(FunctionDef),
}

impl Symbol {
    pub fn name(&self) -> &str {
        match self {
            Symbol::Struct(sym) => sym.name.as_ref(),
            Symbol::Enum(sym) => sym.name.as_ref(),
            Symbol::Function(sym) => sym.name.as_ref(),
        }
    }
}

pub struct PendingSymbol<'a> {
    symbol: PendingProcessing<'a>,
    body: Option<BlockExpression<'a>>,
}

pub enum PendingProcessing<'a> {
    Struct(Struct<'a>),
    Enum(Enum<'a>),
    Function(FunctionPrototype<'a>),
}

impl<'a> PendingProcessing<'a> {
    pub fn meta(&self) -> Option<(String, SymbolType)> {
        Some(match self {
            PendingProcessing::Struct(sym) => (sym.name.as_ref().to_string(), SymbolType::Struct),
            PendingProcessing::Enum(sym) => (sym.name.as_ref().to_string(), SymbolType::Enum),
            PendingProcessing::Function(sym) => (sym.name.as_ref().to_string(), SymbolType::Function),
        })
    }
}

impl<'a> From<Struct<'a>> for PendingProcessing<'a> {
    fn from(value: Struct<'a>) -> Self {
        Self::Struct(value)
    }
}

impl<'a> From<Enum<'a>> for PendingProcessing<'a> {
    fn from(value: Enum<'a>) -> Self {
        Self::Enum(value)
    }
}

impl<'a> From<FunctionPrototype<'a>> for PendingProcessing<'a> {
    fn from(value: FunctionPrototype<'a>) -> Self {
        Self::Function(value)
    }
}

impl<'a> From<Declaration<'a>> for PendingSymbol<'a> {
    fn from(value: Declaration<'a>) -> Self {
        let (symbol, body) = match value {
            Declaration::Enum(v) => (v.into(), None),
            Declaration::Function(v) => (v.prototype.into(), Some(v.body)),
            Declaration::Struct(v) => (v.into(), None),
        };

        PendingSymbol { symbol, body }
    }
}

impl<'a> SymbolLoader for PendingProcessing<'a> {
    type Symbol = Symbol;

    fn load(self, ctx: &impl LoaderContext) -> Result<Self::Symbol> {
        Ok(match self {
            PendingProcessing::Struct(sym) => Symbol::Struct(sym.load(ctx)?),
            PendingProcessing::Enum(sym) => Symbol::Enum(sym.load(ctx)?),
            PendingProcessing::Function(sym) => Symbol::Function(sym.load(ctx)?),
        })
    }
}

fn format_doc(doc: DocBlock) -> Option<Rc<str>> {
    let doc = doc.into_vec();
    if doc.is_empty() {
        None
    } else {
        Some(doc
            .iter()
            .map(|l| l.content)
            .collect::<Vec<_>>()
            .join("\n")
            .into())
    }
}

impl<'a> SymbolLoader for Struct<'a> {
    type Symbol = StructDef;

    fn load(self, ctx: &impl LoaderContext) -> Result<Self::Symbol> {
        Ok(StructDef {
            doc: format_doc(self.doc),
            scope: Scope::from(self.visibility),
            name: self.name.as_ref().to_string(),
            fields: load_struct_fields(ctx, self.body)
                .map_err(|e| format!("struct '{}': {e}", self.name.as_ref()))?,
        })
    }
}

fn load_type(ctx: &impl LoaderContext, typ: Type) -> Result<TypeRef> {
    Ok(match typ {
        Type::I16 => TypeRef::Primitive(PrimitiveType::I16),
        Type::U32 => TypeRef::Primitive(PrimitiveType::U32),
        Type::I64 => TypeRef::Primitive(PrimitiveType::I64),
        Type::F64 => TypeRef::Primitive(PrimitiveType::F64),
        Type::Unit => TypeRef::Primitive(PrimitiveType::Unit),
        Type::Usize => TypeRef::Primitive(PrimitiveType::Usize),
        Type::Identifier(id) => TypeRef::Ref(ctx.get_symbol_ref_from_name(id.as_ref())?),
    })
}

fn load_named_field(ctx: &impl LoaderContext, nf: crate::parser::NamedField) -> Result<NamedField> {
    Ok(NamedField {
        doc: format_doc(nf.doc),
        scope: Scope::from(nf.visibility),
        name: nf.name.as_ref().to_string(),
        typ: load_type(ctx, nf.typ)
            .map_err(|e| format!("field '{}': {e}", nf.name.as_ref()))?,
    })
}

fn load_named_fields(ctx: &impl LoaderContext, fields: Vec<crate::parser::NamedField>) -> Result<Vec<NamedField>> {
    fields
        .into_iter()
        .map(|x| load_named_field(ctx, x))
        .collect::<Result<Vec<_>>>()
}

fn load_tuple_field(ctx: &impl LoaderContext, x: crate::parser::TupleField, offset: usize) -> Result<TupleField> {
    Ok(TupleField {
        scope: Scope::from(x.visibility),
        offset,
        typ: load_type(ctx, x.typ)?,
    })
}

fn load_tuple_fields(ctx: &impl LoaderContext, fields: Vec<crate::parser::TupleField>) -> Result<Vec<TupleField>> {
    fields
        .into_iter()
        .enumerate()
        .map(|(offset, field)| load_tuple_field(ctx, field, offset))
        .collect::<Result<Vec<_>>>()
}

fn load_struct_fields(ctx: &impl LoaderContext, fields: Fields) -> Result<StructFields> {
    Ok(match fields {
        Fields::NamedFields(fields) => StructFields::Named(load_named_fields(ctx, fields)?),
        Fields::TupleFields(fields) => StructFields::Tuple(load_tuple_fields(ctx, fields)?),
        Fields::Unit => StructFields::Unit
    })
}

fn load_enum_variant(ctx: &impl LoaderContext, ev: crate::parser::EnumVariant) -> Result<EnumVariant> {
    Ok(EnumVariant {
        doc: format_doc(ev.doc),
        name: ev.name.as_ref().to_string(),
        fields: match ev.fields {
            Fields::NamedFields(fields) => EnumVariantFields::Named(load_named_fields(ctx, fields)?),
            Fields::TupleFields(fields) => EnumVariantFields::Tuple(load_tuple_fields(ctx, fields)?),
            Fields::Unit => EnumVariantFields::Unit
        },
    })
}

impl<'a> SymbolLoader for Enum<'a> {
    type Symbol = EnumDef;

    fn load(self, ctx: &impl LoaderContext) -> Result<Self::Symbol> {
        Ok(EnumDef {
            doc: format_doc(self.doc),
            scope: Scope::from(self.visibility),
            name: self.name.as_ref().to_string(),
            variants: self.body
                .into_iter()
                .map(|v| load_enum_variant(ctx, v))
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl<'a> BlockExpression<'a> {
    pub fn to_fn_body(&self, parameters: &[Parameter], ctx: &impl LoaderContext) -> Result<Body> {
        Ok(Body::new_with_dyn_params(
            parameters
                .iter()
                .map(Parameter::typ)
                .map(type_ref_to_value_kind)
                .collect::<Vec<_>>()
                .as_ref(),
            |b, mut v| {
                let params: BTreeMap<String, ValueRef> = parameters
                    .iter()
                    .enumerate()
                    .map(|(i, p)| Ok((
                        p.name().to_string(),
                        b.label(take(&mut v[i]), p.name().to_string())?
                    )))
                    .collect::<Result<BTreeMap<_, _>>>()?;

                trait Scope {
                    fn read_variable(&self, name: &Identifier) -> Result<ValueRef>;
                }

                impl Scope for BTreeMap<String, ValueRef> {
                    fn read_variable(&self, name: &Identifier) -> Result<ValueRef> {
                        self.get(name.as_ref()).cloned().ok_or_else(|| format!("variable not defined in scope: {name:?}").into())
                    }
                }

                struct MatchCaseScope<'a, T: ?Sized>(&'a T, &'a BTreeMap<&'a str, ValueRef>);

                impl<'a, T: Scope + ?Sized> Scope for MatchCaseScope<'a, T> {
                    fn read_variable(&self, name: &Identifier) -> Result<ValueRef> {
                        if let Some(v) = self.1.get(name.0) {
                            Ok(v.clone())
                        } else {
                            self.0.read_variable(name)
                        }
                    }
                }

                struct BlockScope<'a, T: ?Sized>(&'a T, RefCell<BTreeMap<ValueRef, ValueRef>>);

                impl<'a, T: Scope + ?Sized> Scope for BlockScope<'a, T> {
                    fn read_variable(&self, name: &Identifier) -> Result<ValueRef> {
                        let original_ref = self.0.read_variable(name)?;
                        let mut mapping = self.1.borrow_mut();
                        let mapped_ref = original_ref.map(mapping.len());
                        Ok(mapping.entry(original_ref).or_insert(mapped_ref).clone())
                    }
                }

                fn visit_expr(b: &mut Body, ctx: &impl LoaderContext, v: &dyn Scope, expr: &Expression) -> Result<ValueRef> {
                    Ok(match expr {
                        Expression::Unit => b.const_unit()?,
                        Expression::Literal(v) => b.const_i64(*v)?,
                        Expression::Identifier(n) => v.read_variable(n)?,
                        Expression::Block(e) => {
                            let e = visit_block(b, ctx, v, e)?;
                            b.block(e)?
                        }
                        Expression::Neg(e) => {
                            let e = visit_expr(b, ctx, v, e)?;
                            b.neg(e)?
                        }
                        Expression::Mul(l, r) => {
                            let l = visit_expr(b, ctx, v, l)?;
                            let r = visit_expr(b, ctx, v, r)?;
                            b.mul(l, r)?
                        }
                        Expression::Add(l, r) => {
                            let l = visit_expr(b, ctx, v, l)?;
                            let r = visit_expr(b, ctx, v, r)?;
                            b.add(l, r)?
                        }
                        Expression::Gt(l, r) => {
                            let l = visit_expr(b, ctx, v, l)?;
                            let r = visit_expr(b, ctx, v, r)?;
                            b.gt(l, r)?
                        }
                        Expression::If(c, t, f) => {
                            let c = visit_expr(b, ctx, v, c)?;
                            let t = visit_block(b, ctx, v, t)?;
                            let f = visit_block(b, ctx, v, f)?;

                            b.if_expr(c, t, f)?
                        }
                        Expression::Call(f, a) => {
                            let f = ctx.get_symbol_ref_from_name(f.as_ref()).or(Err(format!("Function not defined: {:}", f.as_ref())))?;
                            if !matches!(f.1, SymbolType::Function) {
                                return Err("Symbol isn't callable")?;
                            }
                            let args = a
                                .iter()
                                .map(|e| visit_expr(b, ctx, v, e))
                                .collect::<Result<Vec<_>>>()?;
                            b.call(f, ValueKind::Never, args)? // FIXME
                        }
                        Expression::Create(s, fs) => {
                            let sn = s.as_ref();
                            let s = ctx.get_symbol_ref_from_name(sn).or(Err(format!("Struct '{sn}' not defined")))?;
                            if !matches!(s.1, SymbolType::Struct) {
                                return Err(format!("'{sn}' isn't a named struct"))?;
                            }
                            let mut duplicates = BTreeSet::new();
                            let mut initializers = BTreeMap::new();
                            for (field, value) in fs {
                                let field = field.as_ref();
                                let value = visit_expr(b, ctx, v, value)?;
                                if initializers.insert(field, value).is_some() {
                                    duplicates.insert(field);
                                }
                            }
                            if !duplicates.is_empty() {
                                let duplicates = duplicates
                                    .into_iter()
                                    .collect::<Vec<_>>()
                                    .join("', '");
                                return Err(format!("Multiple initializers for fields '{duplicates}' while creating struct '{sn}'"))?;
                            }
                            let fs = initializers
                                .into_iter()
                                .map(|(n, v)| (n.to_owned(), v))
                                .collect::<Vec<_>>();
                            b.create(s, fs)?
                        }
                        Expression::Match(m) => {
                            fn visit_qualifier(
                                ctx: &impl LoaderContext,
                                qualifier: &Qualifier,
                            ) -> Result<SymbolRefOrEnum> {
                                if let Some(ref p) = qualifier.parent {
                                    Ok(SymbolRefOrEnum::Enum(
                                        ctx.get_symbol_ref_from_qualifier(&*p)?,
                                        qualifier.segment.as_ref().to_string(),
                                    ))
                                } else {
                                    Ok(SymbolRefOrEnum::Type(
                                        ctx.get_symbol_ref_from_qualifier(qualifier)?
                                    ))
                                }
                            }

                            fn visit_pattern<'a>(
                                e: &ValueKind,
                                pattern: &'a MatchPattern,
                                ctx: &'a impl LoaderContext,
                                vars_by_name: &mut BTreeMap<&'a str, (usize, ValueKind)>,
                            ) -> Result<MatchPatternOp> {
                                Ok(match pattern {
                                    MatchPattern::Unit => MatchPatternOp::Unit,
                                    MatchPattern::Wildcard => MatchPatternOp::Wildcard,
                                    MatchPattern::Variable(name) => {
                                        let id = vars_by_name.len();
                                        MatchPatternOp::Variable(
                                            vars_by_name
                                                .entry(name.as_ref())
                                                .or_insert_with(|| (id, e.clone()))
                                                .0
                                        )
                                    }
                                    MatchPattern::Union(p) => MatchPatternOp::Union(p
                                        .iter()
                                        .map(|p| visit_pattern(e, p, ctx, vars_by_name))
                                        .collect::<Result<Vec<_>>>()?),
                                    MatchPattern::NumberLiteral(n) => MatchPatternOp::NumberLiteral(n.clone()),
                                    MatchPattern::StringLiteral(s) => MatchPatternOp::StringLiteral(s.to_string()),
                                    MatchPattern::IsEnumOrType(q) => MatchPatternOp::IsTypeOrEnum(visit_qualifier(ctx, q)?),
                                    MatchPattern::TupleStruct { typ, params, exact } => {
                                        let typ = visit_qualifier(ctx, typ)?;
                                        let s = ctx.get_symbol_from_ref(typ.owner_type())?;
                                        let (name, fields) = match s {
                                            Symbol::Struct(StructDef { fields: StructFields::Tuple(fields), name, .. }) => (name, fields),
                                            s => return Err(format!("{s:?} isn't a tuple struct").into())
                                        };

                                        let m = params.len();

                                        MatchPatternOp::TupleStruct {
                                            typ,
                                            params: params
                                                .iter()
                                                .enumerate()
                                                .map(|(i, p)| {
                                                    if let Some(f) = fields.get(i) {
                                                        let typ = type_ref_to_value_kind(f.r#type());
                                                        visit_pattern(&typ, p, ctx, vars_by_name)
                                                    } else {
                                                        let n = fields.len();
                                                        Err(format!("Tuple struct {name} has only {n} fields but there are {m} in the pattern").into())
                                                    }
                                                })
                                                .collect::<Result<Vec<_>>>()?,
                                            exact: exact.clone(),
                                        }
                                    }
                                    MatchPattern::FieldStruct { typ, params, exact } => {
                                        let typ = visit_qualifier(ctx, typ)?;
                                        let s = ctx.get_symbol_from_ref(typ.owner_type())?;
                                        let (name, fields) = match s {
                                            Symbol::Struct(StructDef { fields: StructFields::Named(fields), name, .. }) => (name, fields),
                                            s => return Err(format!("{s:?} isn't a field struct").into())
                                        };

                                        let fields = fields
                                            .iter()
                                            .map(|x| (x.name.as_str(), x))
                                            .collect::<BTreeMap<_, _>>();

                                        MatchPatternOp::FieldStruct {
                                            typ,
                                            params: params
                                                .iter()
                                                .map(|(i, p)| {
                                                    if let Some(f) = fields.get(i.as_ref()) {
                                                        let typ = type_ref_to_value_kind(f.r#type());
                                                        Ok((
                                                            i.as_ref().to_string(),
                                                            visit_pattern(&typ, p, ctx, vars_by_name)?
                                                        ))
                                                    } else {
                                                        Err(format!("Field struct {name} doesn't have field {i:?}").into())
                                                    }
                                                })
                                                .collect::<Result<Vec<_>>>()?,
                                            exact: exact.clone(),
                                        }
                                    }
                                })
                            }

                            let MatchExpression { expression, cases } = &**m;
                            let expr = visit_expr(b, ctx, v, expression)?;
                            let cases = cases.iter().map(|x| {
                                let mut params = BTreeMap::new();
                                let pattern = visit_pattern(expr.kind(), &x.pattern, ctx, &mut params)?;
                                let guard = if let Some(ref e) = x.guard {
                                    Some(visit_block_with_variables(
                                        b,
                                        ctx,
                                        v,
                                        &BlockExpression { expressions: vec![], remainder: Some(e.clone()) },
                                        &params,
                                    )?)
                                } else {
                                    None
                                };
                                let body = if let Expression::Block(ref b) = *x.body {
                                    b.clone()
                                } else {
                                    Rc::new(BlockExpression {
                                        expressions: vec![],
                                        remainder: Some(x.body.clone()),
                                    })
                                };
                                let body = visit_block_with_variables(
                                    b,
                                    ctx,
                                    v,
                                    body.as_ref(),
                                    &params,
                                )?;
                                // let body = Body::new_with_dyn_params(
                                //     params
                                //         .values()
                                //         .cloned()
                                //         .collect::<Vec<_>>()
                                //         .as_slice(),
                                //     |b, v| {
                                //         let params: BTreeMap<String, ValueRef> = params
                                //             .iter()
                                //             .enumerate()
                                //             .map(|(i, (n, _))| Ok((
                                //                 n.to_string(),
                                //                 b.label(v[i].clone(), n.to_string())?
                                //             )))
                                //             .collect::<Result<BTreeMap<_, _>>>()?;
                                //
                                //         expression_to_body(ctx, &*body, &params, b, v)
                                //     },
                                // )?;
                                Ok(MatchCaseOp {
                                    pattern,
                                    body,
                                    guard,
                                })
                            }).collect::<Result<Vec<_>>>()?;
                            b.match_(expr, cases)?
                        }
                    })
                }

                fn visit_block_inner(b: &mut Body, ctx: &impl LoaderContext, v: &dyn Scope, e: &BlockExpression) -> Result<TerminatorOp> {
                    for e in &e.expressions {
                        visit_expr(b, ctx, v, e)?;
                    }
                    let remainder = if let Some(e) = &e.remainder {
                        visit_expr(b, ctx, v, e)?
                    } else {
                        b.const_unit()?
                    };

                    Ok(b.yield_expr(remainder)?)
                }

                fn visit_block(b: &mut Body, ctx: &impl LoaderContext, v: &dyn Scope, e: &BlockExpression) -> Result<(Box<[ValueRef]>, Box<Body>)> {
                    b.derive(|b| {
                        let ns = BlockScope(v, RefCell::new(Default::default()));

                        let terminator_op = visit_block_inner(b, ctx, &ns, e)?;

                        Ok((
                            ns.1.take()
                                .keys()
                                .cloned()
                                .collect(),
                            terminator_op
                        ))
                    })
                }

                fn visit_block_with_variables(
                    b: &mut Body,
                    ctx: &impl LoaderContext,
                    v: &dyn Scope,
                    e: &BlockExpression,
                    vars: &BTreeMap<&str, (usize, ValueKind)>,
                ) -> Result<(Box<[ValueRef]>, Box<Body>)> {
                    let params = vars
                        .values()
                        .map(|(_, v)| v)
                        .cloned()
                        .collect::<Vec<_>>();
                    b.derive_with_params(params.as_slice(), |b, params| {
                        let vars = vars
                            .iter()
                            .map(|(name, (i, _))| b
                                .label(params[*i].clone(), *name)
                                .map(|v| (*name, v)))
                            .collect::<Result<BTreeMap<_, _>>>()?;
                        let bs = BlockScope(v, RefCell::new(Default::default()));
                        let ns = MatchCaseScope(&bs, &vars);

                        let terminator_op = visit_block_inner(b, ctx, &ns, e)?;

                        Ok((
                            bs.1.take()
                                .keys()
                                .cloned()
                                .collect(),
                            terminator_op
                        ))
                    })
                }

                visit_block_inner(b, ctx, &params, self)
            },
        )?)
    }
}

impl<'a> SymbolLoader for FunctionPrototype<'a> {
    type Symbol = FunctionDef;

    fn load(self, ctx: &impl LoaderContext) -> Result<Self::Symbol> {
        Ok(FunctionDef {
            doc: format_doc(self.doc),
            scope: Scope::from(self.visibility),
            name: self.name.as_ref().to_string(),
            params: self.parameters
                .into_iter()
                .map(|p| Ok(Parameter {
                    mutable: p.mutability.into(),
                    name: p.name.as_ref().to_string(),
                    typ: load_type(ctx, p.typ.expect("parameters still requires type parameters"))?,
                }))
                .collect::<Result<Vec<_>>>()?,
            ret_type: self.ret_type
                .map(|t| load_type(ctx, t))
                .unwrap_or(Ok(TypeRef::Primitive(PrimitiveType::Unit)))?,
            body: None,
        })
    }
}

fn type_ref_to_value_kind(typ: &TypeRef) -> ValueKind {
    match typ {
        TypeRef::Primitive(PrimitiveType::Unit) => ValueKind::Unit,
        TypeRef::Primitive(PrimitiveType::I64) => ValueKind::I64,
        TypeRef::Primitive(PrimitiveType::I16) => ValueKind::I16,
        TypeRef::Primitive(PrimitiveType::F64) => ValueKind::F64,
        TypeRef::Primitive(PrimitiveType::Usize) => ValueKind::Usize,
        TypeRef::Ref(sym) => ValueKind::Type(sym.clone()),
        _ => ValueKind::Never
    }
}

pub fn value_kind_to_type_ref(typ: &ValueKind) -> Option<TypeRef> {
    Some(match typ {
        ValueKind::Unit => TypeRef::Primitive(PrimitiveType::Unit),
        ValueKind::I64 => TypeRef::Primitive(PrimitiveType::I64),
        ValueKind::I16 => TypeRef::Primitive(PrimitiveType::I16),
        ValueKind::F64 => TypeRef::Primitive(PrimitiveType::F64),
        ValueKind::Usize => TypeRef::Primitive(PrimitiveType::Usize),
        ValueKind::Type(sym) => TypeRef::Ref(sym.clone()),
        _ => return None
    })
}

#[derive(Debug)]
pub struct StructDef {
    doc: Option<Rc<str>>,
    scope: Scope,
    name: String,
    fields: StructFields,
}

impl StructDef {
    pub fn doc(&self) -> Option<&Rc<str>> {
        self.doc.as_ref()
    }

    pub fn scope(&self) -> Scope {
        self.scope
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn fields(&self) -> &StructFields {
        &self.fields
    }
}

#[derive(Debug)]
pub struct EnumDef {
    doc: Option<Rc<str>>,
    scope: Scope,
    name: String,
    variants: Vec<EnumVariant>,
}

impl EnumDef {
    pub fn doc(&self) -> Option<&Rc<str>> {
        self.doc.as_ref()
    }

    pub fn scope(&self) -> Scope {
        self.scope
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn variants(&self) -> &[EnumVariant] {
        &self.variants
    }
}

#[derive(Debug)]
pub struct Parameter {
    mutable: bool,
    name: String,
    typ: TypeRef,
}

impl Parameter {
    pub fn is_mutable(&self) -> bool {
        self.mutable
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn typ(&self) -> &TypeRef {
        &self.typ
    }
}

#[derive(Debug)]
pub struct FunctionDef {
    doc: Option<Rc<str>>,
    scope: Scope,
    name: String,
    params: Vec<Parameter>,
    ret_type: TypeRef,
    body: Option<Body>,
}

impl FunctionDef {
    pub fn doc(&self) -> Option<&Rc<str>> {
        self.doc.as_ref()
    }

    pub fn scope(&self) -> Scope {
        self.scope
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn params(&self) -> &[Parameter] {
        self.params.as_slice()
    }

    pub fn ret_type(&self) -> &TypeRef {
        &self.ret_type
    }

    pub fn body(&self) -> Option<&Body> {
        self.body.as_ref()
    }
}

#[derive(Debug)]
pub struct EnumVariant {
    doc: Option<Rc<str>>,
    name: String,
    fields: EnumVariantFields,
}

impl EnumVariant {
    pub fn doc(&self) -> Option<&Rc<str>> {
        self.doc.as_ref()
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn fields(&self) -> &EnumVariantFields {
        &self.fields
    }
}

#[derive(Debug)]
pub enum EnumVariantFields {
    Named(Vec<NamedField>),
    Tuple(Vec<TupleField>),
    Unit,
}

#[derive(Debug)]
pub struct TupleField {
    scope: Scope,
    offset: usize,
    typ: TypeRef,
}

impl TupleField {
    pub fn scope(&self) -> Scope {
        self.scope
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn r#type(&self) -> &TypeRef {
        &self.typ
    }
}

#[derive(Debug)]
pub enum StructFields {
    Named(Vec<NamedField>),
    Tuple(Vec<TupleField>),
    Unit,
}

#[derive(Debug)]
pub struct NamedField {
    doc: Option<Rc<str>>,
    scope: Scope,
    name: String,
    typ: TypeRef,
}

impl NamedField {
    pub fn doc(&self) -> Option<&Rc<str>> {
        self.doc.as_ref()
    }

    pub fn scope(&self) -> Scope {
        self.scope
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn r#type(&self) -> &TypeRef {
        &self.typ
    }
}

#[derive(Debug)]
pub enum TypeRef {
    Primitive(PrimitiveType),
    Ref(SymbolRef),
}

pub enum PrimitiveType {
    I16,
    U32,
    I64,
    F64,
    Unit,
    Usize,
}

impl Debug for PrimitiveType {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            PrimitiveType::I16 => write!(f, "i16"),
            PrimitiveType::U32 => write!(f, "u32"),
            PrimitiveType::I64 => write!(f, "i64"),
            PrimitiveType::F64 => write!(f, "f64"),
            PrimitiveType::Unit => write!(f, "()"),
            PrimitiveType::Usize => write!(f, "usize"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SymbolType {
    Struct,
    Enum,
    Function,
}

pub struct Module {
    name: String,
    symbols: Vec<Symbol>,
    by_name: BTreeMap<String, (u32, SymbolType)>,
}

impl Module {
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn symbols(&self) -> &[Symbol] {
        &self.symbols
    }

    pub fn symbols_ref(&self) -> Vec<(SymbolRef, &Symbol)> {
        self.by_name
            .values()
            .cloned()
            .map(|(id, typ)| (SymbolRef(id, typ), &self.symbols[id as usize]))
            .collect()
    }

    pub fn get_by_ref(&self, SymbolRef(id, _): SymbolRef) -> Result<&Symbol> {
        self.symbols
            .get(id as usize)
            .ok_or_else(|| format!("Unallocated Symbol reference: {id:?}").into())
    }

    pub fn get_by_name(&self, name: impl AsRef<str>) -> Option<&Symbol> {
        self.by_name
            .get(name.as_ref())
            .and_then(|(id, _)| self.symbols.get(*id as usize))
    }

    pub fn get_ref_by_name(&self, name: &(impl AsRef<str> + ?Sized)) -> Option<SymbolRef> {
        self.by_name
            .get(name.as_ref())
            .cloned()
            .map(|(id, typ)| SymbolRef(id, typ))
    }
}

impl LoaderContext for Module {
    fn get_symbol_from_ref(&self, r: &SymbolRef) -> Result<&Symbol> {
        self.get_by_ref(r.clone())
    }

    fn get_symbol_ref_from_name(&self, name: &(impl AsRef<str> + ?Sized)) -> Result<SymbolRef> {
        let name = name.as_ref();
        self.by_name
            .get(name)
            .cloned()
            .map(SymbolRef::from)
            .ok_or_else(|| format!("Symbol not defined: {name:}").into())
    }

    fn get_symbol_ref_from_qualifier(&self, name: &Qualifier) -> Result<SymbolRef> {
        match name.parent {
            None => self.get_symbol_ref_from_name(name.segment.as_ref()),
            Some(_) => Err(format!("qualified path not supported yet: {name:?}").into()) // TODO
        }
    }
}

impl ResolverContext for Module {
    fn get_symbol(&self, r: &SymbolRef) -> Result<&Symbol> {
        self.get_by_ref(r.clone())
    }
}

impl Module {
    pub fn parse_tokens(
        name: impl Into<String>,
        mut tokens: Vec<Token>,
    ) -> Result<Self> {
        let mut b = ModuleBuilder::new(name);
        while !tokens.is_empty() {
            b = b.add_declaration(parse_declaration(&mut tokens)?)
        }
        b.build()
    }
}

pub struct ModuleBuilder<'a> {
    module: Module,
    loading: Vec<PendingProcessing<'a>>,
    bodies: Vec<(u32, BlockExpression<'a>)>,
}

impl ModuleBuilder<'static> {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            module: Module {
                name: name.into(),
                symbols: Default::default(),
                by_name: Default::default(),
            },
            loading: Default::default(),
            bodies: Default::default(),
        }
    }
}

impl<'a> ModuleBuilder<'a> {
    fn allocate(&mut self, pending: PendingProcessing<'a>) -> u32 {
        let name = pending.meta();
        let id = self.loading.len() as u32;
        self.loading.push(pending);
        if let Some((name, typ)) = name {
            self.module.by_name.insert(name, (id, typ));
        }
        id
    }

    pub fn add_declaration<'b: 'a>(mut self, symbol: impl Into<PendingSymbol<'a>>) -> ModuleBuilder<'b>
        where 'a: 'b {
        let PendingSymbol { symbol, body } = symbol.into();
        let id = self.allocate(symbol);
        if let Some(body) = body {
            self.bodies.push((id, body));
        }
        self
    }

    pub fn add_struct<'b: 'a>(mut self, symbol: Struct<'b>) -> ModuleBuilder<'b>
        where 'a: 'b {
        self.allocate(PendingProcessing::Struct(symbol));
        self
    }

    pub fn add_enum<'b: 'a>(mut self, symbol: Enum<'b>) -> ModuleBuilder<'b>
        where 'a: 'b {
        self.allocate(PendingProcessing::Enum(symbol));
        self
    }

    pub fn add_function<'b: 'a>(mut self, symbol: FunctionDeclaration<'b>) -> ModuleBuilder<'b>
        where 'a: 'b {
        self.add_declaration(Declaration::Function(symbol))
    }

    pub fn build(mut self) -> Result<Module> {
        for processing in self.loading {
            let symbol = processing.load(&self.module)?;
            self.module.symbols.push(symbol)
        }
        for (id, body) in self.bodies {
            let Symbol::Function(ref f) = self.module.symbols[id as usize] else { return Err("only function can have bodies!")?; };
            let body = body.to_fn_body(f.params(), &self.module)?;
            let Symbol::Function(ref mut f) = self.module.symbols[id as usize] else { return Err("only function can have bodies!")?; };
            f.body = Some(body)
        }
        Ok(self.module)
    }
}