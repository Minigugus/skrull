use alloc::borrow::{Cow, ToOwned};
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::collections::btree_map::Values;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use core::cmp::Ordering;
use core::fmt::{Debug, Display, Formatter};

use crate::bytecode::{Body, TerminatorOp, ValueKind, ValueRef};
use crate::lexer::Token;
use crate::parser::{BlockExpression, Declaration, Enum, Expression, Fields, FunctionDeclaration, Identifier, parse_declaration, Struct, Type, Visibility};

type Result<T> = core::result::Result<T, Cow<'static, str>>;

trait SymbolLoader {
    type Symbol;

    fn load(self, ctx: &impl LoaderContext) -> Result<Self::Symbol>;
}

trait LoaderContext {
    fn get_symbol_from_ref(&self, r: &SymbolRef) -> Result<&Symbol>;
    fn get_symbol_ref_from_name(&self, name: &(impl AsRef<str> + ?Sized)) -> Result<SymbolRef>;

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

pub enum PendingProcessing<'a> {
    Struct(Struct<'a>),
    Enum(Enum<'a>),
    Function(FunctionDeclaration<'a>),
}

impl<'a> PendingProcessing<'a> {
    pub fn meta(&self) -> Option<(String, SymbolType)> {
        Some(match self {
            PendingProcessing::Struct(sym) => (sym.name.as_ref().to_string(), SymbolType::Struct),
            PendingProcessing::Enum(sym) => (sym.name.as_ref().to_string(), SymbolType::Enum),
            PendingProcessing::Function(sym) => (sym.name.as_ref().to_string(), SymbolType::Function(Rc::new(type_to_value_kind(sym.ret_type.unwrap_or(Type::Unit))))),
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

impl<'a> From<FunctionDeclaration<'a>> for PendingProcessing<'a> {
    fn from(value: FunctionDeclaration<'a>) -> Self {
        Self::Function(value)
    }
}

impl<'a> From<Declaration<'a>> for PendingProcessing<'a> {
    fn from(value: Declaration<'a>) -> Self {
        match value {
            Declaration::Enum(v) => v.into(),
            Declaration::Function(v) => v.into(),
            Declaration::Struct(v) => v.into(),
        }
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

impl<'a> SymbolLoader for Struct<'a> {
    type Symbol = StructDef;

    fn load(self, ctx: &impl LoaderContext) -> Result<Self::Symbol> {
        Ok(StructDef {
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

fn load_named_field(ctx: &impl LoaderContext, x: crate::parser::NamedField) -> Result<NamedField> {
    Ok(NamedField {
        scope: Scope::from(x.visibility),
        name: x.name.as_ref().to_string(),
        typ: load_type(ctx, x.typ)
            .map_err(|e| format!("field '{}': {e}", x.name.as_ref()))?,
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

fn load_enum_variant(ctx: &impl LoaderContext, x: crate::parser::EnumVariant) -> Result<EnumVariant> {
    Ok(EnumVariant {
        name: x.name.as_ref().to_string(),
        fields: match x.fields {
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
            scope: Scope::from(self.visibility),
            name: self.name.as_ref().to_string(),
            variants: self.body
                .into_iter()
                .map(|v| load_enum_variant(ctx, v))
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl<'a> SymbolLoader for FunctionDeclaration<'a> {
    type Symbol = FunctionDef;

    fn load(self, ctx: &impl LoaderContext) -> Result<Self::Symbol> {
        let body = Body::new_with_dyn_params(
            self.parameters
                .iter()
                .map(|p| type_to_value_kind(p.typ.unwrap_or(Type::Unit)))
                .collect::<Vec<_>>()
                .as_ref(),
            |b, v| {
                trait Scope {
                    fn read_variable(&self, name: &Identifier) -> core::result::Result<ValueRef, &'static str>;
                }

                impl Scope for BTreeMap<String, ValueRef> {
                    fn read_variable(&self, name: &Identifier) -> core::result::Result<ValueRef, &'static str> {
                        self.get(name.as_ref()).cloned().ok_or("variable not defined in scope")
                    }
                }

                struct BlockScope<'a, T: ?Sized>(&'a T, RefCell<BTreeMap<ValueRef, ValueRef>>);

                impl<'a, T: Scope + ?Sized> Scope for BlockScope<'a, T> {
                    fn read_variable(&self, name: &Identifier) -> core::result::Result<ValueRef, &'static str> {
                        let original_ref = self.0.read_variable(name)?;
                        let mut mapping = self.1.borrow_mut();
                        let mapped_ref = original_ref.map(mapping.len());
                        Ok(mapping.entry(original_ref).or_insert(mapped_ref).clone())
                    }
                }

                let params: BTreeMap<String, ValueRef> = self.parameters
                    .iter()
                    .enumerate()
                    .map(|(i, p)| Ok((
                        p.name.as_ref().to_string(),
                        b.label(v[i].clone(), p.name.as_ref().to_string())?
                    )))
                    .collect::<Result<BTreeMap<_, _>>>()?;

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
                            let f = ctx.get_symbol_ref_from_name(f.as_ref()).or(Err("Function not defined"))?;
                            if !matches!(f.1, SymbolType::Function(_)) {
                                return Err("Symbol isn't callable")?;
                            }
                            let args = a
                                .iter()
                                .map(|e| visit_expr(b, ctx, v, e))
                                .collect::<Result<Vec<_>>>()?;
                            b.call(f, args)?
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

                visit_block_inner(b, ctx, &params, &self.body)
            },
        )?;
        let typ = self.ret_type.map(|t| load_type(ctx, t)).unwrap_or(Ok(TypeRef::Primitive(PrimitiveType::Unit)))?;
        let body_type = body.yield_type();
        // let ret_type = type_to_value_kind(self.ret_type.unwrap_or(Type::Unit));
        let ret_type = type_ref_to_value_kind(&typ);
        if !ret_type.is_assignable_from(&body_type) {
            let body_type = body_type.to_string(ctx);
            let ret_type = ret_type.to_string(ctx);
            return Err(format!("cannot return type `{body_type}` when function is declared to return type `{ret_type}`"))?;
        }
        Ok(FunctionDef {
            scope: Scope::from(self.visibility),
            name: self.name.as_ref().to_string(),
            params: self.parameters
                .into_iter()
                .map(|p| Ok((p.name.as_ref().to_string(), Parameter {
                    mutable: p.mutability.into(),
                    name: p.name.as_ref().to_string(),
                    typ: load_type(ctx, p.typ.expect("parameters still requires type parameters"))?,
                })))
                .collect::<Result<BTreeMap<String, _>>>()?,
            ret_type: typ,
            body,
        })
    }
}

fn type_to_value_kind(typ: Type) -> ValueKind {
    match typ {
        Type::Unit => ValueKind::Unit,
        Type::I64 => ValueKind::I64,
        Type::I16 => ValueKind::I16,
        Type::F64 => ValueKind::F64,
        Type::Usize => ValueKind::Usize,
        _ => ValueKind::Never
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

#[derive(Debug)]
pub struct StructDef {
    scope: Scope,
    name: String,
    fields: StructFields,
}

impl StructDef {
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
    scope: Scope,
    name: String,
    variants: Vec<EnumVariant>,
}

impl EnumDef {
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
    scope: Scope,
    name: String,
    params: BTreeMap<String, Parameter>,
    ret_type: TypeRef,
    body: Body,
}

impl FunctionDef {
    pub fn scope(&self) -> Scope {
        self.scope
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn params(&self) -> Values<'_, String, Parameter> {
        self.params.values()
    }

    pub fn ret_type(&self) -> &TypeRef {
        &self.ret_type
    }

    pub fn body(&self) -> &Body {
        &self.body
    }
}

#[derive(Debug)]
pub struct EnumVariant {
    name: String,
    fields: EnumVariantFields,
}

impl EnumVariant {
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
    scope: Scope,
    name: String,
    typ: TypeRef,
}

impl NamedField {
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
    Function(Rc<ValueKind>),
}

impl SymbolType {
    pub fn as_function_return_type(&self) -> Option<ValueKind> {
        match self {
            SymbolType::Function(k) => Some((**k).clone()),
            _ => None
        }
    }
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
        }
    }
}

impl<'a> ModuleBuilder<'a> {
    fn allocate(&mut self, pending: PendingProcessing<'a>) {
        let name = pending.meta();
        let id = self.loading.len() as u32;
        self.loading.push(pending);
        if let Some((name, typ)) = name {
            self.module.by_name.insert(name, (id, typ));
        }
    }

    pub fn add_declaration<'b: 'a>(mut self, symbol: impl Into<PendingProcessing<'a>>) -> ModuleBuilder<'b>
        where 'a: 'b {
        self.allocate(symbol.into());
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
        self.allocate(PendingProcessing::Function(symbol));
        self
    }

    pub fn build(mut self) -> Result<Module> {
        for processing in self.loading {
            let symbol = processing.load(&self.module)?;
            self.module.symbols.push(symbol)
        }
        Ok(self.module)
    }
}