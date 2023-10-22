use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::collections::btree_map::Values;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt::{Debug, Display, Formatter};

use crate::bytecode::{Body, ValueKind, ValueRef};
use crate::parser::{BlockExpression, Enum, Expression, Fields, FunctionDeclaration, Identifier, Struct, Type, Visibility};

type Result<T> = core::result::Result<T, Cow<'static, str>>;

trait SymbolLoader {
    type Symbol;

    fn load(self, ctx: &impl LoaderContext) -> Result<Self::Symbol>;
}

trait LoaderContext {
    fn get_symbol_from_ref(&self, r: &SymbolRef) -> Result<&Symbol>;
    fn get_symbol_ref_from_name(&self, name: &str) -> Result<SymbolRef>;

    fn get_symbol_from_name(&self, name: &str) -> Result<&Symbol> {
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
pub struct SymbolRef(u32, SymbolType);

#[cfg(test)]
impl SymbolRef {
    pub fn new(id: u32, typ: SymbolType) -> SymbolRef {
        Self(id, typ)
    }
}

impl Eq for SymbolRef {}

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
    pub fn name(&self) -> Option<&str> {
        Some(match self {
            Symbol::Struct(sym) => sym.name.as_ref(),
            Symbol::Enum(sym) => sym.name.as_ref(),
            Symbol::Function(sym) => sym.name.as_ref(),
        })
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
            fields: load_struct_fields(ctx, self.body)?,
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
        typ: load_type(ctx, x.typ)?,
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
                            let s = ctx.get_symbol_ref_from_name(s.as_ref()).or(Err("Function not defined"))?;
                            if !matches!(s.1, SymbolType::Struct) {
                                return Err("Symbol isn't callable")?;
                            }
                            let fs = fs
                                .iter()
                                .map(|(n, i)| Ok((n.as_ref().to_string(), visit_expr(b, ctx, v, i)?)))
                                .collect::<Result<Vec<_>>>()?;
                            b.create(s, fs)?
                        }
                    })
                }

                fn visit_block_inner(b: &mut Body, ctx: &impl LoaderContext, v: &dyn Scope, e: &BlockExpression) -> Result<()> {
                    for e in &e.expressions {
                        visit_expr(b, ctx, v, e)?;
                    }
                    let remainder = if let Some(e) = &e.remainder {
                        visit_expr(b, ctx, v, e)?
                    } else {
                        b.const_unit()?
                    };

                    b.yield_expr(remainder)?;

                    Ok(())
                }

                fn visit_block(b: &mut Body, ctx: &impl LoaderContext, v: &dyn Scope, e: &BlockExpression) -> Result<(Vec<ValueRef>, Box<Body>)> {
                    b.derive(|b| {
                        let ns = BlockScope(v, RefCell::new(Default::default()));

                        visit_block_inner(b, ctx, &ns, e)?;

                        Ok(
                            ns.1.take()
                                .keys()
                                .cloned()
                                .collect()
                        )
                    })
                }

                visit_block_inner(b, ctx, &params, &self.body)?;

                Ok(())
            },
        )?;
        let typ = self.ret_type.map(|t| load_type(ctx, t)).unwrap_or(Ok(TypeRef::Primitive(PrimitiveType::Unit)))?;
        let body_type = body.yield_type();
        // let ret_type = type_to_value_kind(self.ret_type.unwrap_or(Type::Unit));
        let ret_type = type_ref_to_value_kind(&typ);
        if !ret_type.is_assignable_from(&body_type) {
            return Err(format!("cannot return type `{body_type:?}` when function is declared to return type `{ret_type:?}`"))?;
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
        _ => ValueKind::Never
    }
}

fn type_ref_to_value_kind(typ: &TypeRef) -> ValueKind {
    match typ {
        TypeRef::Primitive(PrimitiveType::Unit) => ValueKind::Unit,
        TypeRef::Primitive(PrimitiveType::I64) => ValueKind::I64,
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
}

impl LoaderContext for Module {
    fn get_symbol_from_ref(&self, r: &SymbolRef) -> Result<&Symbol> {
        self.get_by_ref(r.clone())
    }

    fn get_symbol_ref_from_name(&self, name: &str) -> Result<SymbolRef> {
        self.by_name
            .get(name)
            .cloned()
            .map(SymbolRef::from)
            .ok_or_else(|| format!("Symbol not defined: {:?}", name).into())
    }
}

impl ResolverContext for Module {
    fn get_symbol(&self, r: &SymbolRef) -> Result<&Symbol> {
        self.get_by_ref(r.clone())
    }
}

pub struct ModuleBuilder<'a> {
    module: Module,
    loading: Vec<PendingProcessing<'a>>,
}

impl ModuleBuilder<'static> {
    pub fn new(name: impl Into<String>) -> Self {
        ModuleBuilder {
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