use alloc::{format, vec};
use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt::{Debug, Formatter};

use crate::parser::parse_function_declaration;
use crate::types::{Module, ModuleBuilder, ResolverContext, StructFields, Symbol, SymbolRef, SymbolType};

type Result<T> = core::result::Result<T, Cow<'static, str>>;

#[derive(Clone, Eq, PartialEq)]
pub struct ValueRef {
    id: usize,
    param: bool,
    kind: ValueKind,
}

impl ValueRef {
    pub fn kind(&self) -> &ValueKind {
        &self.kind
    }

    pub fn map(&self, id: usize) -> Self {
        Self {
            id,
            param: true,
            kind: self.kind.clone(),
        }
    }
}

impl Debug for ValueRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Self { id, kind, param } = self;
        let param = if *param { "P" } else { "" };
        write!(f, "Ref(#{param}{id}: {kind:?})")
    }
}

impl PartialOrd<Self> for ValueRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Ord for ValueRef {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ValueKind {
    Never,
    Unit,
    Bool,
    I64,
    I16,
    F64,
    Usize,
    Type(SymbolRef),
}

impl ValueKind {
    pub fn is_assignable_to(&self, other: &ValueKind) -> bool {
        match other {
            _ if matches!(self, ValueKind::Never) => true,
            ValueKind::Never => false,
            ValueKind::Unit => matches!(*self, ValueKind::Never | ValueKind::Unit),
            ValueKind::Bool => matches!(*self, ValueKind::Never | ValueKind::Bool),
            ValueKind::I64 => matches!(*self, ValueKind::Never | ValueKind::I64),
            ValueKind::I16 => matches!(*self, ValueKind::Never | ValueKind::I16),
            ValueKind::F64 => matches!(*self, ValueKind::Never | ValueKind::F64),
            ValueKind::Usize => matches!(*self, ValueKind::Never | ValueKind::Usize),
            ValueKind::Type(t) => if let ValueKind::Type(f) = self {
                *f == *t
            } else {
                false
            },
        }
    }

    pub fn is_assignable_from(&self, src: &ValueKind) -> bool {
        match self {
            _ if matches!(src, ValueKind::Never) => true,
            ValueKind::Never => true,
            ValueKind::Unit => matches!(*self, ValueKind::Unit),
            ValueKind::Bool => matches!(*self, ValueKind::Bool),
            ValueKind::I64 => matches!(*self, ValueKind::I64),
            ValueKind::I16 => matches!(*self, ValueKind::I16),
            ValueKind::F64 => matches!(*self, ValueKind::F64),
            ValueKind::Usize => matches!(*self, ValueKind::Usize),
            ValueKind::Type(t) => if let ValueKind::Type(f) = src {
                *f == *t
            } else {
                false
            },
        }
    }

    pub fn is_number(&self) -> bool {
        matches!(
            self,
            ValueKind::Never |
            ValueKind::I64 |
            ValueKind::I16 |
            ValueKind::F64 |
            ValueKind::Usize
        )
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct MatchCaseOp {
    pub pattern: MatchPatternOp,
    pub body: (Box<[ValueRef]>, Box<Body>),
    pub guard: Option<(Box<[ValueRef]>, Box<Body>)>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum MatchPatternOp {
    Unit,
    Wildcard,
    Union(Vec<MatchPatternOp>),
    Variable(usize),
    NumberLiteral(i64),
    StringLiteral(String),
    IsTypeOrEnum(SymbolRefOrEnum),
    TupleStruct { typ: SymbolRefOrEnum, params: Vec<MatchPatternOp>, exact: bool },
    FieldStruct { typ: SymbolRefOrEnum, params: Vec<(String, MatchPatternOp)>, exact: bool },
}

#[derive(Debug, Eq, PartialEq)]
pub enum SymbolRefOrEnum {
    Type(SymbolRef),
    Enum(SymbolRef, String),
}

impl SymbolRefOrEnum {
    pub fn owner_type(&self) -> &SymbolRef {
        match self {
            SymbolRefOrEnum::Type(t) | SymbolRefOrEnum::Enum(t, _) => &t,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum Op {
    Label(ValueRef, String),
    ConstUnit,
    ConstI64(i64),
    Block((Box<[ValueRef]>, Box<Body>)),
    Neg(ValueRef),
    Add(ValueRef, ValueRef),
    Mul(ValueRef, ValueRef),
    Gt(ValueRef, ValueRef),
    If(ValueRef, (Box<[ValueRef]>, Box<Body>), (Box<[ValueRef]>, Box<Body>)),
    Call(SymbolRef, Box<[ValueRef]>),
    Create(SymbolRef, Box<[(String, ValueRef)]>),
    Match(ValueRef, Box<[MatchCaseOp]>),
}

#[derive(Debug, Eq, PartialEq)]
pub enum TerminatorOp {
    Yield(ValueRef),
}

#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    ops: Vec<Op>,
    terminator_op: Option<TerminatorOp>,
    params: Vec<ValueKind>,
    nb_params: usize,
    next_ref_id: usize,
    ret_kind: Option<ValueKind>,
}

impl Body {
    pub fn new_with_dyn_params(
        params: &[ValueKind],
        ops: impl FnOnce(&mut Body, &[ValueRef]) -> Result<TerminatorOp>,
    ) -> Result<Self> {
        let mut i = 0usize;
        let param_refs = params
            .iter()
            .cloned()
            .map(|kind| {
                let id = i;
                i += 1;
                ValueRef {
                    id,
                    param: true,
                    kind,
                }
            })
            .collect::<Vec<_>>();
        let mut body = Self {
            ops: Default::default(),
            terminator_op: None,
            params: param_refs.iter().map(|v| v.kind.clone()).collect(),
            nb_params: 0,
            next_ref_id: 0,
            ret_kind: None,
        };
        let terminator_op = (ops)(&mut body, &param_refs)?;
        body.terminator_op = Some(terminator_op);
        Ok(body)
    }

    fn new_with_params<const N: usize>(params: &[ValueKind; N]) -> (Self, [ValueRef; N]) {
        let mut i = 0usize;
        let param_refs = params.clone().map(|kind| {
            let id = i;
            i += 1;
            ValueRef {
                id,
                param: true,
                kind,
            }
        });
        (Self {
            ops: Default::default(),
            terminator_op: None,
            params: param_refs.iter().map(|v| v.kind.clone()).collect(),
            nb_params: 0,
            next_ref_id: 0,
            ret_kind: None,
        }, param_refs)
    }

    fn new_with_params_and_ops<const N: usize>(
        params: &[ValueKind; N],
        ops: impl FnOnce(&mut Body, [ValueRef; N]) -> Result<TerminatorOp>,
    ) -> Result<Self> {
        let (mut body, params) = Self::new_with_params(params);
        let terminator_op = (ops)(&mut body, params)?;
        body.terminator_op = Some(terminator_op);
        Ok(body)
    }

    pub fn derive(
        &self,
        ops: impl FnOnce(&mut Body) -> Result<(Box<[ValueRef]>, TerminatorOp)>,
    ) -> Result<(Box<[ValueRef]>, Box<Body>)> {
        let mut body = Self {
            ops: Default::default(),
            terminator_op: None,
            params: Default::default(),
            nb_params: 0,
            next_ref_id: 0,
            ret_kind: None,
        };
        let (params, terminal_op) = (ops)(&mut body)?;
        body.terminator_op = Some(terminal_op);
        body.params = params.iter().map(|r| r.kind.clone()).collect();
        Ok((params, Box::new(body)))
    }

    pub fn derive_with_params(
        &self,
        params: &[ValueKind],
        ops: impl FnOnce(&mut Body, &[ValueRef]) -> Result<(Box<[ValueRef]>, TerminatorOp)>,
    ) -> Result<(Box<[ValueRef]>, Box<Body>)> {
        let mut i = 0usize;
        let param_refs = params
            .iter()
            .cloned()
            .map(|kind| {
                let id = i;
                i += 1;
                ValueRef {
                    id,
                    param: true,
                    kind,
                }
            })
            .collect::<Vec<_>>();
        let mut body = Self {
            ops: Default::default(),
            terminator_op: None,
            params: Default::default(),
            nb_params: 0,
            next_ref_id: 0,
            ret_kind: None,
        };
        let (params, terminal_op) = (ops)(&mut body, param_refs.as_slice())?;
        body.terminator_op = Some(terminal_op);
        body.params = param_refs
            .iter()
            .chain(params.iter())
            .map(|r| r.kind.clone())
            .collect();
        Ok((params, Box::new(body)))
    }

    pub fn yield_type(&self) -> ValueKind {
        self.ret_kind.clone().unwrap_or(ValueKind::Unit)
    }

    fn push(&mut self, op: Op, kind: ValueKind) -> Result<ValueRef> {
        self.ops.push(op);
        let id = self.next_ref_id;
        self.next_ref_id += 1;
        Ok(ValueRef {
            id,
            param: false,
            kind,
        })
    }

    pub fn label(&mut self, value: ValueRef, label: impl Into<String>) -> Result<ValueRef> {
        let kind = value.kind.clone();
        self.push(Op::Label(value, label.into()), kind)
    }

    pub fn const_unit(&mut self) -> Result<ValueRef> {
        self.push(Op::ConstUnit, ValueKind::Unit)
    }

    pub fn const_i64(&mut self, value: i64) -> Result<ValueRef> {
        self.push(Op::ConstI64(value), ValueKind::I64)
    }

    pub fn neg(&mut self, value: ValueRef) -> Result<ValueRef> {
        if !value.kind.is_number() {
            Err(format!("arithmetic operation requires a numeric operand; got {:?}", value.kind))?;
        }

        self.push(Op::Neg(value), ValueKind::I64)
    }

    pub fn add(&mut self, left: ValueRef, right: ValueRef) -> Result<ValueRef> {
        if !(left.kind.is_number() && right.kind.is_number()) {
            Err(format!("arithmetic operation requires a numeric operand; got {:?} and {:?}", left.kind, right.kind))?;
        }

        self.push(Op::Add(left, right), ValueKind::I64)
    }

    pub fn mul(&mut self, left: ValueRef, right: ValueRef) -> Result<ValueRef> {
        if !(left.kind.is_number() && right.kind.is_number()) {
            Err(format!("arithmetic operation requires numeric operands; got {:?} and {:?}", left.kind, right.kind))?;
        }

        self.push(Op::Mul(left, right), ValueKind::I64)
    }

    pub fn yield_expr(&mut self, value: ValueRef) -> Result<TerminatorOp> {
        if let Some(kind) = &self.ret_kind {
            if *kind != value.kind {
                Err("return statements must have the same type")?;
            }
        }

        self.ret_kind = Some(value.kind.clone());
        Ok(TerminatorOp::Yield(value))
    }

    pub fn gt(&mut self, left: ValueRef, right: ValueRef) -> Result<ValueRef> {
        if !(left.kind.is_number() && right.kind.is_number()) {
            Err(format!("arithmetic operation requires a numeric operand; got {:?} and {:?}", left.kind, right.kind))?;
        }

        self.push(Op::Gt(left, right), ValueKind::Bool)
    }

    pub fn block(&mut self, block: (Box<[ValueRef]>, Box<Body>)) -> Result<ValueRef> {
        let kind = block.1.ret_kind.clone().unwrap_or(ValueKind::Unit);
        self.push(Op::Block(block), kind)
    }

    pub fn if_expr(
        &mut self,
        cond: ValueRef,
        on_true: (Box<[ValueRef]>, Box<Body>),
        on_false: (Box<[ValueRef]>, Box<Body>),
    ) -> Result<ValueRef> {
        if !matches!(cond.kind, ValueKind::Bool) {
            Err("conditional expression requires a boolean operand")?;
        }

        if on_true.1.ret_kind != on_false.1.ret_kind {
            Err("conditional branches don't return the same type")?;
        }

        let kind = on_true.1.ret_kind.clone().unwrap_or_else(|| ValueKind::Unit);
        self.push(Op::If(cond, on_true, on_false), kind)
    }

    pub fn call(&mut self, func: SymbolRef, ret_type: ValueKind, args: impl Into<Box<[ValueRef]>>) -> Result<ValueRef> {
        self.push(Op::Call(func, args.into()), ret_type)
    }

    pub fn create(&mut self, struct_ref: SymbolRef, fields: impl Into<Box<[(String, ValueRef)]>>) -> Result<ValueRef> {
        let kind = ValueKind::Type(struct_ref.clone());
        self.push(Op::Create(struct_ref, fields.into()), kind)
    }

    pub fn match_(&mut self, expr: ValueRef, cases: impl Into<Box<[MatchCaseOp]>>) -> Result<ValueRef> {
        let cases = cases.into();
        let kind = cases
            .iter()
            .flat_map(|op| op.body.1.ret_kind.clone())
            .next()
            .map(Ok)
            .unwrap_or_else(|| Err("at least 1 case is required in match expression"))?;
        self.push(Op::Match(expr, cases), kind)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Composed {
    Struct(Vec<Value>),
}

#[derive(Debug, Clone, PartialEq)]
enum Value {
    Never,
    Unit,
    Bool(bool),
    I64(i64),
    I16(i16),
    F64(f64),
    Usize(usize),
    Composed(Composed, SymbolRef),
}

impl Value {
    pub fn to_i16(&self) -> Result<i64> {
        match self {
            Value::I64(v) => Ok(v.clone()),
            _ => Err("expected a i64")?
        }
    }

    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Value::Bool(v) => Ok(v.clone()),
            _ => panic!("expected a bool")
        }
    }

    pub fn is_assignable_to(&self, kind: &ValueKind) -> bool {
        match kind {
            ValueKind::Never => matches!(self, Value::Never),
            ValueKind::Unit => matches!(self, Value::Never | Value::Unit),
            ValueKind::Bool => matches!(self, Value::Never | Value::Bool(_)),
            ValueKind::I64 => matches!(self, Value::Never | Value::I64(_)),
            ValueKind::I16 => matches!(self, Value::Never | Value::I16(_)),
            ValueKind::F64 => matches!(self, Value::Never | Value::F64(_)),
            ValueKind::Usize => matches!(self, Value::Never | Value::Usize(_)),
            ValueKind::Type(t) => if let Value::Composed(_, f) = self {
                *f == *t
            } else {
                matches!(self, Value::Never)
            }
        }
    }

    pub fn kind(&self) -> ValueKind {
        match self {
            Value::Never => ValueKind::Never,
            Value::Unit => ValueKind::Unit,
            Value::Bool(_) => ValueKind::Bool,
            Value::I64(_) => ValueKind::I64,
            Value::I16(_) => ValueKind::I16,
            Value::F64(_) => ValueKind::F64,
            Value::Usize(_) => ValueKind::Usize,
            Value::Composed(_, f) => ValueKind::Type(f.clone()),
        }
    }
}

#[derive(Debug)]
struct Heap<'a>(&'a [Value], Vec<Value>);

impl<'a> Heap<'a> {
    pub fn get(&self, r: &ValueRef) -> Option<Value> {
        if r.param {
            self.0.get(r.id).cloned()
        } else {
            self.1.get(r.id).cloned()
        }
    }

    pub fn push(&mut self, v: Value) {
        self.1.push(v)
    }
}

fn eval(ctx: &impl ResolverContext, body: &Body, params: &[Value]) -> Result<Value> {
    if body.params.len() != params.len() {
        Err(format!("invalid number of parameters: expected {} but got {}", body.params.len(), params.len()))?;
    }
    for (i, p) in params.iter().enumerate() {
        let expected = &body.params[i];
        if !p.is_assignable_to(expected) {
            let actual = p.kind();
            Err(format!("arguments not assignable to parameters types: {actual:?} vs {expected:?}"))?;
        }
    }
    let mut heap = Heap(params, Vec::default());
    for op in &body.ops {
        heap.push(match op {
            Op::ConstUnit => Value::Unit,
            Op::ConstI64(v) => Value::I64(*v),
            Op::Label(v, _) => heap.get(v).ok_or("invalid ref in Label op")?.clone(),
            Op::Neg(v) => {
                let v = heap.get(v).ok_or("invalid ref in Neg op")?.to_i16()?;
                Value::I64(-v)
            }
            Op::Add(l, r) => {
                let l = heap.get(l).ok_or("invalid ref in Add op")?.to_i16()?;
                let r = heap.get(r).ok_or("invalid ref in Add op")?.to_i16()?;
                Value::I64(l + r)
            }
            Op::Mul(l, r) => {
                let l = heap.get(l).ok_or("invalid ref in Mul op")?.to_i16()?;
                let r = heap.get(r).ok_or("invalid ref in Mul op")?.to_i16()?;
                Value::I64(l * r)
            }

            Op::Gt(l, r) => {
                let l = heap.get(l).ok_or("invalid ref in Gt op")?.to_i16()?;
                let r = heap.get(r).ok_or("invalid ref in Gt op")?.to_i16()?;
                Value::Bool(l > r)
            }
            Op::Block((refs, body)) => {
                let params = refs
                    .iter()
                    .map(|v| heap.get(v).ok_or("invalid ref in If op".into()))
                    .collect::<Result<Vec<_>>>()?;
                eval(ctx, body, params.as_slice())?
            }
            Op::If(c, t, f) => {
                let c = heap.get(c).ok_or("invalid ref in If op")?.to_bool()?;
                let (refs, body) = if c { t } else { f };
                let params = refs
                    .iter()
                    .map(|v| heap.get(v).ok_or("invalid ref in If op".into()))
                    .collect::<Result<Vec<_>>>()?;
                eval(ctx, body, params.as_slice())?
            }
            Op::Call(f, a) => {
                let f = ctx.get_symbol(&f).or(Err("function symbol not found"))?;

                let Symbol::Function(f) = f else { return Err("symbol not a function")?; };

                let params = a
                    .iter()
                    .map(|v| heap.get(v).ok_or("invalid ref in Call op".into()))
                    .collect::<Result<Vec<_>>>()?;

                if let Some(body) = f.body() {
                    eval(ctx, body, params.as_slice())?
                } else {
                    let n = f.name();
                    return Err(format!("function {n} doesn't have a body"))?
                }
            }
            Op::Create(sr, fs) => {
                let s = ctx.get_symbol(&sr).or(Err("struct symbol not found"))?;

                let Symbol::Struct(s) = s else { return Err("symbol not a struct")?; };

                let mut fields = match s.fields() {
                    StructFields::Named(n) => Ok(n),
                    StructFields::Tuple(_) => Err("tuple structs cannot be constructed as a named struct"),
                    StructFields::Unit => Err("unit structs cannot be constructed as a named struct")
                }?;

                let mut missing_fields = vec![];

                let mut initializers = fs.iter()
                    .cloned()
                    .collect::<BTreeMap<_, _>>();

                let mut values: Vec<Value> = Vec::with_capacity(fields.len());

                for f in fields {
                    let value = match initializers.remove(f.name()) {
                        Some(value) => heap
                            .get(&value)
                            .ok_or_else(|| format!("invalid ref in field initializer '{}' for struct '{}'", f.name(), s.name()))?,
                        None => {
                            missing_fields.push(f.name());
                            continue;
                        }
                    };
                    values.push(value);
                }

                if !initializers.is_empty() {
                    return Err(format!("fields '{}' don't exist on struct '{}'", initializers.keys().map(AsRef::as_ref).collect::<Vec<_>>().join("', '"), s.name()).into());
                }
                if !missing_fields.is_empty() {
                    missing_fields.reverse();
                    return Err(format!("missing fields '{}' while creating a '{}'", missing_fields.join("', '"), s.name()).into());
                }

                Value::Composed(Composed::Struct(values), sr.clone())
            }
            Op::Match(expr, cases) => {
                fn eval_pattern(
                    value: &Value,
                    pattern: &MatchPatternOp,
                    ctx: &impl ResolverContext,
                    heap: &Heap,
                    vars: &mut Vec<Value>,
                ) -> Result<bool> {
                    Ok(match pattern {
                        MatchPatternOp::Unit => matches!(value.kind(), ValueKind::Unit),
                        MatchPatternOp::Wildcard => true,
                        MatchPatternOp::Union(p) => {
                            for p in p {
                                if eval_pattern(value, p, ctx, heap, vars)? {
                                    return Ok(true);
                                }
                            }
                            false
                        }
                        MatchPatternOp::Variable(id) => {
                            if *id >= vars.len() {
                                vars.resize(id + 1, Value::Never)
                            }
                            vars[*id] = value.clone();
                            true
                        }
                        MatchPatternOp::NumberLiteral(n) => match value {
                            Value::I64(actual) => *n == *actual,
                            Value::I16(actual) => *n == *actual as i64,
                            Value::F64(actual) => *n == *actual as i64,
                            _ => false
                        },
                        MatchPatternOp::StringLiteral(_) => false,
                        MatchPatternOp::IsTypeOrEnum(_) => false,
                        MatchPatternOp::TupleStruct { .. } => false,
                        MatchPatternOp::FieldStruct { typ, params, exact } => {
                            let s = match typ {
                                SymbolRefOrEnum::Type(s) => ctx.get_symbol(s)?,
                                SymbolRefOrEnum::Enum(_, _) => return Ok(false)
                            };
                            let name = s.name();
                            let s = match s {
                                Symbol::Struct(s) => s.fields(),
                                _ => return Ok(false)
                            };
                            let s = match s {
                                StructFields::Named(s) => s,
                                _ => return Ok(false)
                            };
                            let fields_by_name = s.iter()
                                .enumerate()
                                .map(|(i, f)| (f.name(), i))
                                .collect::<BTreeMap<_, _>>();
                            let s = match value {
                                Value::Composed(s, _) => s,
                                _ => return Ok(false)
                            };
                            let values = match s {
                                Composed::Struct(s) => s
                            };
                            for (field, pattern) in params {
                                let Some(id) = fields_by_name.get(field.as_str()) else { return Err(format!("Field {field} does not exist on struct {name}"))?; };
                                let Some(value) = values.get(*id) else { return Err(format!("invalid ref in struct match pattern: {name}.{field}"))? };
                                if !eval_pattern(value, pattern, ctx, heap, vars)? {
                                    return Ok(false);
                                }
                            }
                            true
                            // match typ {
                            //     SymbolRefOrEnum::Enum(e, v) => {
                            //         let e = ctx.get_symbol(e)?;
                            //         match e {
                            //             Symbol::Enum(EnumDef { variants, .. }) => {
                            //                 match value {
                            //                     Value::Composed(e, _) => match e {
                            //                         Composed::Struct(s) => s.
                            //                     }
                            //                     _ => false
                            //                 }
                            //             }
                            //             _ => false
                            //         }
                            //     }
                            //     _ => false,
                            // }
                        }
                    })
                }

                let mut vars = vec![];
                for x in cases.as_ref().iter() {
                    let value = heap
                        .get(expr)
                        .ok_or_else(|| format!("invalid ref in match pattern"))?;
                    if eval_pattern(&value, &x.pattern, ctx, &heap, &mut vars)? {
                        let prettyvars = format!("{vars:?}");
                        let prettyparams = format!("{:?}", x.body.0);
                        let prettyheap = format!("{heap:?}");
                        let params = vars
                            .into_iter()
                            .map(Ok)
                            .chain(x.body.0
                                .iter()
                                .map(|v| heap.get(v).ok_or(format!("invalid ref in Match op: {v:?}, {prettyvars}, {prettyparams}, {prettyheap}").into())))
                            .collect::<Result<Vec<_>>>()?;
                        return eval(ctx, &x.body.1, params.as_slice());
                    }
                }

                Value::Never
            }
        });
    }
    if let Some(op) = &body.terminator_op {
        match op {
            TerminatorOp::Yield(ref v) => return heap.get(v).ok_or("invalid ref in Yield terminator op".into())
        }
    }
    Err("missing Return op")?
}

#[test]
fn it_works() -> Result<()> {
    let on_true = Body::new_with_params_and_ops(
        &[ValueKind::I64],
        |body, [_1]| {
            body.yield_expr(_1)
        },
    )?;

    let on_false = Body::new_with_params_and_ops(
        &[],
        |body, []| {
            let _1 = body.const_i64(0)?;
            body.yield_expr(_1)
        },
    )?;

    let body = Body::new_with_params_and_ops(
        &[ValueKind::I64],
        |body, [_1]| {
            let _2 = body.label(_1, "x")?;
            let _3 = body.const_i64(2)?;
            let _4 = body.neg(_2)?;
            let _5 = body.add(_4, _3)?;
            let _6 = body.label(_5.clone(), "y")?;
            let _7 = body.const_i64(0)?;
            let _8 = body.gt(_5, _7)?;
            let _9 = body.if_expr(_8, (vec![_6].into_boxed_slice(), Box::new(on_true)), (vec![].into_boxed_slice(), Box::new(on_false)))?;
            body.yield_expr(_9)
        },
    )?;

    let value = eval(&(), &body, &[Value::I64(-4)]);

    assert_eq!(Ok(Value::I64(6)), value);

    let value = eval(&(), &body, &[Value::I64(4)]);

    assert_eq!(Ok(Value::I64(0)), value);

    Ok(())
}

#[test]
fn it_prints_functions_in_rust() -> Result<()> {
    use crate::lexer::Token;

    //language=rust
    let mut tokens = Token::parse_ascii(r#"
pub fn life(mut a: i64) -> i64 {
  40 + 2 * if a > 40 { (1) + { (4) } } else { 0 }
}
"#)?;

    let root = parse_function_declaration(&mut tokens)?;

    let module = ModuleBuilder::new("my_first_module")
        .add_function(root)
        .build()?;

    let res = if let Some(Symbol::Function(f)) = module.get_by_name("life") {
        let Some(body) = f.body() else { return Err("missing body")? };
        eval(&module, body, &[Value::I64(41)])?
    } else {
        Value::Unit
    };

    assert_eq!(Value::I64(50), res);

    Ok(())
}

#[test]
fn it_runs_recursive_fibonacci() -> Result<()> {
    use crate::lexer::Token;

    //language=rust
    let mut tokens = Token::parse_ascii(r#"
pub fn fib(n: i64) -> i64 {
  if 2 > n {
    n
  } else {
    fib(n + -1) + fib(n + -2)
  }
}
"#)?;

    let root = parse_function_declaration(&mut tokens)?;

    let module = ModuleBuilder::new("my_first_module")
        .add_function(root)
        .build()?;

    if let Some(Symbol::Function(f)) = module.get_by_name("fib") {
        let Some(body) = f.body() else { return Err("missing body")? };
        assert_eq!([
                       eval(&module, body, &[Value::I64(26)])?
                   ], [
                       Value::I64(121393)
                   ]);
    }

    Ok(())
}

#[test]
fn it_runs_fn_with_composed_types() -> Result<()> {
    use crate::lexer::Token;

    //language=rust
    let tokens = Token::parse_ascii(r#"
fn new_point(x: i64, y: i64) -> Point {
  Point {
    x,
    y: y + -1
  }
}

pub fn sum(x: i64, y: i64) -> i64 {
  maybe_get_x(Some {
    point: new_point(x, y)
  })
}

// declaration order shouldn't matter

struct Point {
  x: i64,
  y: i64,
}

struct Some { point: Point }

pub fn get_x(point: Point) -> i64 {
  match point {
    Point { x, y } => x + y
  }
}

pub fn maybe_get_x(maybe_point: Some) -> i64 {
  match maybe_point {
    42 => get_x(Point { x: 1, y: 2 }),
    Some { point: Point { x: 41, y: _ } } => -1,
    Some { point: p } => get_x(p),
  }
}
"#)?;

    let module = Module::parse_tokens("my_first_module", tokens)?;

    let Some(point_ref @ SymbolRef(_, SymbolType::Struct)) = module.get_ref_by_name("Point") else {
        return Err("'Point' is supposed to be declared as a struct")?;
    };

    let body = get_function_body(&module, "new_point")?;
    assert_eq!(
        [
            eval(&module, body, &[Value::I64(0), Value::I64(1)])?,
        ], [
            Value::Composed(
                Composed::Struct(vec![Value::I64(0), Value::I64(0)]),
                point_ref.clone(),
            )
        ]
    );

    let body = get_function_body(&module, "get_x")?;
    assert_eq!(
        [
            eval(&module, body, &[Value::Composed(
                Composed::Struct(vec![Value::I64(42), Value::I64(1337)]),
                point_ref,
            )])?,
        ], [
            Value::I64(1379)
        ]
    );

    let body = get_function_body(&module, "sum")?;
    assert_eq!(
        [
            eval(&module, body, &[Value::I64(42), Value::I64(1338)])?,
        ], [
            Value::I64(1379)
        ]
    );

    Ok(())
}

fn get_function_body<'a>(module: &'a Module, name: &str) -> Result<&'a Body> {
    let Some(Symbol::Function(f)) = module.get_by_name(name) else {
        return Err(format!("'{name}' is supposed to be a function"))?;
    };

    f.body().ok_or("missing function body".into())
}
