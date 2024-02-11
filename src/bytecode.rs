use alloc::{format, vec};
use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt::{Debug, Formatter};

use crate::parser::{parse_declaration, parse_function_declaration};
use crate::types::{ModuleBuilder, ResolverContext, StructFields, Symbol, SymbolRef, SymbolType};

type Result<T> = core::result::Result<T, Cow<'static, str>>;

#[derive(Clone, Eq, PartialEq)]
pub struct ValueRef {
    id: usize,
    param: bool,
    kind: ValueKind,
}

impl ValueRef {
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
enum Op {
    Label(ValueRef, String),
    ConstUnit,
    ConstI64(i64),
    Block((Vec<ValueRef>, Box<Body>)),
    Neg(ValueRef),
    Add(ValueRef, ValueRef),
    Mul(ValueRef, ValueRef),
    Gt(ValueRef, ValueRef),
    If(ValueRef, (Vec<ValueRef>, Box<Body>), (Vec<ValueRef>, Box<Body>)),
    Call(SymbolRef, Vec<ValueRef>),
    Create(SymbolRef, Vec<(String, ValueRef)>),
    Yield(ValueRef),
}

#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    ops: Vec<Op>,
    params: Vec<ValueKind>,
    nb_params: usize,
    next_ref_id: usize,
    ret_kind: Option<ValueKind>,
}

impl Body {
    pub fn new_with_dyn_params(
        params: &[ValueKind],
        ops: impl FnOnce(&mut Body, &[ValueRef]) -> Result<()>,
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
            params: param_refs.iter().map(|v| v.kind.clone()).collect(),
            nb_params: 0,
            next_ref_id: 0,
            ret_kind: None,
        };
        (ops)(&mut body, &param_refs)?;
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
            params: param_refs.iter().map(|v| v.kind.clone()).collect(),
            nb_params: 0,
            next_ref_id: 0,
            ret_kind: None,
        }, param_refs)
    }

    fn new_with_params_and_ops<const N: usize>(
        params: &[ValueKind; N],
        ops: impl FnOnce(&mut Body, [ValueRef; N]) -> Result<()>,
    ) -> Result<Self> {
        let (mut body, params) = Self::new_with_params(params);
        (ops)(&mut body, params)?;
        Ok(body)
    }

    pub fn derive(
        &self,
        ops: impl FnOnce(&mut Body) -> Result<Vec<ValueRef>>,
    ) -> Result<(Vec<ValueRef>, Box<Body>)> {
        let mut body = Self {
            ops: Default::default(),
            params: Default::default(),
            nb_params: 0,
            next_ref_id: 0,
            ret_kind: None,
        };
        let params = (ops)(&mut body)?;
        body.params = params.iter().map(|r| r.kind.clone()).collect();
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
            Err(format!("arithmetic operation requires a numeric operand; got {:?} and {:?}", left.kind, right.kind))?;
        }

        self.push(Op::Mul(left, right), ValueKind::I64)
    }

    pub fn yield_expr(&mut self, value: ValueRef) -> Result<ValueRef> {
        if let Some(kind) = &self.ret_kind {
            if *kind != value.kind {
                Err("return statements must have the same type")?;
            }
        }

        self.ret_kind = Some(value.kind.clone());
        self.push(Op::Yield(value), ValueKind::Never)
    }

    pub fn gt(&mut self, left: ValueRef, right: ValueRef) -> Result<ValueRef> {
        if !(left.kind.is_number() && right.kind.is_number()) {
            Err(format!("arithmetic operation requires a numeric operand; got {:?} and {:?}", left.kind, right.kind))?;
        }

        self.push(Op::Gt(left, right), ValueKind::Bool)
    }

    pub fn block(&mut self, block: (Vec<ValueRef>, Box<Body>)) -> Result<ValueRef> {
        let kind = block.1.ret_kind.clone().unwrap_or(ValueKind::Unit);
        self.push(Op::Block(block), kind)
    }

    pub fn if_expr(
        &mut self,
        cond: ValueRef,
        on_true: (Vec<ValueRef>, Box<Body>),
        on_false: (Vec<ValueRef>, Box<Body>),
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

    pub fn call(&mut self, func: SymbolRef, args: impl Into<Vec<ValueRef>>) -> Result<ValueRef> {
        let Some(k) = func.typ().as_function_return_type() else { return Err("symbol isn't callable")?; };
        self.push(Op::Call(func, args.into()), k)
    }

    pub fn create(&mut self, struct_ref: SymbolRef, fields: impl Into<Vec<(String, ValueRef)>>) -> Result<ValueRef> {
        let kind = ValueKind::Type(struct_ref.clone());
        self.push(Op::Create(struct_ref, fields.into()), kind)
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
}

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
        if !p.is_assignable_to(&body.params[i]) {
            Err("arguments not assignable to parameters types")?;
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

            Op::Yield(v) => return Ok(heap.get(v).ok_or("invalid ref in Return op")?.clone()),
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

                eval(ctx, f.body(), params.as_slice())?
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
        });
    }
    Err("missing Return op")?
}

#[test]
fn it_works() -> Result<()> {
    let on_true = Body::new_with_params_and_ops(
        &[ValueKind::I64],
        |body, [_1]| {
            body.yield_expr(_1)?;
            Ok(())
        },
    )?;

    let on_false = Body::new_with_params_and_ops(
        &[],
        |body, []| {
            let _1 = body.const_i64(0)?;
            body.yield_expr(_1)?;
            Ok(())
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
            let _9 = body.if_expr(_8, (vec![_6], Box::new(on_true)), (vec![], Box::new(on_false)))?;
            let _a = body.yield_expr(_9)?;
            Ok(())
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
        // assert_eq!(f.body().ops, Vec::default());
        eval(&module, f.body(), &[Value::I64(41)])?
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
        let body = f.body();
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
    let mut tokens = Token::parse_ascii(r#"
fn new_point(x: i64, y: i64) -> Point {
  Point {
    x,
    y: y + -1
  }
}

// declaration order shouldn't matter

struct Point {
  x: i64,
  y: i64,
}
"#)?;

    let module = ModuleBuilder::new("my_first_module")
        .add_declaration(parse_declaration(&mut tokens)?)
        .add_declaration(parse_declaration(&mut tokens)?)
        .build()?;

    let Some(point_ref @ SymbolRef(_, SymbolType::Struct)) = module.get_ref_by_name("Point") else {
        return Err("'Point' is supposed to be declared as a struct")?;
    };

    let Some(Symbol::Function(f)) = module.get_by_name("new_point") else {
        return Err("'new_point' is supposed to be a function")?;
    };

    let body = f.body();
    assert_eq!(
        [
            eval(&module, body, &[Value::I64(0), Value::I64(1)])?,
        ], [
            Value::Composed(
                Composed::Struct(vec![Value::I64(0), Value::I64(0)]),
                point_ref,
            )
        ]
    );

    Ok(())
}
