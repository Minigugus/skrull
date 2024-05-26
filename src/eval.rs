use alloc::{format, vec};
use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use crate::bytecode::{SkBody, SkMatchPatternOp, SkOp, SymbolRefOrEnum, SkTerminatorOp, SkValueKind, ValueRef};
use crate::parser::parse_function_declaration;
use crate::types::{Module, ModuleBuilder, ResolverContext, StructFields, Symbol, SymbolRef, SymbolType};

type Result<T> = core::result::Result<T, Cow<'static, str>>;

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

    pub fn is_assignable_to(&self, kind: &SkValueKind) -> bool {
        match kind {
            SkValueKind::Never => matches!(self, Value::Never),
            SkValueKind::Unit => matches!(self, Value::Never | Value::Unit),
            SkValueKind::Bool => matches!(self, Value::Never | Value::Bool(_)),
            SkValueKind::I64 => matches!(self, Value::Never | Value::I64(_)),
            SkValueKind::I16 => matches!(self, Value::Never | Value::I16(_)),
            SkValueKind::F64 => matches!(self, Value::Never | Value::F64(_)),
            SkValueKind::Usize => matches!(self, Value::Never | Value::Usize(_)),
            SkValueKind::Type(t) => if let Value::Composed(_, f) = self {
                *f == *t
            } else {
                matches!(self, Value::Never)
            }
        }
    }

    pub fn kind(&self) -> SkValueKind {
        match self {
            Value::Never => SkValueKind::Never,
            Value::Unit => SkValueKind::Unit,
            Value::Bool(_) => SkValueKind::Bool,
            Value::I64(_) => SkValueKind::I64,
            Value::I16(_) => SkValueKind::I16,
            Value::F64(_) => SkValueKind::F64,
            Value::Usize(_) => SkValueKind::Usize,
            Value::Composed(_, f) => SkValueKind::Type(f.clone()),
        }
    }
}

#[derive(Debug)]
struct Heap<'a> {
    depth: usize,
    parent: Option<&'a Heap<'a>>,
    params: &'a [Value],
    locals: Vec<Value>,
}

impl<'a> Heap<'a> {
    pub fn root(params: &'a [Value]) -> Self {
        Self {
            depth: 0,
            parent: None,
            params,
            locals: vec![],
        }
    }

    pub fn nested<'b>(&'b self, params: &'a [Value]) -> Heap<'b> where 'b: 'a {
        Self {
            depth: self.depth + 1,
            parent: Some(self),
            params,
            locals: vec![],
        }
    }

    pub fn get(&self, r: &ValueRef) -> Option<Value> {
        match r {
            ValueRef::Parameter(bd, _, pi, _) if *bd == self.depth => self.params.get(*pi).cloned(),
            ValueRef::Local(bd, _, oi, _) if *bd == self.depth => self.locals.get(*oi).cloned(),
            r => self.parent.and_then(|p| p.get(r))
        }
    }

    pub fn push(&mut self, v: Value) {
        self.locals.push(v)
    }
}

fn eval(ctx: &impl ResolverContext, body: &SkBody, params_or_heap: core::result::Result<&[Value], Heap>) -> Result<Value> {
    let mut heap = match params_or_heap {
        Err(heap) => heap,
        Ok(params) => {
            if body.params().len() != params.len() {
                Err(format!("invalid number of parameters: expected {} but got {}", body.params().len(), params.len()))?;
            }
            for (i, p) in params.iter().enumerate() {
                let expected = &body.params()[i];
                if !p.is_assignable_to(expected) {
                    let actual = p.kind();
                    Err(format!("arguments not assignable to parameters types: {actual:?} vs {expected:?}"))?;
                }
            }

            Heap::root(params)
        }
    };
    for op in body.ops() {
        heap.push(match op {
            SkOp::ConstUnit => Value::Unit,
            SkOp::ConstBool(v) => Value::Bool(*v),
            SkOp::ConstI64(v) => Value::I64(*v),
            SkOp::Label(v, _) => heap.get(v).ok_or("invalid ref in Label op")?.clone(),
            SkOp::Neg(v) => {
                let v = heap.get(v).ok_or("invalid ref in Neg op")?.to_i16()?;
                Value::I64(-v)
            }
            SkOp::Add(l, r) => {
                let l = heap.get(l).ok_or("invalid ref in Add op")?.to_i16()?;
                let r = heap.get(r).ok_or("invalid ref in Add op")?.to_i16()?;
                Value::I64(l + r)
            }
            SkOp::Mul(l, r) => {
                let l = heap.get(l).ok_or("invalid ref in Mul op")?.to_i16()?;
                let r = heap.get(r).ok_or("invalid ref in Mul op")?.to_i16()?;
                Value::I64(l * r)
            }

            SkOp::Gt(l, r) => {
                let l = heap.get(l).ok_or("invalid ref in Gt op")?.to_i16()?;
                let r = heap.get(r).ok_or("invalid ref in Gt op")?.to_i16()?;
                Value::Bool(l > r)
            }
            SkOp::Block(body) => {
                eval(ctx, body, Err(heap.nested(&[])))?
            }
            SkOp::If(c, t, f) => {
                let c = heap.get(c).ok_or("invalid ref in If op")?.to_bool()?;
                let b = if c { t } else { f };
                eval(ctx, b, Err(heap.nested(&[])))?
            }
            SkOp::Call(f, _, a) => {
                let f = ctx.get_symbol(&f).or(Err("function symbol not found"))?;

                let Symbol::Function(f) = f else { return Err("symbol not a function")?; };

                let params = a
                    .iter()
                    .map(|v| heap.get(v).ok_or("invalid ref in Call op".into()))
                    .collect::<Result<Vec<_>>>()?;

                if let Some(body) = f.body() {
                    eval(ctx, body, Ok(params.as_slice()))?
                } else {
                    let n = f.name();
                    return Err(format!("function {n} doesn't have a body"))?;
                }
            }
            SkOp::Create(sr, fs) => {
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
            SkOp::Match(expr, cases) => {
                fn eval_pattern(
                    value: &Value,
                    pattern: &SkMatchPatternOp,
                    ctx: &impl ResolverContext,
                    heap: &Heap,
                    vars: &mut Vec<Value>,
                ) -> Result<bool> {
                    Ok(match pattern {
                        SkMatchPatternOp::Unit => matches!(value.kind(), SkValueKind::Unit),
                        SkMatchPatternOp::Wildcard => true,
                        SkMatchPatternOp::Union(p) => {
                            for p in p {
                                if eval_pattern(value, p, ctx, heap, vars)? {
                                    return Ok(true);
                                }
                            }
                            false
                        }
                        SkMatchPatternOp::Variable(id) => {
                            if *id >= vars.len() {
                                vars.resize(id + 1, Value::Never)
                            }
                            vars[*id] = value.clone();
                            true
                        }
                        SkMatchPatternOp::BooleanLiteral(v) => match value {
                            Value::Bool(actual) => *v == *actual,
                            _ => false
                        },
                        SkMatchPatternOp::NumberLiteral(n) => match value {
                            Value::I64(actual) => *n == *actual,
                            Value::I16(actual) => *n == *actual as i64,
                            Value::F64(actual) => *n == *actual as i64,
                            _ => false
                        },
                        SkMatchPatternOp::StringLiteral(_) => false,
                        SkMatchPatternOp::IsTypeOrEnum(_) => false,
                        SkMatchPatternOp::TupleStruct { .. } => false,
                        SkMatchPatternOp::FieldStruct { typ, params, exact } => {
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
                                let Some(value) = values.get(*id) else { return Err(format!("invalid ref in struct match pattern: {name}.{field}"))?; };
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
                        return eval(ctx, &x.body, Err(heap.nested(vars.as_slice())));
                    }
                }

                Value::Never
            }
        });
    }
    match body.terminator_op() {
        SkTerminatorOp::Yield(ref v) => return heap.get(v).ok_or("invalid ref in Yield terminator op".into())
    }
    Err("missing Return op")?
}

#[test]
fn it_works() -> Result<()> {
    let body = SkBody::isolated(
        &[SkValueKind::I64],
        |body, params| {
            let _1 = params.into_iter().next().expect("one parameter exactly");
            let _2 = body.label(_1, "x")?;
            let _3 = body.const_i64(2)?;
            let _4 = body.neg(_2)?;
            let _5 = body.add(_4, _3)?;
            let _6 = body.label(_5.clone(), "y")?;
            let _7 = body.const_i64(0)?;
            let _8 = body.gt(_5, _7)?;
            let _9_t = body.body([], |bb, _| {
                bb.yield_expr(_6)
            })?;
            let _9_f = body.body([], |bb, _| {
                let _1_0 = bb.const_i64(0)?;
                bb.yield_expr(_1_0)
            })?;
            let _9 = body.if_expr(_8, _9_t, _9_f)?;
            body.yield_expr(_9)
        },
    )?;

    let value = eval(&(), &body, Ok(&[Value::I64(-4)]));

    assert_eq!(Ok(Value::I64(6)), value);

    let value = eval(&(), &body, Ok(&[Value::I64(4)]));

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
        let Some(body) = f.body() else { return Err("missing body")?; };
        eval(&module, body, Ok(&[Value::I64(41)]))?
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
        let Some(body) = f.body() else { return Err("missing body")?; };
        // assert_eq!("", format!("{:#?}", body));
        assert_eq!([
                       eval(&module, body, Ok(&[Value::I64(10)]))?
                   ], [
                       Value::I64(55)
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

    let Some(point_ref) = module
        .get_ref_by_name("Point")
        .filter(|sr| matches!(sr.typ(), SymbolType::Struct)) else {
        return Err("'Point' is supposed to be declared as a struct")?;
    };

    let body = get_function_body(&module, "new_point")?;
    assert_eq!(
        [
            eval(&module, body, Ok(&[Value::I64(0), Value::I64(1)]))?,
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
            eval(&module, body, Ok(&[Value::Composed(
                Composed::Struct(vec![Value::I64(42), Value::I64(1337)]),
                point_ref,
            )]))?,
        ], [
            Value::I64(1379)
        ]
    );

    let body = get_function_body(&module, "sum")?;
    assert_eq!(
        [
            eval(&module, body, Ok(&[Value::I64(42), Value::I64(1338)]))?,
        ], [
            Value::I64(1379)
        ]
    );

    Ok(())
}

fn get_function_body<'a>(module: &'a Module, name: &str) -> Result<&'a SkBody> {
    let Some(Symbol::Function(f)) = module.get_by_name(name) else {
        return Err(format!("'{name}' is supposed to be a function"))?;
    };

    f.body().ok_or("missing function body".into())
}
