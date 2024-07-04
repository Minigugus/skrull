use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::vec::Vec;

use crate::bytecode::{SkMatchCaseOp, SkMatchPatternOp, SymbolRefOrEnum};
use crate::mlir::ops::{BlockBuilder, Typed};

use super::{ToTsResolver, TsBody, TsModule, TsModuleBuilder, TsOp, TsScope, TsTerminatorOp, TsValueKind, TsValueRef};

impl TsBody {
    pub fn visit_match_case<'a>(
        builder: &TsModuleBuilder,
        module: &TsModule,
        b: &mut BlockBuilder<TsValueKind, TsOp, TsTerminatorOp>,
        scope: &mut TsScope,
        op: &'a SkMatchCaseOp,
        mut after: impl Iterator<Item=&'a SkMatchCaseOp>,
        expr: TsValueRef,
    ) -> Result<TsOp, Cow<'static, str>> {
        let mut vars = Default::default();
        Self::visit_match_variables(builder, module, b, &op.pattern, &expr.typ(), &mut vars)?;
        let cond = Self::visit_match_pattern(builder, module, b, &op.pattern, &expr, &vars)?;
        if let Some(cond) = cond {
            // TODO guard
            let tb = b.body([], |bb, params| {
                let scope = scope.nested(vars
                    .into_values()
                    .map(|v| bb.op(TsOp::VarGet(v)))
                    .collect::<Vec<_>>());
                Self::visit_body(&op.body, builder, module, bb, scope)
            })?;
            let fb = if let Some(next) = after.next() {
                b.body([], move |bb, args| {
                    // let op = Self::visit_match_case(builder, module, bb, &mut scope.nested(args.as_slice()), next, after, expr)?;
                    let op = Self::visit_match_case(builder, module, bb, scope, next, after, expr)?;
                    Ok(TsTerminatorOp::ReturnValue(bb.op(op)))
                })
            } else {
                b.body([], move |bb, args| {
                    Ok(TsTerminatorOp::Unreachable("match is supposed to cover all possible cases!!!".into()))
                })
            }?;
            return Ok(TsOp::If(cond, tb, fb));
        }
        let scope = scope.nested(vars
            .into_values()
            .map(|v| b.op(TsOp::VarGet(v)))
            .collect::<Vec<_>>());
        Ok(match Self::visit_body(&op.body, builder, module, b, scope)? {
            TsTerminatorOp::Return => TsOp::ConstNull,
            TsTerminatorOp::ReturnValue(v) => TsOp::Nop(v),
            TsTerminatorOp::Unreachable(msg) => TsOp::Error(msg) // FIXME
        })
    }

    fn visit_match_pattern(
        builder: &TsModuleBuilder,
        module: &TsModule,
        b: &mut BlockBuilder<TsValueKind, TsOp, TsTerminatorOp>,
        pattern: &SkMatchPatternOp,
        expr: &TsValueRef,
        vars: &BTreeMap<usize, TsValueRef>,
    ) -> Result<Option<TsValueRef>, Cow<'static, str>> {
        Ok(match &pattern {
            SkMatchPatternOp::Unit => Some(b.op(TsOp::Error("Unit pattern not implemented yet".into()))), // FIXME how should Unit patterns be handled?
            SkMatchPatternOp::Wildcard => None,
            // SkMatchPatternOp::Wildcard => Some(b.op(TsOp::ConstBool(true))),
            SkMatchPatternOp::Union(nested) => {
                let nested = nested
                    .iter()
                    .map(|np| Self::visit_match_pattern(builder, module, b, np, expr, vars))
                    .collect::<Result<Option<Vec<TsValueRef>>, _>>()?;

                if let Some(nested) = nested {
                    nested
                        .into_iter()
                        .reduce(|l, r| b.op(TsOp::Or(l, r)))
                } else {
                    None
                }
            }
            // SkMatchPatternOp::Variable(id) => Some(b.op(TsOp::Error("Variable pattern not implemented yet".into()))), // TODO
            SkMatchPatternOp::Variable(id) => {
                if let Some(var) = vars.get(id) {
                    b.op(TsOp::VarSet(var.clone(), expr.clone()))
                } else {
                    b.op(TsOp::Error(format!("No var id {id}").into()))
                };
                // Some(b.op(TsOp::ConstBool(true)))
                None
            }
            SkMatchPatternOp::BooleanLiteral(v) => {
                let v = b.op(TsOp::ConstBool(*v));
                Some(b.op(TsOp::Eq(expr.clone(), v)))
            }
            SkMatchPatternOp::NumberLiteral(n) => {
                let n = b.op(TsOp::ConstNumber((*n).into()));
                Some(b.op(TsOp::Eq(expr.clone(), n)))
            }
            SkMatchPatternOp::StringLiteral(v) => {
                let v = b.op(TsOp::ConstString(v.clone()));
                Some(b.op(TsOp::Eq(expr.clone(), v)))
            }
            SkMatchPatternOp::IsTypeOrEnum(s) => {
                let s = match s {
                    SymbolRefOrEnum::Type(s) => Ok(builder.convert_refs(s)),
                    SymbolRefOrEnum::Enum(s, v) => {
                        let s = builder.convert_refs(s);
                        module
                            .resolve_enum_variant(&*s.1, &*v)
                            .ok_or_else(|| format!("Enum {} doesn't have a variant named {v}", s.1).into())
                    }
                };
                Some(match s {
                    Ok(s) => b.op(TsOp::InstanceOf(expr.clone(), s)),
                    Err(e) => b.op(TsOp::Error(e)),
                })
            }
            SkMatchPatternOp::TupleStruct { typ, params, .. } => {
                let s = match typ {
                    SymbolRefOrEnum::Type(s) => Ok(builder.convert_refs(s)),
                    SymbolRefOrEnum::Enum(s, v) => {
                        let s = builder.convert_refs(s);
                        module
                            .resolve_enum_variant(&*s.1, &*v)
                            .ok_or_else(|| format!("Enum {} doesn't have a variant named {v}", s.1).into())
                    }
                };
                Some(match s {
                    Ok(s) => {
                        let e = b.op(TsOp::InstanceOf(expr.clone(), s.clone()));

                        let tb = b.body([], |bb, _| {
                            let expr = bb.op(TsOp::Cast(expr.clone(), s.clone()));

                            let nested = params
                                .iter()
                                .enumerate()
                                .map(|(i, x)| {
                                    if let Some(field) = module.resolve_record_field(&s, format!("_{i}").as_str()) {
                                        let typ = TsValueKind::TsType(field.typ.clone());
                                        let op = bb.op(TsOp::GetTupleField(expr.clone(), typ, i));
                                        Self::visit_match_pattern(builder, module, bb, x, &op, vars)
                                    } else {
                                        Err(Cow::from(format!("Tuple field {i} not found on {}", s.1)))
                                    }
                                })
                                .collect::<Result<Option<Vec<TsValueRef>>, _>>()?;

                            let nested = if let Some(nested) = nested {
                                nested
                                    .into_iter()
                                    .reduce(|l, r| bb.op(TsOp::Or(l, r)))
                            } else {
                                None
                            };

                            let v = nested.unwrap_or_else(|| bb.op(TsOp::ConstBool(true)));
                            Ok(TsTerminatorOp::ReturnValue(v))
                        })?;

                        let fb = b.body([], |bb, _| {
                            let v = bb.op(TsOp::ConstBool(false));
                            Ok(TsTerminatorOp::ReturnValue(v))
                        })?;

                        b.op(TsOp::If(e, tb, fb))
                    }
                    Err(e) => b.op(TsOp::Error(e)),
                })
            }
            SkMatchPatternOp::FieldStruct { typ, params, .. } => {
                let s = match typ {
                    SymbolRefOrEnum::Type(s) => Ok(builder.convert_refs(s)),
                    SymbolRefOrEnum::Enum(s, v) => {
                        let s = builder.convert_refs(s);
                        module
                            .resolve_enum_variant(&*s.1, &*v)
                            .ok_or_else(|| format!("Enum {} doesn't have a variant named {v}", s.1).into())
                    }
                };
                Some(match s {
                    Ok(s) => {
                        let e = b.op(TsOp::InstanceOf(expr.clone(), s.clone()));

                        let tb = b.body([], |bb, _| {
                            let expr = bb.op(TsOp::Cast(expr.clone(), s.clone()));

                            let nested = params
                                .iter()
                                .map(|(f, x)| {
                                    if let Some(field) = module.resolve_record_field(&s, f.as_str()) {
                                        let typ = TsValueKind::TsType(field.typ.clone());
                                        let op = bb.op(TsOp::GetRecordField(expr.clone(), typ, f.clone()));
                                        Self::visit_match_pattern(builder, module, bb, x, &op, vars)
                                    } else {
                                        Err(Cow::from(format!("Record field {f} not found on {}", s.1)))
                                    }
                                })
                                .collect::<Result<Option<Vec<TsValueRef>>, _>>()?;

                            let nested = if let Some(nested) = nested {
                                nested
                                    .into_iter()
                                    .reduce(|l, r| bb.op(TsOp::Or(l, r)))
                            } else {
                                None
                            };

                            let v = nested.unwrap_or_else(|| bb.op(TsOp::ConstBool(true)));
                            Ok(TsTerminatorOp::ReturnValue(v))
                        })?;

                        let fb = b.body([], |bb, _| {
                            let v = bb.op(TsOp::ConstBool(false));
                            Ok(TsTerminatorOp::ReturnValue(v))
                        })?;

                        b.op(TsOp::If(e, tb, fb))
                    }
                    Err(e) => b.op(TsOp::Error(e)),
                })
            }
        })
    }

    fn visit_match_variables(
        builder: &TsModuleBuilder,
        module: &TsModule,
        b: &mut BlockBuilder<TsValueKind, TsOp, TsTerminatorOp>,
        pattern: &SkMatchPatternOp,
        kind: &TsValueKind,
        vars: &mut BTreeMap<usize, TsValueRef>,
    ) -> Result<(), Cow<'static, str>> {
        match pattern {
            SkMatchPatternOp::Union(nested) => {
                for n in nested {
                    Self::visit_match_variables(
                        builder,
                        module,
                        b,
                        n,
                        kind,
                        vars,
                    )?;
                }
            }
            SkMatchPatternOp::Variable(id) => {
                vars.entry(*id)
                    .or_insert_with(|| b.op(TsOp::VarDef(kind.clone())));
            }
            SkMatchPatternOp::TupleStruct { typ, params, .. } => {
                let s = match typ {
                    SymbolRefOrEnum::Type(s) => builder.convert_refs(s),
                    SymbolRefOrEnum::Enum(s, v) => {
                        let s = builder.convert_refs(s);
                        module
                            .resolve_enum_variant(&*s.1, &*v)
                            .ok_or_else(|| format!("Enum {} doesn't have a variant named {v}", s.1))?
                    }
                };

                let nested = params
                    .iter()
                    .enumerate()
                    .map(|(i, np)| {
                        if let Some(field) = module.resolve_record_field(&s, format!("_{i}").as_str()) {
                            Ok((&field.typ, np))
                        } else {
                            Err(Cow::from(format!("Tuple field {i} not found on {}", s.1)))
                        }
                    })
                    .collect::<Result<Vec<(_, _)>, _>>()?;

                for (kind, np) in nested {
                    Self::visit_match_variables(
                        builder,
                        module,
                        b,
                        np,
                        &TsValueKind::TsType(kind.clone()),
                        vars,
                    )?;
                }
            }
            SkMatchPatternOp::FieldStruct { typ, params, .. } => {
                let s = match typ {
                    SymbolRefOrEnum::Type(s) => builder.convert_refs(s),
                    SymbolRefOrEnum::Enum(s, v) => {
                        let s = builder.convert_refs(s);
                        module
                            .resolve_enum_variant(&*s.1, &*v)
                            .ok_or_else(|| format!("Enum {} doesn't have a variant named {v}", s.1))?
                    }
                };

                let nested = params
                    .iter()
                    // .filter(|(_, op)| !matches!(op, SkMatchPatternOp::Wildcard))
                    .map(|(f, np)| {
                        if let Some(field) = module.resolve_record_field(&s, f.as_str()) {
                            Ok((&field.typ, np))
                        } else {
                            Err(Cow::from(format!("Record field {f} not found on {}", s.1)))
                        }
                    })
                    .collect::<Result<Vec<(_, _)>, _>>()?;

                for (kind, np) in nested {
                    Self::visit_match_variables(
                        builder,
                        module,
                        b,
                        np,
                        &TsValueKind::TsType(kind.clone()),
                        vars,
                    )?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

mod tests {
    use alloc::borrow::Cow;
    use alloc::format;
    use alloc::string::ToString;

    use crate::lexer::Token;
    use crate::types::Module;

    use super::TsModule;

    #[test]
    fn it_transform_match_complex() -> Result<(), Cow<'static, str>> {
        // tokenize
        let tokens = /*language=rust*/Token::parse_ascii(r#"pub enum Priced {
  Limit { price: f64 },
  Market,
  StopLimit { stop_price: f64 },
}

pub fn get_price(priced: Priced) -> f64 {
  match priced {
    Priced::Limit { price } | Priced::StopLimit { stop_price: price } => price,
    Priced::Market => 0 // TODO support SkOp::Create for enum variants
  }
}
"#)?;

        // parse + semantic analysis
        let module = Module::parse_tokens("skrull_test_transform_match", tokens)?;

        // transform
        let ts = TsModule::try_from(&module)?;

        assert_eq!(
            format!("{ts}"),
            /*language=ts*/(r#"class Priced_Limit {
  public constructor(
    public readonly price: number
  ) { }
}

class Priced_Market {
  public constructor() { }
}

class Priced_StopLimit {
  public constructor(
    public readonly stop_price: number
  ) { }
}

export type Priced =
  | Priced_Limit
  | Priced_Market
  | Priced_StopLimit;

export function get_price(priced: Priced): number {
  let _var_0_1: number = null!;
  let _if_0_3: boolean;
  if (priced instanceof Priced_Limit) {
    const _1_0: Priced_Limit = (priced) as (Priced_Limit);
    const _1_1: number = _1_0.price;
    _var_0_1 = _1_1;
    _if_0_3 = true;
  } else {
    _if_0_3 = false;
  }
  let _if_0_5: boolean;
  if (priced instanceof Priced_StopLimit) {
    const _1_0: Priced_StopLimit = (priced) as (Priced_StopLimit);
    const _1_1: number = _1_0.stop_price;
    _var_0_1 = _1_1;
    _if_0_5 = true;
  } else {
    _if_0_5 = false;
  }
  let _if_0_7: number;
  if (_if_0_3 || _if_0_5) {
    _if_0_7 = _var_0_1;
  } else {
    let _if_1_1: number;
    if (priced instanceof Priced_Market) {
      _if_1_1 = 0;
    } else {
      throw new Error("UNREACHABLE: match is supposed to cover all possible cases!!!");
    }
    _if_0_7 = _if_1_1;
  }
  return _if_0_7;
}

"#.to_string())
        );

        Ok(())
    }

    #[test]
    fn it_transform_match_simple() -> Result<(), Cow<'static, str>> {
        // tokenize
        let tokens = /*language=rust*/Token::parse_ascii(r#"
pub fn just_get(n: i64) -> i64 {
  match n {
    _ => n
  }
}

pub fn get(n: i64) -> i64 {
  match n {
    n => n
  }
}

pub fn get_or_default_if_zero(n: i64, def: i64) -> i64 {
  match n {
    0 => def,
    n => n
  }
}

pub fn plus_one_then_default_if_zero(n: i64, def: i64) -> i64 {
  match n + 1 {
    0 => def,
    n => n
  }
}

pub fn plus_one_then_default_if_zero_twice(n: i64, def: i64) -> i64 {
  match n + 1 {
    0 => def,
    n => match n {
      0 => def,
      n => n
    }
  }
}

"#)?;

        // parse + semantic analysis
        let module = Module::parse_tokens("skrull_test_transform_simple_match", tokens)?;

        // transform
        let ts = TsModule::try_from(&module)?;

        assert_eq!(
            format!("{ts}"),
            /*language=ts*/(r#"export function just_get(n: number): number {
  return n;
}

export function get(n: number): number {
  let _var_0_1: number = null!;
  _var_0_1 = n;
  return _var_0_1;
}

export function get_or_default_if_zero(n: number, def: number): number {
  let _if_0_4: number;
  if (n == 0) {
    _if_0_4 = def;
  } else {
    let _var_1_0: number = null!;
    _var_1_0 = n;
    _if_0_4 = _var_1_0;
  }
  return _if_0_4;
}

export function plus_one_then_default_if_zero(n: number, def: number): number {
  const _0_3: number = n + 1;
  let _if_0_6: number;
  if (_0_3 == 0) {
    _if_0_6 = def;
  } else {
    let _var_1_0: number = null!;
    _var_1_0 = _0_3;
    _if_0_6 = _var_1_0;
  }
  return _if_0_6;
}

export function plus_one_then_default_if_zero_twice(n: number, def: number): number {
  const _0_3: number = n + 1;
  let _if_0_6: number;
  if (_0_3 == 0) {
    _if_0_6 = def;
  } else {
    let _var_1_0: number = null!;
    _var_1_0 = _0_3;
    let _if_1_6: number;
    if (_var_1_0 == 0) {
      _if_1_6 = def;
    } else {
      let _var_2_0: number = null!;
      _var_2_0 = _var_1_0;
      _if_1_6 = _var_2_0;
    }
    _if_0_6 = _if_1_6;
  }
  return _if_0_6;
}

"#.to_string())
        );

        Ok(())
    }
}
