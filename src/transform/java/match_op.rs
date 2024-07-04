use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::vec::Vec;

use crate::bytecode::{SkMatchCaseOp, SkMatchPatternOp, SymbolRefOrEnum};
use crate::mlir::ops::{BlockBuilder, Typed};

use super::{JavaBody, JavaModule, JavaModuleBuilder, JavaOpN, JavaScope, JavaTerminatorOpN, JavaValueKind, JavaValueRef, ToJavaResolver};

impl JavaBody {
    pub fn visit_match_case<'a>(
        builder: &JavaModuleBuilder,
        module: &JavaModule,
        b: &mut BlockBuilder<JavaValueKind, JavaOpN, JavaTerminatorOpN>,
        scope: &mut JavaScope,
        op: &'a SkMatchCaseOp,
        mut after: impl Iterator<Item=&'a SkMatchCaseOp>,
        expr: JavaValueRef,
    ) -> Result<JavaOpN, Cow<'static, str>> {
        let mut vars = Default::default();
        Self::visit_match_variables(builder, module, b, &op.pattern, &expr.typ(), &mut vars)?;
        let cond = Self::visit_match_pattern(builder, module, b, &op.pattern, &expr, &vars)?;
        if let Some(cond) = cond {
            // TODO guard
            let tb = b.body([], |bb, params| {
                let scope = scope.nested(vars
                    .into_values()
                    .map(|v| bb.op(JavaOpN::VarGet(v)))
                    .collect::<Vec<_>>());
                Self::visit_body(&op.body, builder, module, bb, scope)
            })?;
            let fb = if let Some(next) = after.next() {
                b.body([], move |bb, args| {
                    // let op = Self::visit_match_case(builder, module, bb, &mut scope.nested(args.as_slice()), next, after, expr)?;
                    let op = Self::visit_match_case(builder, module, bb, scope, next, after, expr)?;
                    Ok(JavaTerminatorOpN::ReturnValue(bb.op(op)))
                })
            } else {
                b.body([], move |bb, args| {
                    Ok(JavaTerminatorOpN::Unreachable("match is supposed to cover all possible cases!!!".into()))
                })
            }?;
            return Ok(JavaOpN::If(cond, tb, fb));
        }
        let scope = scope.nested(vars
            .into_values()
            .map(|v| b.op(JavaOpN::VarGet(v)))
            .collect::<Vec<_>>());
        Ok(match Self::visit_body(&op.body, builder, module, b, scope)? {
            JavaTerminatorOpN::Return => JavaOpN::ConstNull,
            JavaTerminatorOpN::ReturnValue(v) => JavaOpN::Nop(v),
            JavaTerminatorOpN::Unreachable(msg) => JavaOpN::Error(msg) // FIXME
        })
    }

    fn visit_match_pattern(
        builder: &JavaModuleBuilder,
        module: &JavaModule,
        b: &mut BlockBuilder<JavaValueKind, JavaOpN, JavaTerminatorOpN>,
        pattern: &SkMatchPatternOp,
        expr: &JavaValueRef,
        vars: &BTreeMap<usize, JavaValueRef>,
    ) -> Result<Option<JavaValueRef>, Cow<'static, str>> {
        Ok(match &pattern {
            SkMatchPatternOp::Unit => Some(b.op(JavaOpN::Error("Unit pattern not implemented yet".into()))), // FIXME how should Unit patterns be handled?
            SkMatchPatternOp::Wildcard => None,
            // SkMatchPatternOp::Wildcard => Some(b.op(JavaOpN::ConstBool(true))),
            SkMatchPatternOp::Union(nested) => {
                let nested = nested
                    .iter()
                    .map(|np| Self::visit_match_pattern(builder, module, b, np, expr, vars))
                    .collect::<Result<Option<Vec<JavaValueRef>>, _>>()?;

                if let Some(nested) = nested {
                    nested
                        .into_iter()
                        .reduce(|l, r| b.op(JavaOpN::Or(l, r)))
                } else {
                    None
                }
            }
            // SkMatchPatternOp::Variable(id) => Some(b.op(JavaOpN::Error("Variable pattern not implemented yet".into()))), // TODO
            SkMatchPatternOp::Variable(id) => {
                if let Some(var) = vars.get(id) {
                    b.op(JavaOpN::VarSet(var.clone(), expr.clone()))
                } else {
                    b.op(JavaOpN::Error(format!("No var id {id}").into()))
                };
                // Some(b.op(JavaOpN::ConstBool(true)))
                None
            }
            SkMatchPatternOp::BooleanLiteral(v) => {
                let v = b.op(JavaOpN::ConstBool(*v));
                Some(b.op(JavaOpN::Eq(expr.clone(), v)))
            }
            SkMatchPatternOp::NumberLiteral(n) => {
                let n = b.op(JavaOpN::ConstLong(*n));
                Some(b.op(JavaOpN::Eq(expr.clone(), n)))
            }
            SkMatchPatternOp::StringLiteral(v) => {
                let v = b.op(JavaOpN::ConstString(v.clone()));
                Some(b.op(JavaOpN::Eq(expr.clone(), v)))
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
                    Ok(s) => b.op(JavaOpN::InstanceOf(expr.clone(), s)),
                    Err(e) => b.op(JavaOpN::Error(e)),
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
                        let e = b.op(JavaOpN::InstanceOf(expr.clone(), s.clone()));

                        let tb = b.body([], |bb, _| {
                            let expr = bb.op(JavaOpN::Cast(expr.clone(), s.clone()));

                            let nested = params
                                .iter()
                                .enumerate()
                                .map(|(i, x)| {
                                    if let Some(field) = module.resolve_record_field(&s, format!("_{i}").as_str()) {
                                        let typ = JavaValueKind::JavaType(field.typ.clone());
                                        let op = bb.op(JavaOpN::GetTupleField(expr.clone(), typ, i));
                                        Self::visit_match_pattern(builder, module, bb, x, &op, vars)
                                    } else {
                                        Err(Cow::from(format!("Tuple field {i} not found on {}", s.1)))
                                    }
                                })
                                .collect::<Result<Option<Vec<JavaValueRef>>, _>>()?;

                            let nested = if let Some(nested) = nested {
                                nested
                                    .into_iter()
                                    .reduce(|l, r| bb.op(JavaOpN::Or(l, r)))
                            } else {
                                None
                            };

                            let v = nested.unwrap_or_else(|| bb.op(JavaOpN::ConstBool(true)));
                            Ok(JavaTerminatorOpN::ReturnValue(v))
                        })?;

                        let fb = b.body([], |bb, _| {
                            let v = bb.op(JavaOpN::ConstBool(false));
                            Ok(JavaTerminatorOpN::ReturnValue(v))
                        })?;

                        b.op(JavaOpN::If(e, tb, fb))
                    }
                    Err(e) => b.op(JavaOpN::Error(e)),
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
                        let e = b.op(JavaOpN::InstanceOf(expr.clone(), s.clone()));

                        let tb = b.body([], |bb, _| {
                            let expr = bb.op(JavaOpN::Cast(expr.clone(), s.clone()));

                            let nested = params
                                .iter()
                                .map(|(f, x)| {
                                    if let Some(field) = module.resolve_record_field(&s, f.as_str()) {
                                        let typ = JavaValueKind::JavaType(field.typ.clone());
                                        let op = bb.op(JavaOpN::GetRecordField(expr.clone(), typ, f.clone()));
                                        Self::visit_match_pattern(builder, module, bb, x, &op, vars)
                                    } else {
                                        Err(Cow::from(format!("Record field {f} not found on {}", s.1)))
                                    }
                                })
                                .collect::<Result<Option<Vec<JavaValueRef>>, _>>()?;

                            let nested = if let Some(nested) = nested {
                                nested
                                    .into_iter()
                                    .reduce(|l, r| bb.op(JavaOpN::Or(l, r)))
                            } else {
                                None
                            };

                            let v = nested.unwrap_or_else(|| bb.op(JavaOpN::ConstBool(true)));
                            Ok(JavaTerminatorOpN::ReturnValue(v))
                        })?;

                        let fb = b.body([], |bb, _| {
                            let v = bb.op(JavaOpN::ConstBool(false));
                            Ok(JavaTerminatorOpN::ReturnValue(v))
                        })?;

                        b.op(JavaOpN::If(e, tb, fb))
                    }
                    Err(e) => b.op(JavaOpN::Error(e)),
                })
            }
        })
    }

    fn visit_match_variables(
        builder: &JavaModuleBuilder,
        module: &JavaModule,
        b: &mut BlockBuilder<JavaValueKind, JavaOpN, JavaTerminatorOpN>,
        pattern: &SkMatchPatternOp,
        kind: &JavaValueKind,
        vars: &mut BTreeMap<usize, JavaValueRef>,
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
                    .or_insert_with(|| b.op(JavaOpN::VarDef(kind.clone())));
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
                        &JavaValueKind::JavaType(kind.clone()),
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
                        &JavaValueKind::JavaType(kind.clone()),
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
    use alloc::string::ToString;

    use crate::lexer::Token;
    use crate::types::Module;

    use super::JavaModule;

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
        let java = JavaModule::try_from(&module)?;

        assert_eq!(
            java.resolve("skrull_test_transform_match.Priced").map(ToString::to_string),
            /*language=java*/Some(r#"package skrull_test_transform_match;

public sealed interface Priced {

record Limit(
  double price
) implements skrull_test_transform_match.Priced { }
record Market() implements skrull_test_transform_match.Priced { }
record StopLimit(
  double stop_price
) implements skrull_test_transform_match.Priced { }

  public static double get_price(skrull_test_transform_match.Priced priced) {
    double _var_0_1 = 0d;
    final boolean _if_0_3;
    if (priced instanceof skrull_test_transform_match.Priced.Limit) {
      final skrull_test_transform_match.Priced.Limit _1_0 = ((skrull_test_transform_match.Priced.Limit) priced);
      final double _1_1 = _1_0.price();
      _var_0_1 = _1_1;
      _if_0_3 = true;
    } else {
      _if_0_3 = false;
    }
    final boolean _if_0_5;
    if (priced instanceof skrull_test_transform_match.Priced.StopLimit) {
      final skrull_test_transform_match.Priced.StopLimit _1_0 = ((skrull_test_transform_match.Priced.StopLimit) priced);
      final double _1_1 = _1_0.stop_price();
      _var_0_1 = _1_1;
      _if_0_5 = true;
    } else {
      _if_0_5 = false;
    }
    final double _if_0_7;
    if (_if_0_3 || _if_0_5) {
      _if_0_7 = _var_0_1;
    } else {
      final short _if_1_1;
      if (priced instanceof skrull_test_transform_match.Priced.Market) {
        _if_1_1 = 0;
      } else {
        throw new AssertionError("UNREACHABLE: match is supposed to cover all possible cases!!!");
      }
      _if_0_7 = _if_1_1;
    }
    return _if_0_7;
  }


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
        let java = JavaModule::try_from(&module)?;

        assert_eq!(
            java.resolve("skrull_test_transform_simple_match.Utils").map(ToString::to_string),
            /*language=java*/Some(r#"package skrull_test_transform_simple_match;

public final class Utils {

  public static long just_get(long n) {
    return n;
  }
  public static long get(long n) {
    long _var_0_1 = 0l;
    _var_0_1 = n;
    return _var_0_1;
  }
  public static long get_or_default_if_zero(long n, long def) {
    final long _if_0_4;
    if (n == 0) {
      _if_0_4 = def;
    } else {
      long _var_1_0 = 0l;
      _var_1_0 = n;
      _if_0_4 = _var_1_0;
    }
    return _if_0_4;
  }
  public static long plus_one_then_default_if_zero(long n, long def) {
    final long _0_3 = n + 1;
    final long _if_0_6;
    if (_0_3 == 0) {
      _if_0_6 = def;
    } else {
      long _var_1_0 = 0l;
      _var_1_0 = _0_3;
      _if_0_6 = _var_1_0;
    }
    return _if_0_6;
  }
  public static long plus_one_then_default_if_zero_twice(long n, long def) {
    final long _0_3 = n + 1;
    final long _if_0_6;
    if (_0_3 == 0) {
      _if_0_6 = def;
    } else {
      long _var_1_0 = 0l;
      _var_1_0 = _0_3;
      final long _if_1_6;
      if (_var_1_0 == 0) {
        _if_1_6 = def;
      } else {
        long _var_2_0 = 0l;
        _var_2_0 = _var_1_0;
        _if_1_6 = _var_2_0;
      }
      _if_0_6 = _if_1_6;
    }
    return _if_0_6;
  }

}"#.to_string())
        );

        Ok(())
    }
}
