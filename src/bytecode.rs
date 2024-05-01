use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Debug;

use crate::mlir::ops::{BlockBuilder, Body, RuntimeValue, Typed};
use crate::types::SymbolRef;

type Result<T> = core::result::Result<T, Cow<'static, str>>;

pub type ValueRef = RuntimeValue<SkValueKind>;

#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub enum SkValueKind {
    #[default]
    Never,
    Unit,
    Bool,
    I64,
    I16,
    F64,
    Usize,
    Type(SymbolRef),
}

impl Typed for SkOp {
    type ValueType = SkValueKind;

    fn typ(&self) -> Self::ValueType {
        fn bin_op_typ<T: Typed<ValueType=SkValueKind>>(l: &T, r: &T) -> SkValueKind {
            let l = l.typ();
            let r = r.typ();
            if l == r {
                l
            } else {
                SkValueKind::Never
            }
        }

        match self {
            SkOp::Label(v, _) => v.typ(),
            SkOp::ConstUnit => SkValueKind::Unit,
            SkOp::ConstI64(_) => SkValueKind::I64,
            SkOp::Block(b) => b.typ(),
            SkOp::Neg(v) => v.typ(),
            SkOp::Add(l, r) => bin_op_typ(l, r),
            SkOp::Mul(l, r) => bin_op_typ(l, r),
            SkOp::Gt(_, _) => SkValueKind::Bool,
            SkOp::If(_, t, f) => bin_op_typ(t, f),
            SkOp::Call(_, rt, _) => rt.clone(),
            SkOp::Create(t, _) => SkValueKind::Type(t.clone()),
            SkOp::Match(_, p) => p.get(0)
                .map(|c| c.body.typ())
                .unwrap_or(SkValueKind::Never)
        }
    }
}

impl Typed for SkTerminatorOp {
    type ValueType = SkValueKind;

    fn typ(&self) -> Self::ValueType {
        match self {
            SkTerminatorOp::Yield(v) => v.typ()
        }
    }
}

impl SkValueKind {
    pub fn is_assignable_to(&self, other: &SkValueKind) -> bool {
        match other {
            _ if matches!(self, SkValueKind::Never) => true,
            SkValueKind::Never => false,
            SkValueKind::Unit => matches!(*self, SkValueKind::Never | SkValueKind::Unit),
            SkValueKind::Bool => matches!(*self, SkValueKind::Never | SkValueKind::Bool),
            SkValueKind::I64 => matches!(*self, SkValueKind::Never | SkValueKind::I64),
            SkValueKind::I16 => matches!(*self, SkValueKind::Never | SkValueKind::I16),
            SkValueKind::F64 => matches!(*self, SkValueKind::Never | SkValueKind::F64),
            SkValueKind::Usize => matches!(*self, SkValueKind::Never | SkValueKind::Usize),
            SkValueKind::Type(t) => if let SkValueKind::Type(f) = self {
                *f == *t
            } else {
                false
            },
        }
    }

    pub fn is_assignable_from(&self, src: &SkValueKind) -> bool {
        match self {
            _ if matches!(src, SkValueKind::Never) => true,
            SkValueKind::Never => true,
            SkValueKind::Unit => matches!(*self, SkValueKind::Unit),
            SkValueKind::Bool => matches!(*self, SkValueKind::Bool),
            SkValueKind::I64 => matches!(*self, SkValueKind::I64),
            SkValueKind::I16 => matches!(*self, SkValueKind::I16),
            SkValueKind::F64 => matches!(*self, SkValueKind::F64),
            SkValueKind::Usize => matches!(*self, SkValueKind::Usize),
            SkValueKind::Type(t) => if let SkValueKind::Type(f) = src {
                *f == *t
            } else {
                false
            },
        }
    }

    pub fn is_number(&self) -> bool {
        matches!(
            self,
            SkValueKind::Never |
            SkValueKind::I64 |
            SkValueKind::I16 |
            SkValueKind::F64 |
            SkValueKind::Usize
        )
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct SkMatchCaseOp {
    pub pattern: SkMatchPatternOp,
    pub body: SkBody,
    pub guard: Option<SkBody>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum SkMatchPatternOp {
    Unit,
    Wildcard,
    Union(Vec<SkMatchPatternOp>),
    Variable(usize),
    NumberLiteral(i64),
    StringLiteral(String),
    IsTypeOrEnum(SymbolRefOrEnum),
    TupleStruct { typ: SymbolRefOrEnum, params: Vec<SkMatchPatternOp>, exact: bool },
    FieldStruct { typ: SymbolRefOrEnum, params: Vec<(String, SkMatchPatternOp)>, exact: bool },
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
pub enum SkOp {
    Label(ValueRef, String),
    ConstUnit,
    ConstI64(i64),
    Block(SkBody),
    Neg(ValueRef),
    Add(ValueRef, ValueRef),
    Mul(ValueRef, ValueRef),
    Gt(ValueRef, ValueRef),
    If(ValueRef, SkBody, SkBody),
    Call(SymbolRef, SkValueKind, Box<[ValueRef]>),
    Create(SymbolRef, Box<[(String, ValueRef)]>),
    Match(ValueRef, Box<[SkMatchCaseOp]>),
}

#[derive(Debug, Eq, PartialEq)]
pub enum SkTerminatorOp {
    Yield(ValueRef),
}

pub type SkBody = Body<SkValueKind, SkOp, SkTerminatorOp>;
pub type SkBlockBuilder<'a> = BlockBuilder<'a, SkValueKind, SkOp, SkTerminatorOp>;

impl<'a> SkBlockBuilder<'a> {
    pub fn label(&mut self, value: ValueRef, label: impl Into<String>) -> Result<ValueRef> {
        Ok(self.op(SkOp::Label(value, label.into())))
    }

    pub fn const_unit(&mut self) -> Result<ValueRef> {
        Ok(self.op(SkOp::ConstUnit))
    }

    pub fn const_i64(&mut self, value: i64) -> Result<ValueRef> {
        Ok(self.op(SkOp::ConstI64(value)))
    }

    pub fn neg(&mut self, value: ValueRef) -> Result<ValueRef> {
        if !value.typ().is_number() {
            Err(format!("arithmetic operation requires numeric operands; got {:?}", value.typ()))?;
        }

        Ok(self.op(SkOp::Neg(value)))
    }

    pub fn add(&mut self, left: ValueRef, right: ValueRef) -> Result<ValueRef> {
        if !(left.typ().is_number() && right.typ().is_number()) {
            Err(format!("arithmetic operation requires numeric operands; got {:?} and {:?}", left.typ(), right.typ()))?;
        }

        Ok(self.op(SkOp::Add(left, right)))
    }

    pub fn mul(&mut self, left: ValueRef, right: ValueRef) -> Result<ValueRef> {
        if !(left.typ().is_number() && right.typ().is_number()) {
            Err(format!("arithmetic operation requires numeric operands; got {:?} and {:?}", left.typ(), right.typ()))?;
        }

        Ok(self.op(SkOp::Mul(left, right)))
    }

    pub fn yield_expr(&mut self, value: ValueRef) -> Result<SkTerminatorOp> {
        Ok(SkTerminatorOp::Yield(value))
    }

    pub fn gt(&mut self, left: ValueRef, right: ValueRef) -> Result<ValueRef> {
        if !(left.typ().is_number() && right.typ().is_number()) {
            Err(format!("arithmetic operation requires numeric operands; got {:?} and {:?}", left.typ(), right.typ()))?;
        }

        Ok(self.op(SkOp::Gt(left, right)))
    }

    pub fn block_op(&mut self, block: SkBody) -> Result<ValueRef> {
        Ok(self.op(SkOp::Block(block)))
    }

    pub fn if_expr(
        &mut self,
        cond: ValueRef,
        on_true: SkBody,
        on_false: SkBody,
    ) -> Result<ValueRef> {
        if !matches!(cond.typ(), SkValueKind::Bool) {
            Err(format!("conditional expression requires a boolean operand; got {:?}", cond.typ()))?;
        }

        if on_true.typ() != on_false.typ() {
            Err(format!("conditional branches don't return the same type; got {:?} and {:?}", on_true.typ(), on_false.typ()))?;
        }

        Ok(self.op(SkOp::If(cond, on_true, on_false)))
    }

    pub fn call(&mut self, func: SymbolRef, ret_type: SkValueKind, args: impl Into<Box<[ValueRef]>>) -> Result<ValueRef> {
        Ok(self.op(SkOp::Call(func, ret_type, args.into())))
    }

    pub fn create(&mut self, struct_ref: SymbolRef, fields: impl Into<Box<[(String, ValueRef)]>>) -> Result<ValueRef> {
        Ok(self.op(SkOp::Create(struct_ref, fields.into())))
    }

    pub fn match_(&mut self, expr: ValueRef, cases: impl Into<Box<[SkMatchCaseOp]>>) -> Result<ValueRef> {
        let cases = cases.into();
        Ok(self.op(SkOp::Match(expr, cases)))
    }
}
