use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Display, Formatter, Write};
use core::hash::{Hash, Hasher};

pub type Result<T> = core::result::Result<T, Cow<'static, str>>;

pub type ParameterId = usize;
pub type BodyDepth = usize;
pub type BlockId = usize;
pub type OpId = usize;

pub trait Typed {
    type ValueType;

    fn typ(&self) -> Self::ValueType;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RuntimeValue<VT> {
    Parameter(BodyDepth, BlockId, ParameterId, VT),
    Local(BodyDepth, BlockId, OpId, VT),
}

#[derive(Debug, Eq, PartialEq)]
pub struct Block<VT, OP: Typed<ValueType=VT>, TOP: Typed<ValueType=VT>> {
    parameters: Vec<VT>,
    terminator_op: TOP,
    ops: Vec<OP>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Body<VT, OP: Typed<ValueType=VT>, TOP: Typed<ValueType=VT>> {
    depth: BodyDepth,
    entry: Block<VT, OP, TOP>,
    others: Vec<Block<VT, OP, TOP>>,
}

pub struct BlockBuilder<'a, VT, OP: Typed<ValueType=VT>, TOP: Typed<ValueType=VT>> {
    block_id: BlockId,
    body_depth: BodyDepth,
    blocks: &'a mut Vec<Block<VT, OP, TOP>>,
    ops: Vec<OP>,
}

pub enum RefId {
    Param(ParameterId),
    Op(OpId),
}

impl<T> RuntimeValue<T> {
    pub fn is_param(&self) -> bool {
        matches!(self, RuntimeValue::Parameter(_, _, _, _))
    }

    pub fn id(&self) -> RefId {
        match *self {
            RuntimeValue::Parameter(_, _, pi, _) => RefId::Param(pi),
            RuntimeValue::Local(_, _, oi, _) => RefId::Op(oi)
        }
    }
}

// impl<T> Hash for RuntimeValue<T> {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         let (bd, bi, id) = match self {
//             RuntimeValue::Parameter(bd, bi, pi, _) => {
//                 state.write_i8(0);
//                 (bd, bi, pi as usize)
//             }
//             RuntimeValue::Local(bd, bi, oi, _) => {
//                 state.write_i8(1);
//                 (bd, bi, oi)
//             }
//         };
//         bd.hash(state);
//         bi.hash(state);
//         id.hash(state);
//     }
// }

impl<T: Clone> Typed for RuntimeValue<T> {
    type ValueType = T;

    fn typ(&self) -> Self::ValueType {
        match self {
            RuntimeValue::Parameter(_, _, _, t) | RuntimeValue::Local(_, _, _, t) => t
        }.clone()
    }
}

impl<VT, OP: Typed<ValueType=VT>, TOP: Typed<ValueType=VT>> Typed for Block<VT, OP, TOP> {
    type ValueType = VT;

    fn typ(&self) -> Self::ValueType {
        self.terminator_op.typ()
    }
}

impl<VT, OP: Typed<ValueType=VT>, TOP: Typed<ValueType=VT>> Typed for Body<VT, OP, TOP> {
    type ValueType = VT;

    fn typ(&self) -> Self::ValueType {
        self.entry.typ()
    }
}

impl<'a, VT, OP: Typed<ValueType=VT>, TOP: Typed<ValueType=VT>> BlockBuilder<'a, VT, OP, TOP> {
    pub fn op(
        &mut self,
        op: OP,
    ) -> RuntimeValue<VT> {
        let vt = op.typ();
        let id = self.ops.len();
        self.ops.push(op);
        RuntimeValue::Local(self.body_depth, self.block_id, id, vt)
    }

    pub fn body(
        &mut self,
        parameters: impl Into<Vec<VT>>,
        builder: impl FnOnce(&mut BlockBuilder<VT, OP, TOP>, Vec<RuntimeValue<VT>>) -> Result<TOP>,
    ) -> Result<Body<VT, OP, TOP>> where VT: Clone {
        Body::new(self.body_depth + 1, parameters, builder)
    }

    pub fn block(
        &mut self,
        parameters: impl Into<Vec<VT>>,
        builder: impl FnOnce(&mut BlockBuilder<VT, OP, TOP>, Vec<RuntimeValue<VT>>) -> Result<TOP>,
    ) -> Result<BlockId> where VT: Clone {
        let bi = self.block_id;
        let parameters = parameters.into();
        let parameter_values = parameters
            .iter()
            .enumerate()
            .map(|(id, t)| RuntimeValue::Parameter(self.body_depth, bi, id, t.clone()))
            .collect();
        let mut bb = BlockBuilder {
            block_id: 0,
            body_depth: self.body_depth,
            blocks: self.blocks,
            ops: vec![],
        };
        let terminator_op = (builder)(&mut bb, parameter_values)?;
        let ops = bb.ops;
        let id = self.blocks.len();
        self.blocks.push(Block {
            parameters,
            terminator_op,
            ops,
        });
        Ok(id)
    }
}

impl<VT, OP: Typed<ValueType=VT>, TOP: Typed<ValueType=VT>> Body<VT, OP, TOP> {
    pub fn op(&self, typ: &RuntimeValue<VT>) -> Option<&OP> {
        match *typ {
            RuntimeValue::Local(bd, _, oi, _) if bd == self.depth => self.entry.ops.get(oi),
            _ => None,
        }
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn entries(&self) -> impl Iterator<Item=(RuntimeValue<VT>, &OP)> {
        let depth = self.depth;
        self.entry.ops
            .iter()
            .enumerate()
            .map(move |(oi, op)| (RuntimeValue::Local(depth, 0, oi, op.typ()), op))
    }

    pub fn ops(&self) -> &[OP] {
        self.entry.ops.as_slice()
    }

    pub fn terminator_op(&self) -> &TOP {
        &self.entry.terminator_op
    }

    pub fn params(&self) -> &[VT] {
        self.entry.parameters.as_slice()
    }

    fn new(
        depth: BodyDepth,
        parameters: impl Into<Vec<VT>>,
        builder: impl FnOnce(&mut BlockBuilder<VT, OP, TOP>, Vec<RuntimeValue<VT>>) -> Result<TOP>,
    ) -> Result<Body<VT, OP, TOP>> where VT: Clone {
        let parameters = parameters.into();
        let parameter_values = parameters
            .iter()
            .enumerate()
            .map(|(id, t)| RuntimeValue::Parameter(depth, 0, id, t.clone()))
            .collect();
        let mut blocks = vec![];
        let mut bb = BlockBuilder {
            block_id: 0,
            body_depth: depth,
            blocks: &mut blocks,
            ops: vec![],
        };
        let terminator_op = (builder)(&mut bb, parameter_values)?;
        Ok(Body {
            depth,
            entry: Block {
                parameters,
                terminator_op,
                ops: bb.ops,
            },
            others: blocks,
        })
    }

    pub fn isolated(
        parameters: impl Into<Vec<VT>>,
        builder: impl FnOnce(&mut BlockBuilder<VT, OP, TOP>, Vec<RuntimeValue<VT>>) -> Result<TOP>,
    ) -> Result<Body<VT, OP, TOP>> where VT: Clone {
        Self::new(0, parameters, builder)
    }
}

pub struct GroupPrinter<T> {
    depth: usize,
    indent: bool,
    open: &'static str,
    close: &'static str,
    separator: &'static str,
    wrapped: T,
}

pub struct Printer<T> {
    depth: usize,
    wrapped: T,
}

impl<'a, T> Display for RuntimeValue<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            RuntimeValue::Parameter(bd, _, v, _) => write!(f, "%{bd}_{v}"),
            RuntimeValue::Local(bd, _, o, _) => write!(f, "${bd}_{o}"),
        }
    }
}

impl<'a, T> Display for GroupPrinter<&'a Vec<T>> where Printer<(usize, &'a T)>: Display {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut iter = self.wrapped.iter().enumerate();
        f.write_str(self.open)?;
        if let Some(first) = iter.next() {
            let cd = self.depth + 1;
            if self.indent {
                indent(f, cd)?;
            }
            Display::fmt(&Printer { depth: cd, wrapped: first }, f)?;
            while let Some(next) = iter.next() {
                f.write_str(self.separator)?;
                if self.indent {
                    indent(f, cd)?;
                }
                Display::fmt(&Printer { depth: cd, wrapped: next }, f)?;
            }
            if self.indent && !self.close.is_empty() {
                indent(f, self.depth)?;
            }
        }
        f.write_str(self.close)
    }
}

impl<'a, VT: Display, OP: Typed<ValueType=VT>, TOP: Typed<ValueType=VT>> Display for Printer<&'a Block<VT, OP, TOP>>
    where Printer<(usize, &'a VT)>: Display,
          Printer<(usize, &'a OP)>: Display,
          Printer<&'a TOP>: Display {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let params = GroupPrinter { depth: self.depth, indent: false, open: "|", close: "|", separator: ", ", wrapped: &self.wrapped.parameters };
        let typ = &self.wrapped.typ();
        let terminator_op = Printer { depth: self.depth + 1, wrapped: &self.wrapped.terminator_op };
        if self.wrapped.ops.is_empty() {
            write!(f, "{} -> {} {{{}}}",
                   params,
                   typ,
                   terminator_op)
        } else {
            write!(f, "{} -> {} {};{}{}{}}}",
                   params,
                   typ,
                   GroupPrinter { depth: self.depth, indent: true, open: "{", close: "", separator: ";", wrapped: &self.wrapped.ops },
                   Indent(self.depth + 1),
                   terminator_op,
                   Indent(self.depth))
        }
    }
}

impl<'a, VT: Display, OP: Typed<ValueType=VT>, TOP: Typed<ValueType=VT>> Display for Printer<(usize, &'a Block<VT, OP, TOP>)>
    where Printer<(usize, &'a VT)>: Display,
          Printer<(usize, &'a OP)>: Display,
          Printer<&'a TOP>: Display {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&Printer { depth: self.depth, wrapped: self.wrapped.1 }, f)
        // write!(f, "_{}", Printer { depth: self.depth, wrapped: &self.wrapped.parameters })
        // write!(f, "{} ", Printer { depth: self.depth, wrapped: &self.wrapped.ops })
    }
}

impl<'a, VT: Display, OP: Typed<ValueType=VT>, TOP: Typed<ValueType=VT>> Display for Printer<&'a Body<VT, OP, TOP>>
    where Printer<(usize, &'a VT)>: Display,
          Printer<(usize, &'a OP)>: Display,
          Printer<&'a TOP>: Display {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        // write!(f, "_{}", Printer { depth: self.depth, wrapped: &self.wrapped.parameters })
        if self.wrapped.others.is_empty() {
            Display::fmt(&Printer { depth: self.depth, wrapped: &self.wrapped.entry }, f)
        } else {
            write!(f, "{}{}{}",
                   Printer { depth: self.depth, wrapped: &self.wrapped.entry },
                   Indent(self.depth),
                   GroupPrinter { depth: self.depth, indent: true, open: "", close: "", separator: "", wrapped: &self.wrapped.others })
        }
    }
}

struct Indent(usize);

impl Display for Indent {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        indent(f, self.0)
    }
}

fn indent(f: &mut Formatter<'_>, len: usize) -> core::fmt::Result {
    f.write_char('\n')?;
    for _ in 0..len {
        f.write_char(' ')?;
        f.write_char(' ')?;
    }
    Ok(())
}

mod tests {
    use alloc::borrow::Cow;
    use alloc::format;
    use alloc::vec::Vec;
    use core::fmt::{Display, Formatter};

    use super::{BlockBuilder, Body, Printer, Result, RuntimeValue, Typed};

    #[derive(Copy, Clone, Eq, PartialEq)]
    pub enum NativeType {
        Bool,
        Usize,
        Str,
    }

    #[derive(Copy, Clone, Eq, PartialEq)]
    pub enum ValueType {
        Unit,
        Native(NativeType),
    }

    pub enum ConstValue {
        Bool(bool),
        Usize(usize),
        Str(Cow<'static, str>),
    }

    pub enum Op {
        Const(ConstValue),
        VarDeclare(RuntimeValue<ValueType>, Cow<'static, str>),
        VarGet(RuntimeValue<ValueType>),
        Neg(RuntimeValue<ValueType>),
        Add(RuntimeValue<ValueType>, RuntimeValue<ValueType>),
        If(Body<ValueType, Op, TerminatorOp>, Body<ValueType, Op, TerminatorOp>, Option<Body<ValueType, Op, TerminatorOp>>),
    }

    pub enum TerminatorOp {
        YieldNone,
        Yield(RuntimeValue<ValueType>),
    }

    impl ValueType {
        pub fn is_number(&self) -> bool {
            match self {
                ValueType::Native(NativeType::Usize) => true,
                _ => false
            }
        }
    }

    impl Typed for Op {
        type ValueType = ValueType;

        fn typ(&self) -> Self::ValueType {
            match self {
                Op::Const(cv) => ValueType::Native(match cv {
                    ConstValue::Bool(_) => NativeType::Bool,
                    ConstValue::Usize(_) => NativeType::Usize,
                    ConstValue::Str(_) => NativeType::Str,
                }),
                Op::VarDeclare(rv, _) => rv.typ(),
                Op::VarGet(_) => todo!("Variable value types"),
                Op::Neg(rv) => rv.typ(),
                Op::Add(a, _) => a.typ(),
                Op::If(_, t, None) => ValueType::Unit,
                Op::If(_, t, _) => t.typ(),
            }
        }
    }

    impl Typed for TerminatorOp {
        type ValueType = ValueType;

        fn typ(&self) -> Self::ValueType {
            match self {
                TerminatorOp::YieldNone => ValueType::Unit,
                TerminatorOp::Yield(rv) => rv.typ()
            }
        }
    }

    impl Display for ConstValue {
        fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
            match self {
                ConstValue::Bool(v) => write!(f, "{v}"),
                ConstValue::Usize(v) => write!(f, "{v}"),
                ConstValue::Str(v) => write!(f, "{v}"),
            }
        }
    }

    impl Display for NativeType {
        fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
            match self {
                NativeType::Bool => f.write_str("bool"),
                NativeType::Usize => f.write_str("usize"),
                NativeType::Str => f.write_str("str"),
            }
        }
    }

    impl<'a> Display for ValueType {
        fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
            match self {
                ValueType::Unit => f.write_str("()"),
                ValueType::Native(t) => Display::fmt(t, f),
            }
        }
    }

    impl<'a> Display for Printer<(usize, &'a ValueType)> {
        fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
            let (id, vt) = self.wrapped;
            write!(f, "%{}_{id}: {vt}", self.depth - 1)
        }
    }

    impl<'a> Display for Printer<(usize, &'a Op)> {
        fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
            // f.write_str("")
            let Printer { depth, wrapped: (id, _) } = self;
            let depth = *depth;
            let av = RuntimeValue::Local(depth - 1, 0, *id, ValueType::Unit);
            let vt = self.wrapped.1.typ();
            match self.wrapped.1 {
                // Op::Const(ConstValue::Bool(v)) => write!(f, "{av}: {vt} = const.bool {v}"),
                // Op::Const(ConstValue::Usize(v)) => write!(f, "{av}: {vt} = const.usize {v}"),
                // Op::Const(ConstValue::Str(v)) => write!(f, "{av}: {vt} = const.str {v}"),
                Op::Const(v) => write!(f, "{av}: {vt} = const {v}"),
                Op::VarDeclare(v, n) => write!(f, "{av}: {vt} = var.declare {v} '{n}'"),
                Op::VarGet(v) => write!(f, "{av}: {vt} = var.get {v}"),
                Op::Neg(v) => write!(f, "{av}: {vt} = math.neg {v}"),
                Op::Add(l, r) => write!(f, "{av}: {vt} = math.add {l} {r}"),
                Op::If(p, t, Some(e)) => write!(f, "{av}: {vt} = statement.if.else ({}) ({}) ({})",
                                                Printer { depth, wrapped: p },
                                                Printer { depth, wrapped: t },
                                                Printer { depth, wrapped: e }),
                Op::If(p, t, None) => write!(f, "{av}: {vt} = statement.if ({}) ({})",
                                             Printer { depth, wrapped: p },
                                             Printer { depth, wrapped: t }),
            }
        }
    }

    impl<'a> Display for Printer<&'a TerminatorOp> {
        fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
            match self.wrapped {
                TerminatorOp::YieldNone => write!(f, "yield.none"),
                TerminatorOp::Yield(rv) => write!(f, "yield.value {}", rv),
            }
        }
    }

    #[test]
    fn it_can_print_body() -> Result<()> {
        impl<'a> BlockBuilder<'a, ValueType, Op, TerminatorOp> {
            pub fn _const(&mut self, c: ConstValue) -> Result<RuntimeValue<ValueType>> {
                Ok(self.op(Op::Const(c)))
            }

            pub fn neg(&mut self, n: RuntimeValue<ValueType>) -> Result<RuntimeValue<ValueType>> {
                if n.typ().is_number() {
                    Ok(self.op(Op::Neg(n)))
                } else {
                    Err(format!("expected a number for Neg op, but got {}", n.typ()))?
                }
            }

            pub fn add(&mut self, l: RuntimeValue<ValueType>, r: RuntimeValue<ValueType>) -> Result<RuntimeValue<ValueType>> {
                if l.typ().is_number() && r.typ().is_number() {
                    Ok(self.op(Op::Add(l, r)))
                } else {
                    Err(format!("expected 2 numbers for Add op, but got {} and {}", l.typ(), r.typ()))?
                }
            }

            pub fn _if(
                &mut self,
                p: impl FnOnce(&mut BlockBuilder<ValueType, Op, TerminatorOp>, Vec<RuntimeValue<ValueType>>) -> Result<TerminatorOp>,
                t: impl FnOnce(&mut BlockBuilder<ValueType, Op, TerminatorOp>, Vec<RuntimeValue<ValueType>>) -> Result<TerminatorOp>,
                e: impl FnOnce(&mut BlockBuilder<ValueType, Op, TerminatorOp>, Vec<RuntimeValue<ValueType>>) -> Result<TerminatorOp>,
            ) -> Result<RuntimeValue<ValueType>> {
                let p = self.body([], p)?;
                let t = self.body([], t)?;
                let e = self.body([], e)?;
                if t.typ() == e.typ() {
                    Ok(self.op(Op::If(p, t, Some(e))))
                } else {
                    Err(format!("if then and else blocks must return the same type: {} != {}", t.typ(), e.typ()))?
                }
            }
        }

        let body = Body::isolated(
            &[ValueType::Native(NativeType::Usize)],
            |bb, params| {
                let [_1] = *params.as_slice() else { return Err("expected 1 parameter")?; };
                let _2 = bb._const(ConstValue::Usize(1))?;
                let _3 = bb.neg(_2)?;
                let _4 = bb.add(_1, _3)?;
                let _5 = bb._if(
                    |bb, _| {
                        let _1_0 = bb._const(ConstValue::Bool(true))?;
                        Ok(TerminatorOp::Yield(_1_0))
                    },
                    |bb, _| {
                        Ok(TerminatorOp::Yield(_4))
                    },
                    |bb, _| {
                        let _1_0 = bb.op(Op::Const(ConstValue::Usize(0)));
                        Ok(TerminatorOp::Yield(_1_0))
                    },
                )?;
                Ok(TerminatorOp::Yield(_5))
            },
        )?;

        assert_eq!(format!("{}", Printer { depth: 0, wrapped: &body }), r#"|%0_0: usize| -> usize {
  $0_0: usize = const 1;
  $0_1: usize = math.neg $0_0;
  $0_2: usize = math.add %0_0 $0_1;
  $0_3: usize = statement.if.else (|| -> bool {
    $1_0: bool = const true;
    yield.value $1_0
  }) (|| -> usize {yield.value $0_2}) (|| -> usize {
    $1_0: usize = const 0;
    yield.value $1_0
  });
  yield.value $0_3
}"#);

        Ok(())
    }
}