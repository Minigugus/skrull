use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::mlir::ops::{BodyDepth, OpId, ParameterId, RuntimeValue};

pub struct Scopes<VT> {
    live: Vec<Scoped<VT>>,
}

struct Scoped<VT> {
    params: Vec<RuntimeValue<VT>>,
    locals: BTreeMap<usize, RuntimeValue<VT>>,
}

pub struct Scope<'a, VT> {
    scopes: &'a mut Scopes<VT>,
    depth: usize,
}

impl<VT> Scopes<VT> {
    fn allocate(&mut self, depth: usize, params: impl Into<Vec<RuntimeValue<VT>>>) -> Scope<VT> {
        self.live.push(Scoped {
            params: params.into(),
            locals: Default::default(),
        });
        Scope {
            scopes: self,
            depth,
        }
    }

    pub fn new() -> Self {
        Self {
            live: vec![],
        }
    }

    pub fn root(&mut self, params: impl Into<Vec<RuntimeValue<VT>>>) -> Scope<VT> {
        self.allocate(0, params)
    }
}

impl<'a, VR> Drop for Scope<'a, VR> {
    fn drop(&mut self) {
        self.scopes.live.pop();
    }
}

impl<'a, VT> Scope<'a, VT> {
    pub fn nested(&mut self, params: impl Into<Vec<RuntimeValue<VT>>>) -> Scope<VT> {
        self.scopes.allocate(self.depth + 1, params)
    }

    fn get_param(&self, bd: BodyDepth, pi: ParameterId) -> Option<&RuntimeValue<VT>> {
        self.scopes.live
            .get(bd)
            .and_then(|s| s.params.get(pi))
    }

    fn get_op(&self, bd: BodyDepth, oi: OpId) -> Option<&RuntimeValue<VT>> {
        self.scopes.live
            .get(bd)
            .and_then(|s| s.locals.get(&oi))
    }

    pub fn get<FVT>(&self, r: &RuntimeValue<FVT>) -> Option<&RuntimeValue<VT>> {
        match r {
            RuntimeValue::Parameter(bd, _, pi, _) if *bd <= self.depth => self.get_param(*bd, *pi),
            RuntimeValue::Local(bd, _, oi, _) if *bd <= self.depth => self.get_op(*bd, *oi),
            _ => None
        }
    }

    pub fn bind<FVT>(&mut self, from: &RuntimeValue<FVT>, to: RuntimeValue<VT>) -> Result<(), Cow<'static, str>> {
        match from {
            RuntimeValue::Local(bd, _, oi, _) => if let Some(s) = self.scopes.live.get_mut(*bd) {
                s.locals.insert(*oi, to);
                Ok(())
            } else {
                Err("unknown depth")?
            },
            RuntimeValue::Parameter(..) => Err("cannot bind to a parameter")?,
            _ => Err("unknown depth or runtime value type")?
        }
    }
}

pub struct Loc {
    pub depth: BodyDepth,
    pub id: OpId,
    pub p: bool,
}

impl<'a, T> From<&'a RuntimeValue<T>> for Loc {
    fn from(value: &'a RuntimeValue<T>) -> Self {
        match value {
            RuntimeValue::Parameter(bd, _, pi, _) => Loc {
                depth: *bd,
                id: *pi,
                p: true,
            },
            RuntimeValue::Local(bd, _, oi, _) => Loc {
                depth: *bd,
                id: *oi,
                p: false,
            }
        }
    }
}

#[test]
fn scope_works() {
    let mut scopes = Scopes::<()>::new();
    let mut root = scopes.root(&[RuntimeValue::Parameter(0, 0, 0, ())]);

    assert_eq!(None, root.get(&RuntimeValue::Local(0, 0, 1, ())).cloned());

    {
        let mut nested = root.nested(&[RuntimeValue::Parameter(1, 0, 0, ())]);

        nested
            .bind(&RuntimeValue::Local(1, 0, 0, ()), RuntimeValue::Local(1, 42, 0, ()))
            .expect("bind should works");

        nested
            .bind(&RuntimeValue::Local(0, 0, 1, ()), RuntimeValue::Local(0, 43, 1, ()))
            .expect("bind should works");

        assert_eq!(None, nested.get(&RuntimeValue::Local(0, 0, 0, ())).cloned());
        assert_eq!(Some(RuntimeValue::Local(1, 42, 0, ())), nested.get(&RuntimeValue::Local(1, 0, 0, ())).cloned());
        assert_eq!(Some(RuntimeValue::Parameter(0, 0, 0, ())), nested.get(&RuntimeValue::Parameter(0, 0, 0, ())).cloned());
    }

    assert_eq!(Some(RuntimeValue::Parameter(0, 0, 0, ())), root.get(&RuntimeValue::Parameter(0, 0, 0, ())).cloned());
    assert_eq!(Some(RuntimeValue::Local(0, 43, 1, ())), root.get(&RuntimeValue::Local(0, 0, 1, ())).cloned());
}
