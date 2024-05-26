use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter};

use crate::mlir::ops::{BodyDepth, OpId, ParameterId, RuntimeValue};

pub struct Scopes<T> {
    live: Vec<Scoped<T>>,
}

struct Scoped<T> {
    params: Vec<T>,
    locals: BTreeMap<usize, T>,
}

pub struct Scope<'a, T> {
    scopes: &'a mut Scopes<T>,
    depth: usize,
}

impl<T> Scopes<T> {
    fn allocate(&mut self, depth: usize, params: impl Into<Vec<T>>) -> Scope<T> {
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

    pub fn root(&mut self, params: impl Into<Vec<T>>) -> Scope<T> {
        self.allocate(0, params)
    }
}

impl<'a, VR: Debug> Debug for Scope<'a, VR> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        struct ScopeDisplay<'a, VR>(&'a Scoped<VR>);

        impl<'a, VR: Debug> Debug for ScopeDisplay<'a, VR> {
            fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
                let mut list = f.debug_list();
                for v in &self.0.params {
                    list.entry(v);
                }
                list.entry(&self.0.locals);
                list.finish()
            }
        }

        let mut list = f.debug_list();
        for scope in self.scopes.live.iter().take(self.depth + 1) {
            list.entry(&ScopeDisplay(scope));
        }
        list.finish()
    }
}

impl<'a, VR> Drop for Scope<'a, VR> {
    fn drop(&mut self) {
        self.scopes.live.pop();
    }
}

impl<'a, T> Scope<'a, T> {
    pub fn nested(&mut self, params: impl Into<Vec<T>>) -> Scope<T> {
        self.scopes.allocate(self.depth + 1, params)
    }

    fn get_param(&self, bd: BodyDepth, pi: ParameterId) -> Option<&T> {
        self.scopes.live
            .get(bd)
            .and_then(|s| s.params.get(pi))
    }

    fn get_op(&self, bd: BodyDepth, oi: OpId) -> Option<&T> {
        self.scopes.live
            .get(bd)
            .and_then(|s| s.locals.get(&oi))
    }

    pub fn get<FVT>(&self, r: &RuntimeValue<FVT>) -> Option<&T> {
        match r {
            RuntimeValue::Parameter(bd, _, pi, _) if *bd <= self.depth => self.get_param(*bd, *pi),
            RuntimeValue::Local(bd, _, oi, _) if *bd <= self.depth => self.get_op(*bd, *oi),
            _ => None
        }
    }

    pub fn bind<FVT>(&mut self, from: &RuntimeValue<FVT>, to: T) -> Result<(), Cow<'static, str>> {
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
    let mut scopes = Scopes::<&'static str>::new();
    let mut root = scopes.root(&["b"]);

    assert_eq!(None, root.get(&RuntimeValue::Local(0, 0, 1, ())).cloned());

    {
        let mut nested = root.nested(&["d"]);

        nested
            .bind(&RuntimeValue::Local(1, 0, 0, ()), "a")
            .expect("bind should works");

        nested
            .bind(&RuntimeValue::Local(0, 0, 1, ()), "c")
            .expect("bind should works");

        assert_eq!(None, nested.get(&RuntimeValue::Local(0, 0, 0, ())).cloned());
        assert_eq!(Some("a"), nested.get(&RuntimeValue::Local(1, 0, 0, ())).cloned());
        assert_eq!(Some("b"), nested.get(&RuntimeValue::Parameter(0, 0, 0, ())).cloned());
    }

    assert_eq!(Some("b"), root.get(&RuntimeValue::Parameter(0, 0, 0, ())).cloned());
    assert_eq!(Some("c"), root.get(&RuntimeValue::Local(0, 0, 1, ())).cloned());
}
