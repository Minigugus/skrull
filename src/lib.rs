#![no_std]

extern crate alloc;

pub mod lexer;
pub mod parser;
pub mod types;
pub mod printer;
pub mod bytecode;
pub mod transform;
pub mod eval;
mod scope;

pub mod mlir {
    pub mod ops;
}
