#![no_std]

extern crate alloc;

pub mod lexer;
pub mod parser;
pub mod types;
pub mod printer;
pub mod bytecode;
pub mod transformer;
pub mod eval;

pub mod mlir {
    pub mod ops;
}
