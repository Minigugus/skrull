#![no_std]

extern crate alloc;

mod lexer;
mod parser;
mod types;
mod printer;
mod bytecode;
mod transformer;
mod mlir {
    pub mod ops;
}
