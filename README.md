# Skrull

> A language and a semantic source-to-source compiler.

Skrull is a Rust-like language (currently a subset of Rust).
Unlike Rust, it is designed NOT to give low level access to underlying OS or architecture.
Indeed, it doesn't and won't support unsafe code, pointers and FFI.
Actually it won't even give access to external resources at all 🙂 (no `println` for instance).
It also aim to be lazily compilable, i.e. to compile very fast in dev mode but to allow costly and
very efficient optimizations; therefore, macros won't be supported too 😊. However, it still focuses
on zero-cost abstractions so static types and analysis will remain.

### Wait, what? What's the point of program not able to interact with the outside world!?

I didn't say it couldn't interact with the outside world, just that the std/compiler won't give such access
:wink:. Skrull is a source-to-source compiler, therefore it can still access external resources via
dependencies injection, where dependency implementation is written in the target language (e.g. Java ou
TypeScript). The big advantage is that it makes Skrull safe by design, even for the developer (supply-chain
attacks not possible by design). Also, the source-to-source design makes it even safer as it becomes easy
to understand what a program does really. Oh and, by the way, the source-to-source compiler in [this
project is `#![no_std]`](src/lib.rs), with no unsafe code, no dependencies, just plain Rust code
(the `clap` dependency is only for the `skrull` CLI).

### Ok but IO access seems to be really cumbersome then?

It's less obvious than a `println` indeed, but Skrull is mainly for high-level, business logic, not for
low-level files manipulations and so on. Skrull best shine for sharing business logic across multiple
parts of the application, especially one written in different languages like Java and TypeScript for
the back-end and front-end, respectively. It then shines for protocols too: (de)serialization written
in Skrull works in all supported target languages, so there's only one source of truth for protocol
implementation.

### You convinced me, where is the download button?

Skrull is still highly experimental, there's no download button yet, sorry. However, you
can clone the project and run `cargo run -- --help` to use the CLI. You can fine some examples
in tests across the project ([e.g. this one is interesting](src/transform/java.rs#L1239)).

A lot of features are missing, I work on them when I have time so don't expect Skrull to become
production-ready anytime soon. It's an experimental, personal project I wrote mainly to test ideas
I have. Don't hesitate to open a discussion thread if you're interested 🙂.

## License

MIT
