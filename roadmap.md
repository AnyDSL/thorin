# Roadmap

## Basic Infrastructure

- [x] arena allocation of nodes
- [x] remove extra data from subclasses of `Def`
- [x] remove RTTI
- [x] `Lam` and `App` instead of `Continuation`
- [x] make analyses work with all nominals, not just `Lam`s
- [x] remove `Type` and make it a `Def`
- [x] PTS calculus of constructions style IR (mostly done)
- [x] remove arg/param lists
- [x] `Sigma`/`Arr` + `Tuple/Pack` and friends
- [-] `Union` and friends
- [ ] `Intersection` and friends (maybe?)
- [x] new recursive `rewrite`
- [ ] remove old `mangle` in favor or `rewrite` (wip)
- [x] remove old `import` in favor of `rewrite`
- [x] remove old `cleanup` in favor of `rewrite`
- [x] make IR extensible

## Memory

- [x] `div`/`mod` with side effect
- [ ] `alloc`/`dealloc` in favor of `slot` and `global`

## Axioms

- [x] use axioms instead of C++ classes
- [ ] remove `Lamm::Intrinsics` in favor of axioms (wip)
- [ ] rewrite normalizations (wip)
- [ ] add frontend types `sint m w`, `uint m w`, `float m w` that automatically convert to `int w`/`real w`; the mode `m` is glued to the op. (wip)
- [ ] make operations polymorphic in rank and dimensions

## Optimizations

- [ ] finish new optimizer (wip)
- [ ] rewrite all optimizations to work with new optimizer
    - [ ] inliner      (wip)
    - [ ] partial eval (wip)
    - [ ] mem2reg      (wip)
    - [ ] scalarize
    - [ ] eta conv
    - [ ] copy prop
    - [ ] tail rec elim (maybe can be merged with copy prop)
    - [ ] closure elim
    - [ ] closure conv
    - [ ] compile ptrn (wip)
    - [x] codegen prepare
    - [ ] reg2mem (for aggregates)
    - [ ] acc prepare (phase that prepares special `vectorize`/`cuda` and friends for code generation)
- [ ] remove old optimizations/passes and `replace` infrastructure

## Type Checking

- [ ] type checking (wip)
- [ ] `ErrorHandler` (wip)

## Debugging

- [x] remove static state from logging
- [x] remove `Loc`/`Debug` and make it a `Def`
- [x] add `meta`` field
- [ ] unit testing with gtest
- [ ] gcov integration

## Module support

- [ ] polish output (wip)
- [ ] frontend to read it again
- [ ] integrate with debugging infrastructure

## Backend

- [ ] rewrite C backend
- [ ] rewrite LLVM backend (wip)

## Future

* Thorin-based vectorizer
* C interface of thorin
* add possibility to add new thorin nodes/axioms from frontends

## Far Future

* native backends
