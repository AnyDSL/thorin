# Roadmap

## Basic Infrastructure

* arena allocation of nodes (done)
* remove extra data from subclasses of `Def` (done)
* remove RTTI (done)
* `Lam` and `App` instead of `Continuation` (done)
* make analyses work with all nominals, not just `Lam`s (done)
* remove `Type` and make it a `Def` (done)
* PTS calculus of constructions style IR (mostly done)
* remove arg/param lists (done)
* `Sigma`/`Arr` + `Tuple/Pack` and friends (done)
* `Union` and friends (wip)
* `Intersection` and friends (maybe?)
* new recursive `rewrite` (done)
* remove old `mangle` in favor or `rewrite` (wip)
* remove old `import` in favor of `rewrite` (done)
* remove old `cleanup` in favor of `rewrite` (done)
* make IR extensible (done)

## Memory

* `div`/`mod` with side effect (done)
* `alloc`/`dealloc` in favor of `slot` and `global`

## Axioms

* use axioms instead of C++ classes (done)
* remove `Lamm::Intrinsics` in favor of axioms (wip)
* rewrite normalizations (wip)
* add frontend types `sint m w`, `uint m w`, `float m w` that automatically convert to `int w`/`real w`; the mode `m` is glued to the op. (wip)
* make operations polymorphic in rank and dimensions

## Optimizations

* finish new optimizer (wip)
* rewrite all optimizations to work with new optimizer
    * inliner      (wip)
    * partial eval (wip)
    * mem2reg      (wip)
    * scalarize
    * eta conv
    * copy prop
    * tail rec elim (maybe can be merged with copy prop)
    * closure elim
    * closure conv
    * compile ptrn (wip)
    * codegen prepare (done)
    * reg2mem (for aggregates)
    * acc prepare (phase that prepares special `vectorize`/`cuda` and friends for code generation)
* remove old optimizations/passes and `replace` infrastructure

## Type Checking

* type checking (wip)
* `ErrorHandler` (wip)

## Debugging

* remove static state from logging (done)
* remove `Loc`/`Debug` and make it a `Def` (done)
* add `meta`` field (done)
* unit testing with gtest
* gcov integration

## Module support

* polish output (wip)
* frontend to read it again
* integrate with debugging infrastructure

## Backend

* rewrite C backend
* rewrite LLVM backend (wip)

## Future

* Thorin-based vectorizer
* C interface of thorin
* add possibility to add new thorin nodes/axioms from frontends

## Far Future

* native backends
