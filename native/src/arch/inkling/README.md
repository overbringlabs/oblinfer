# Inkling pack

This pack is Inkling-2-Mini: the `model/` sources (weights, state, decode step, prefill, DSpark drafter,
speculation, generator, vision) and the `kernels/` decode and prefill kernels written against its
shapes, with the pack root on the include path so those sources keep including `"model/..."` and
`"kernels/..."`. Other packs will sit beside it under `src/arch/`, each with its own `model/` and
`kernels/` and its own collection of `ingine_kernels_<area>` targets; nothing outside `src/arch/<pack>/`
depends on a pack's shapes.
