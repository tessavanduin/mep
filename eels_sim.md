# EELS-in-MEEP: what changed and why

Reproduction of the free-electron EELS simulations in Bezard *et al.*,
*High-Efficiency Coupling of Free Electrons to Sub-λ³ Modal Volume, High-Q
Photonic Cavities*, **ACS Nano 18, 10417 (2024)**.

The target is Eq. (1) of the paper — the García de Abajo loss probability for an
electron flying along the slot (x):

```
Γ(ℏω, y, z) = (e v / π ℏ ω) · Re ∫ Ê_x^ind(v t, y, z, ω) e^{+iω t} dt
Ê_x^ind(x, y, z, ω) = ∫ E_x^ind(x, y, z, t) e^{−iω t} dt
```

The recipe: FDTD with a moving electron current source → record `E_x(x,t)` along
the path → subtract an identical empty run to get the **induced** field → temporal
FT → project onto the electron phase with the prefactor.

---

## The physics corrections

**1. The source amplitude is now set (absolute normalisation).**
The electron is injected as a `J_x` current source. Its amplitude was previously
commented out, so the recorded field had an arbitrary scale and the paper's whole
selling point — absolute, fit-parameter-free probabilities — was lost. The
amplitude is fixed by charge conservation: while the hopping point source sits in
one voxel the charge must transit it, giving

```
j_x = q · resolution³ · β        (MEEP units)
```

(the `resolution³` the student used was actually the right power; the bug was the
missing amplitude and the SI/MEEP unit mix). See
`helper_functions.electron_source_amplitude`.

**2. The flux box is demoted to a vacuum charge meter.**
`∮ D·dA = Q_enclosed` is Gauss's law — a charge meter, not the EELS observable,
and in a dielectric it also picks up bound charge. Dividing the field by it (old
`cell 2`) corrupted the empty-subtraction. It now survives only as an optional
one-off **vacuum** check (`EELS_3D.py --charge-check`) that the source really
injects one electron. The loss probability comes from the trajectory projection,
never from a flux.

**3. Empty subtraction with identical normalisation.**
`E_ind = E_crystal − E_empty`, both runs using the same source amplitude and the
same (x,t) grid, so the bare electron field and the entrance/exit transients
cancel exactly. No per-run division by anything.

**4. Ring-down is recorded.**
The field is now recorded on the whole monitor line at **every** time step for the
**full** run (transit + `--ringdown-factor × transit`), instead of only while the
electron was inside the crystal. Mode line shapes live in the ring-down *after*
the electron has left; truncating it (old `before_time(end_time, …)`) destroys the
spectrum. **Caveat:** the Q ≈ 2.5×10⁵ cavity line cannot be resolved by direct
FDTD at all — its ring-down is astronomically long. The paper computes that mode
separately with the Green-tensor / modal-expansion method (their Eqs. 2–3); do the
same for the cavity itself. Direct FDTD here gives the slot/band modes and the
overall β-peak envelope.

**5. Robust phase / time-origin convention.**
The projection uses the *actual electron arrival time* at each pixel,
`t_e(x) = (x − x_start)/v`, with the temporal FT taken in absolute simulation time
(`t = 0` at sim start). Because Γ takes the **real part**, a wrong phase origin
silently changes the answer; tying both the FT origin and the projection phase to
the same electron event removes that ambiguity. The prefactor `e v/(πℏω)`, after
converting `∫dt → ∫dx/v`, becomes `e/(πℏω)∫dx` — so the per-pixel sum needs **no**
extra `1/v`, and the ad-hoc `/h_bar/1000*dt` and `conversion_factor` of the first
version are gone.

---

## The single normalisation constant (derived, not fitted)

Converting the dimensionless MEEP trajectory sum to SI:

```
E_SI      = (e / ε₀ a²) · E_meep            (Coulomb-law field unit, Q_e ≡ 1 MEEP charge)
Ê_SI      = (a/c) · E_SI · Ê_meep           (extra a/c from dt in the FT)
dx_SI     = a · dx_meep
Γ(ω)      = (e / π ℏ ω_SI) · Re Σ_j Ê_SI(x_j) e^{iω t_e,j} dx_SI
```

The length scale `a` cancels completely and leaves

```
Γ(ω) = (4 α / ω_SI) · Re Σ_j Ê_x^ind,meep(x_j, ω) · e^{iω t_e,j} · dx_meep
```

with **α the fine-structure constant**. EELS probabilities scaling with α is a
known sanity check, and its clean appearance here is a good sign the bookkeeping
is right. Implemented in `helper_functions.gamma_si_prefactor` and
`eels_postprocess.compute_gamma`. Per-energy units (`1/eV`, → `%` ×100) follow
from `dE = ℏ dω`.

**Still calibrate once.** The constant assumes the source injects exactly one
MEEP unit of charge. Run `--charge-check` in vacuum; if `∮D·dA = q_eff ≠ 1`,
divide the induced field by `q_eff`. For full confidence, validate the *absolute*
scale against an analytic case (aloof trajectory past a dielectric sphere, or bulk
/ Cherenkov loss in an infinite dielectric) before trusting the photonic-crystal
numbers.

---

## Files

| file | role |
|---|---|
| `helper_functions.py` | units, source amplitude, SI prefactor, vacuum charge meter |
| `EELS_3D.py` | FDTD field generation (one impact parameter per run) |
| `eels_postprocess.py` | induced field → FT → Γ(ω) → Gaussian broadening |

Requires the project's `geometries.py` (unchanged; provides
`SlottedTriangleLattice` / `SlottedTriangleLatticeCavity`). The old
`divergence.py` import is dropped.

### Typical run

```bash
# vacuum reference (same cell), then the crystal, then the cavity
python EELS_3D.py --empty
python EELS_3D.py                 # slotted PhC, beam in slot centre
python EELS_3D.py --cavity        # with shifted central holes
python EELS_3D.py --charge-check --empty   # one-off normalisation check
```

```python
from eels_postprocess import spectrum
E, G = spectrum("EELS_3D-out/EELS_3D-CRYSTAL_...h5",
                "EELS_3D-out/EELS_3D-EMPTY_...h5", method="dft")  # or "pade"
```

### Validation done

A synthetic single-mode induced field (even, phase-matched, damped) fed through
`temporal_FT → compute_gamma → gaussian_broaden` returns a **positive** peak at
the input mode energy, off-resonance background ~10⁻⁴ of the peak, and the peak
**collapses** when the electron velocity is detuned from the phase-matching
condition — confirming the FT convention, projection phase, and α-prefactor are
wired correctly. (Absolute scale still needs the analytic calibration above.)

### Remaining / suggested

- Use `method="dft"` to validate, `method="pade"` for sharper lines from a
  truncated series (Padé can produce spurious poles — always cross-check vs DFT).
- 18 nm (~24 px/a) is borderline; run one convergence check at finer resolution
  and rely on MEEP subpixel smoothing.
- Only `E_x` along x (parallel geometry) is recorded; the perpendicular geometry
  needs `E_z` with the electron along z.
- Exploit mirror symmetry (`symmetries=`) for a large speed-up.
