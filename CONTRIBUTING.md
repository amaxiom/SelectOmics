# Contributing to SelectOmics

Thank you for considering it. This document says what the project expects, so
that a contribution is not rejected for a reason nobody wrote down.

Taking part also means agreeing to the [Code of Conduct](CODE_OF_CONDUCT.md).
It is short, it comes down to discussing the work rather than the person doing
it, and it says where to report anything that troubles you.

---

## If you are here to report a problem

**Include your class counts.** It is the single most useful thing a SelectOmics
bug report can contain, and it is not busywork.

In the 0.7.1 bug sweep, six of the twelve defects found were one failure mode in
different clothes: **a class missing from some split**. It surfaced as a numpy
shape mismatch in cross-validation, as an unhandled error in one validation
protocol, as a held-out AUC reported as `nan`, as that `nan` labelled a `poor`
generalisation gap, as an opaque scikit-learn message on a continuous target,
and as gaps in the learning curve. Every one of them was silent or cryptic, and
every one was diagnosed from the class counts.

So please give:

- **The class counts**, overall and, if you can, in the training and test
  splits. The pipeline logs both at `verbose=True`.
- **What you ran**: the algorithm and `config.diff_from_defaults()`, which is
  every setting you changed and nothing else.
- **Whether a second algorithm agrees.** Re-run with `algorithm="RF"` if you used
  XGBoost, or the reverse. Behaviour that follows the algorithm points somewhere
  different from behaviour that follows the data.
- **SelectOmics, scikit-learn, xgboost, numpy and Python versions.**
- **What you expected, and what happened.**

There are templates under **Issues** that ask for exactly this.

**Before you report**, three things that look like bugs and are not:

- **The delivered panel is not the last step's.** The pipeline evaluation
  recommended an earlier step, because a later one pruned past the point where
  it helped. `results["recommendation"]["reason"]` says why.
- **Step 3 did not run.** It skips when fewer than 30 features reach it, which
  on wide data is most of the time. The log says so.
- **Adequacy warnings.** Small-n data trips the sample-size checks. They qualify
  the result; they do not stop the run.

If the documentation failed to tell you any of this, that is worth reporting
too.

---

## The one rule that is specific to this project

**A guard added in one place must be checked in its siblings.**

SelectOmics does the same thing in several places: three validation protocols
score every panel, four algorithms run through every step, and several code
paths split data into folds. A condition that one of them handles, the others
usually must handle too.

The 0.7.1 sweep found the consequence. `leave_one_out_validation` and
`bootstrap_validation` both skipped an iteration whose training split lacked a
class. `stratified_cv_validation`, the protocol carrying half the weight of the
recommendation, did not, and raised instead, losing every feature set's result
rather than one diagnostic row. The guard existed twice and was missing once.

So if you find yourself adding a guard to one protocol, one algorithm, or one
fold loop, stop and look for its siblings.

---

## Getting set up

```bash
git clone https://github.com/amaxiom/SelectOmics.git
cd SelectOmics
pip install -e ".[dev,viz]"
pytest
```

The suite fits real models on synthetic data and takes several minutes. Tests
marked `slow` are deselected by default; run them with `pytest -m slow`.

---

## Before opening a pull request

**The suite must pass.** Do not relax a test to make it pass. If behaviour
genuinely has to change, change the test in the same pull request and say in
the description why the old expectation was wrong.

**Add a test for the bug you fixed.** Reproduce it first, watch the test fail,
then fix it. A fix without a test that would have caught it can come undone.

**Do not change a published number without re-running what produced it.**
Selection is sensitive in ways that are easy to miss: hyperparameters are tuned
on the full matrix before Step 1 removes anything, so even a change in constant
columns can move the delivered panel. If your change could plausibly move a
result, re-run the affected benchmark or notebook and update the figure from
that run.

---

## Documentation

**Numbers in documentation must come from a run, not from memory.** Several
errors in this project's history were figures that drifted from the code that
produced them: a metrics filename, a default value, a lesson in the examples
whose numbers had changed underneath it.

If you quote a result, say where it came from. If you change behaviour that a
guide describes, update the guide in the same pull request. `docs/` holds the
Markdown guides; `user_guide/` holds the LaTeX source of the PDF, which is
rebuilt with `latexmk -pdf SelectOmics_User_Guide.tex`.

House style, for consistency with what is already written:

- No em dashes or en dashes. Use a comma, a colon, a semicolon or a full stop.
- Explain why a choice was made, not only what it does.
- Report negative results plainly. `benchmarks/BENCHMARKS.md` says where the
  method loses, and it stays that way.
- A warning that cannot be acted on is noise. Say what happened, why it matters,
  and what to do.

---

## Releases

```bash
python -m build
python -m twine check dist/*
```

Then install the built wheel into a clean environment and exercise it there,
including `selectomics --version` and one short pipeline run. Building is not
evidence that the artefact works.

---

## Configuration compatibility

Public API is what `SelectOmics.__all__` exports. A configuration field that is
removed does not break old config files: an unknown key loads with a
`UserWarning` naming it and is ignored. Say in `CHANGELOG.md` what replaced a
removed field, and why, because the warning alone cannot.
