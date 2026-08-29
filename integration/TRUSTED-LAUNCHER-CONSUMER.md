# Track D trusted-launcher consumer contract

This is Track D's final input to the OD-14 interface design. It defines what
Track H's non-forking profile/child-table interface must express and the
consumer campaign Track D will run after the common packet returns. It does not
define Track H's syntax, copy launcher bytes, accept OD-13/OD-14 implementation
results, or claim adoption.

The machine authorities are:

- `trusted-launcher-consumer-requirements.json` for the profile capability and
  child-table contract; and
- `trusted-launcher-consumer-campaign.json` for the complete case population,
  expected named outcomes, evidence fields and verdict rules.

Both **2/2 documents** are canonical JSON and are checked by
`tools/check_trusted_launcher_consumer_readiness.py`.

## Track H interface conformance return

Track H may choose its interface representation, but its return must map
**20/20 TLIF requirements** to exact interface fields or mechanisms and bind
each mapping to evidence bytes. The conformance return has exactly these
**3/3 fields per row**: `requirement_id`, `interface_field_or_mechanism` and
`evidence_sha256`. This input specifies **20/20 requirements**; the currently
accepted conformance-return population is **0/20 rows**.

The interface must provide all of the following categories:

- immutable profile, ordered child-table and terminal-set identities;
- retained no-follow subject, authority and dependency closures;
- live launcher, bootstrap and interpreter identity;
- exact first-process environment, cwd, module and import-path boundaries;
- data-selected child kind, path/module, argv, logical program and I/O limits;
- bootstrap-mediated Python children with no env-shebang, PATH or generated
  console-script interpreter authority;
- post-child closure and descendant-lifecycle verification;
- **1/1** bounded canonical result channel and complete result identity binding;
- cancellation without descendant or success leakage; and
- pre-run/post-run preservation plus native operator-facing profile selection.

The interface must exclude **5/5 forbidden shapes**: a consumer-specific native
launcher, a consumer wrapper before the native launcher, env-shebang or
PATH-selected authority, subject-authored identity/result claims, and promotion
before post-descendant verification.

## TD-P1 profile requirement

TD-P1 binds **1/1** independent `git archive` export of the exact product commit
and its nested candidate manifest. Its ordered child population is **3/3**:

1. the externally bound candidate semantic validator;
2. the `plebian-hardware` replay through the bound bootstrap/interpreter; and
3. the `plebian-model-sizer` replay through that same boundary.

The profile rechecks subject, runtime and dependency closure after **3/3
children**, waits for every descendant, and emits **1/1 terminal record** whose
terminal set covers **3/3 children**. Success compares **9/9 invocation/fixture
pairs** byte-for-byte and preserves blocked, unknown and null semantics. It is
still not P1 freeze: identical product tree, candidate manifest, profile,
result and signatory bytes remain prerequisites.

## TD-HW profile requirement

TD-HW binds the exact product export and exact staged `plebian-hardware`
distribution/RECORD. Its ordered child population is **5/5**:

1. the hardware unittest suite from an empty external cwd;
2. staged `plebian-hardware show`;
3. staged `plebian-hardware inventory --json`;
4. staged `plebian-hardware gpu --json`; and
5. staged invalid argv `plebian-hardware --json inventory`, requiring status 2
   with empty stdout and **1/1 bounded redacted diagnostic line**.

The profile rechecks closure after **5/5 children**, waits for every descendant
and emits **1/1 terminal record** over **5/5 terminal members**. Launcher integrity
does not qualify hardware: physical, VM, denied-probe and non-x86 coverage
remain **4/4 separate prerequisites**, and every live record remains
`qualification_eligible=false` until those gates pass.

## Consumer campaign population

The common matrix contains **41/41 definitions**: CAL **6/6**, ENV **7/7**,
ID **6/6**, DEP **1/1**, ORD **2/2**, SUB **9/9** and RES **10/10**. The
secondary execution-chain matrix contains CHN **6/6**.

TD-P1 applies the common matrix to **1/1** candidate-validator target, CHN to
**2/2** replay targets, and **13/13** replay-startup variants to each of those
**2/2** replay targets. Those variants are the Python-on-PATH replacement plus
the complete product of **2/2 membership modes × 3/3 startup-hook kinds × 2/2
replay-visible roots**. With its **1/1** clean baseline, TD-P1 has **80/80
case-family target definitions**.

TD-HW applies the common matrix to **5/5** production surfaces (`make
hardware-check` and **4/4** staged invocations), CHN to **5/5** intentional
child chains, and adds **1/1** clean baseline. TD-HW has **236/236 case-family
target definitions**.

The combined campaign therefore has **316/316 case-family target definitions**.
Every definition runs on **2/2 independent exports**, producing **632/632
required case-family target-export rows**. Acceptance requires **632/632**
`PASS` or `REFUSED-AS-NAMED`, **0/632** NULL, and **0/632** HARNESS-FAIL.

Each **1/1 row** aggregates **1/1 case family**, **1/1 target** and **1/1
export**. It cannot hide an omitted mutation behind that aggregate: before
execution the harness freezes the complete one-change and required-combination
variant manifest for the family. Each row returns **12/12 completion fields**,
including the grade; required, executed, PASS, REFUSED-AS-NAMED, NULL and
HARNESS-FAIL variant denominators; and the variant- and evidence-manifest
hashes. A row is nonrejecting only when `variants_required` equals
`variants_executed` and the sum of `variants_pass` plus
`variants_refused_as_named`; the **2/2 rejecting populations** must each be
**0/N** over that same required-variant denominator.

Every row also carries its mutation delta/reachability proof, first-process
identity, marker/body-marker observations, canonical record, expected/observed
refusal, lifecycle evidence and pre/post preservation manifests. The campaign
requires **11/11 evidence categories**, identical frozen identities across
**2/2 exports**, and no open finding at any of **4/4 severity levels**.

## Present boundary

This package contains **0/632 result rows**, accepts **0/20 interface
conformance rows** and **0/2 upstream independent exports**, and consumes
**0/8 upstream return identities**. OD-13 and OD-14 remain **2/2 assignments**
rather than implementation results, and D-PR-003 condition 2 remains untouched.
The package removes consumer-design work from the critical path; it does not
move the adoption verdict.
