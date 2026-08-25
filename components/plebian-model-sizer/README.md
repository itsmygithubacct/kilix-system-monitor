# plebian-model-sizer

This directory reserves the F106 component boundary and contains no executable
or fit-engine implementation yet.

F100 has not passed U5 and F100-C0 is not frozen. Consequently there is no
authoritative OS, display, service, RAM, VRAM, disk or concurrency reserve
object against which resource arithmetic can be implemented. A provisional
policy object is explicitly not an entry condition and is not present here.

The pre-freeze schemas and fixtures under ../../contracts/p1-candidate retain
null capacity identities, null reserve/available arithmetic and unknown or
blocked verdicts. The seven-command invocation candidate includes future sizer
calls so consumers can test fail-closed fixtures, but this component does not
install or impersonate those replay binaries.

D4 starts only after both gates are recorded. Until then, the absence of a
plebian-model-sizer entry point is intentional and checked by the aggregate
test suite.
