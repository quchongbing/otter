# Security policy

Otter is scientific software and is not intended to enforce a security
boundary. Nevertheless, dependency, archive-loading, code-execution, or data
integrity vulnerabilities should be reported privately through the repository
host’s security-advisory channel rather than through a public issue.

Only the latest development release is supported while the project is below
version 1.0. Please include a minimal reproducer, affected version or commit,
and an assessment of whether untrusted input is required.

Otter’s public NPZ state format is deliberately loadable with
`allow_pickle=False`. New interchange formats must preserve that property
unless a security review explicitly documents an alternative.
