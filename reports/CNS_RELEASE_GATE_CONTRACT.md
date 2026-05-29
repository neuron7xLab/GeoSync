# CNS Release Gate Contract

This static report defines the repository contract for the CNS release gate. Runtime JSON evidence is generated under `results/`.

## Checks

- protocol config validation
- manifest creation
- manifest verification
- report presence validation
- weighted quality decision

## Boundaries

- release gate only
- no runtime performance assertion
- no permanent JSON evidence committed to source control
