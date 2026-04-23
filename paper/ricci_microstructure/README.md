# Ricci Microstructure arXiv Submission

Working draft of the Ricci cross-sectional edge paper, derived from the
10-axis empirical audit in `research/microstructure/FINDINGS.md`.

## Files

| Path | Role |
|------|------|
| `paper.md` | Self-contained draft: abstract, introduction, methodology, results summary, economic analysis, related work, limitations, conclusion. Converts to LaTeX via `pandoc paper.md -o paper.tex --bibliography=references.bib`. |
| `references.bib` | BibTeX entries for every citation used in `paper.md`. |
| `../../research/microstructure/FINDINGS.md` | Evidence appendix. Referenced verbatim from `paper.md`; reviewers re-run the pipeline from this document's replication section. |

## Target venues (ordered by speed to citable record)

1. **arXiv q-fin.TR** — preprint, 1-day turnaround, no review. Submit first.
2. **SSRN eLibrary — Mathematical Finance eJournal** — parallel to arXiv, for quant-finance indexing.
3. **Journal of Computational Finance** — practitioner rigour. Target for formal submission two weeks after arXiv.
4. **Quantitative Finance (Taylor & Francis)** — pure quant audience. Lead with Ricci theory, empirics secondary.

## Gate: arXiv submission checklist

- [ ] Abstract ≤ 1920 characters (arXiv limit).
- [ ] Title ≤ 240 characters.
- [ ] All figures rendered to PDF/EPS (not PNG) for LaTeX build.
- [ ] `references.bib` compiles under BibTeX (no TODO entries).
- [ ] Replication section cites commit SHA on `main` where evidence is frozen.
- [ ] Section 8 ("Limitations") covers single-session substrate, simulation-only P&L, single asset class, latency budget, absent execution-topology model.
- [ ] License statement: MIT code, CC-BY-4.0 text.
- [ ] Author ORCID attached.

## Open items (do not block arXiv submission)

- Multi-session substrate activation (FINDINGS §10 · U1). Adds one
  robustness paragraph if completed before submission, not required.
- Live-paper execution engine on Binance testnet (FINDINGS §10 · U2).
  Out of scope for v1; would upgrade the economic-viability section
  from simulation to semi-live.
