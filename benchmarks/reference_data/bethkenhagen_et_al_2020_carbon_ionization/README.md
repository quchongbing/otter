# Bethkenhagen et al. (2020), carbon ionization

These CSV files are user-supplied digitizations of the carbon panel in
Fig. 3 of:

M. Bethkenhagen *et al.*, “Carbon ionization at gigabar pressures: An
ab initio perspective on astrophysical high-density plasmas,” *Physical
Review Research* **2**, 023260 (2020),
[doi:10.1103/PhysRevResearch.2.023260](https://doi.org/10.1103/PhysRevResearch.2.023260).

The article is distributed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
These coordinates are a digitization, not an author-supplied numerical
table; use the published article as the scientific source.

Each file has two comma-separated columns:

1. mass density in g cm\(^{-3}\);
2. the Fig. 3 carbon ionization state \(Z^{\rm free}\).

The file labels follow the published caption:

- `DFT-MD.csv`: conductivity/TRK-sum-rule \(Z^{\rm free}\), Eq. (5);
- `Purgatorio.csv`: Purgatorio average-atom prediction;
- `OPAL.csv`: OPAL prediction;
- `ATOMIC.csv`: ATOMIC prediction;
- `BU-EK.csv`: Beth–Uhlenbeck with Ecker–Kröll IPD;
- `BU-SP.csv`: Beth–Uhlenbeck with Stewart–Pyatt IPD;
- `PB.csv`: Beth–Uhlenbeck with Stewart–Pyatt IPD and Pauli blocking.

The different curves do not share one unique microscopic partition of
“free” electrons. In particular, the paper derives its DFT-MD values from
the conduction-band conductivity, whereas Otter reports both
\(\bar Z=Z-Q_{\rm ion}(R_{\rm WS})\) and \(Z^*=n_e^0/n_i\). The gallery
therefore uses these curves as definition-aware literature context rather
than claiming pointwise equivalence.
