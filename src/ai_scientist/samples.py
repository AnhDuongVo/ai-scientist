"""A bundled demo target. Uses a minimal placeholder PDB so the pipeline runs with no downloads.

For a realistic run, replace `pdb` with a real target structure (download a PDB and read it), and set
`contigs`/`hotspot_res` to the epitope you want to target. The in-repo RFdiffusion example in NVIDIA's
Protein Binder Design blueprint targets the SARS-CoV-2 spike RBD; see the README.
"""

from __future__ import annotations

from .schemas import Target

# Minimal valid single-chain PDB (a few CA atoms) so RFdiffusion input validation passes offline.
_PLACEHOLDER_PDB = (
    "\n".join(
        f"ATOM  {i + 1:>5}  CA  ALA A{i + 1:>4}    {i * 3.8:8.3f}   0.000   0.000  1.00  0.00           C"
        for i in range(20)
    )
    + "\nEND\n"
)

DEMO_TARGET = Target(
    name="Example target (placeholder)",
    pdb=_PLACEHOLDER_PDB,
    binder_chain_target="A",
    contigs="A1-20/0 50-70",  # keep target A1-20, chain break, generate a 50-70 aa binder
    hotspot_res=["A10", "A11", "A12"],
    description="Placeholder target for the offline demo. Replace with a real structure for real runs.",
)
