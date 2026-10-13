import asyncio

import streamlit as st

from ai_scientist.bionemo import SimulatedBioNeMo
from ai_scientist.llm import FakeLLM
from ai_scientist.pipeline import report_markdown, run
from ai_scientist.samples import DEMO_TARGET

st.set_page_config(layout="wide", page_title="ai-scientist workflow")
st.markdown(
    """<style>.block-container{padding-top:1.3rem;max-width:1400px}h1{font-size:1.9rem!important}h2{font-size:1.3rem!important}h3{font-size:1.1rem!important}div[data-testid=stMetricValue]{font-size:1.5rem}textarea{font-size:13px!important}</style>""",
    unsafe_allow_html=True,
)


st.title("ai-scientist: target → candidate research report")
st.caption(
    "OFFLINE SIMULATION · Actual repository orchestration and ranking · Placeholder target; simulated RFdiffusion, ProteinMPNN and Boltz-2 responses. No live models, literature retrieval or biological predictions."
)
a, b = st.columns([1, 1.4])
with a:
    st.subheader("Input · target and design settings")
    st.write("**" + DEMO_TARGET.name + "**")
    st.caption(DEMO_TARGET.description)
    st.code(DEMO_TARGET.contigs, language=None)
    st.write("Hotspots: " + ", ".join(DEMO_TARGET.hotspot_res))
    backbones = st.number_input("Backbones", 1, 4, 2)
    seqs = st.number_input("Sequences per backbone", 1, 4, 3)
    seed = st.number_input("Simulation seed", 0, 9, 0)
    with st.expander("Inspect placeholder target PDB"):
        st.code(DEMO_TARGET.pdb, language=None)
    st.caption(
        "No target sequence is supplied: the folding stage evaluates the binder alone. Confidence does not measure binding."
    )
    if st.button("Run simulated design workflow", type="primary"):

        class TracedBio(SimulatedBioNeMo):
            def __init__(self, seed):
                super().__init__(seed=seed)
                self.calls = []

            def rfdiffusion(self, *args, **kw):
                out = super().rfdiffusion(*args, **kw)
                self.calls.append(
                    {"Stage": "RFdiffusion · simulated", "Output": "Backbone PDB", "Seed": kw.get("random_seed")}
                )
                return out

            def proteinmpnn(self, *args, **kw):
                out = super().proteinmpnn(*args, **kw)
                self.calls.append(
                    {
                        "Stage": "ProteinMPNN · simulated",
                        "Output": str(kw.get("num_seq_per_target")) + " sequences",
                        "Seed": None,
                    }
                )
                return out

            def boltz2(self, *args, **kw):
                out = super().boltz2(*args, **kw)
                self.calls.append({"Stage": "Boltz-2 · simulated", "Output": "Binder-only confidence", "Seed": None})
                return out

        bio = TracedBio(seed)
        result = asyncio.run(
            run(
                DEMO_TARGET,
                FakeLLM({}),
                bio,
                n_backbones=backbones,
                seqs_per_backbone=seqs,
                top_k=backbones * seqs,
                papers=[],
            )
        )
        st.session_state.update(result=result, calls=bio.calls, settings=(backbones, seqs, seed))
with b:
    st.subheader("Output · ranked candidates and trace")
    if st.session_state.get("settings") != (backbones, seqs, seed):
        st.info("Run the repository pipeline to generate the simulated candidates.")
    else:
        result = st.session_state.result
        st.write(
            f"**{result.n_backbones} backbones → {result.n_candidates} candidate sequences → ranked research report**"
        )
        st.dataframe(
            [
                {
                    "Rank": c.rank,
                    "ID": c.id,
                    "Composite": c.composite,
                    "Confidence (simulated)": c.complex_confidence,
                    "MPNN score": c.mpnn_score,
                    "Length": len(c.sequence),
                    "Co-folded": c.co_folded,
                }
                for c in result.top
            ],
            hide_index=True,
        )
        st.caption("Heuristic: 0.7 × confidence + 0.3 × max(0, 1.5 − MPNN score) / 1.5")
        with st.expander("Inspect candidate sequence", expanded=True):
            cid = st.selectbox("Candidate", [c.id for c in result.top])
            c = next(c for c in result.top if c.id == cid)
            st.code(c.sequence, language=None)
            st.caption(f"Backbone {c.backbone_index} · {len(c.sequence)} amino acids · simulated sequence")
        with st.expander("Inspect actual pipeline calls"):
            st.dataframe(st.session_state.calls, hide_index=True)
        with st.expander("Research report and limitations"):
            st.markdown(report_markdown(result))
        st.download_button("Download candidate report", report_markdown(result), "candidate_report.md")
        st.download_button("Download candidate JSON", result.model_dump_json(indent=2), "candidates.json")
