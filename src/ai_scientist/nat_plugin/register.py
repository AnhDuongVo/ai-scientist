"""NeMo Agent Toolkit plugin: `design_binders` takes a target name and returns a ranked candidate report.

Input: "target_name | contigs | hotspot_res(comma)" (uses the placeholder PDB and the simulator unless a
real BioNeMo endpoint is configured through BIONEMO_BASE_URL / NGC_API_KEY)."""

from __future__ import annotations

import os

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


class DesignBindersConfig(FunctionBaseConfig, name="design_binders"):
    llm_name: LLMRef
    backbones: int = 3
    seqs_per_backbone: int = 3
    use_real_bionemo: bool = False


@register_function(config_type=DesignBindersConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def design_binders(config: DesignBindersConfig, builder: Builder):
    from ..bionemo import NIMBioNeMo, SimulatedBioNeMo
    from ..pipeline import report_markdown, run
    from ..samples import DEMO_TARGET
    from .llm import LangChainAdapter

    model = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm = LangChainAdapter(model, model)
    use_real = config.use_real_bionemo or bool(os.getenv("BIONEMO_BASE_URL"))
    bio = NIMBioNeMo() if use_real else SimulatedBioNeMo()

    async def _run(request: str) -> str:
        """Design candidate protein binders. Input: 'target_name | contigs | hotspot1,hotspot2' (fields optional)."""
        parts = [p.strip() for p in request.split("|")]
        target = DEMO_TARGET.model_copy(update={"name": parts[0] or DEMO_TARGET.name})
        if len(parts) > 1 and parts[1]:
            target = target.model_copy(update={"contigs": parts[1]})
        if len(parts) > 2 and parts[2]:
            target = target.model_copy(update={"hotspot_res": [h.strip() for h in parts[2].split(",")]})
        report = await run(target, llm, bio, config.backbones, config.seqs_per_backbone, papers=[])
        return report_markdown(report)

    yield FunctionInfo.from_fn(_run, description=_run.__doc__)
