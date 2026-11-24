"""Sophon-Realtime-32B-E2E: thinker (DuplexJev v1.5 + tag-writing, merged) -> E2E talker (our engine, one frame per decode step)
-> E2E decoder (2-step DiT + HiFT)."""
import os

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.outputs.output_modality import TensorAccumulationStrategy, register_key_accumulation_strategy

from .env import vision

register_key_accumulation_strategy("codes.audio", TensorAccumulationStrategy.CONCAT_DIM0)
THINKER_ARCH = os.environ.get("SOPHON_THINKER_ARCH", "DuplexJevVisionForConditionalGeneration" if vision() else "DuplexJevForConditionalGeneration")
STOP = 151645

SOPHON_REALTIME_PIPELINE = PipelineConfig(
    model_type="sophon_realtime",
    model_arch=THINKER_ARCH,
    hf_architectures=("SophonRealtimeForConditionalGeneration",),
    duplex_plugin="sophon_realtime.duplex.SophonRealtimeDuplexPlugin",
    default_session_mode="turn",
    stages=(
        StagePipelineConfig(
            stage_id=0, model_stage="thinker", execution_type=StageExecutionType.LLM_AR, input_sources=(),
            final_output=True, final_output_type="text", owns_tokenizer=True, requires_multimodal_data=True,
            engine_output_type="latent", model_subdir="thinker", tokenizer_subdir="thinker", model_arch=THINKER_ARCH,
            async_chunk_process_next_stage_input_func="sophon_realtime.proc.thinker2talker_async_chunk",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1, model_stage="e2e_talker", execution_type=StageExecutionType.LLM_AR, input_sources=(0,),
            engine_output_type="latent", model_subdir="talker220", tokenizer_subdir="talker220", model_arch="SophonE2ETalker",
            async_chunk_process_next_stage_input_func="sophon_realtime.proc.talker2decoder_async_chunk",
            sampling_constraints={"detokenize": False, "stop_token_ids": [STOP]},
            requires_full_payload_input=True,
        ),
        StagePipelineConfig(
            stage_id=2, model_stage="e2e_decoder", execution_type=StageExecutionType.LLM_GENERATION, input_sources=(1,),
            final_output=True, final_output_type="audio", engine_output_type="audio", model_subdir="talker220",
            tokenizer_subdir="talker220", model_arch="SophonE2EDecoder", requires_full_payload_input=True,
        ),
    ),
)
