"""Sophon-Realtime-32B-Tag: thinker (DuplexJev v1.5 + tag-writing, merged) -> tag-tts talker (CosyVoice3 LM, bistream text input)
-> code2wav (CosyVoice3 flow + HiFT, vLLM-Omni native)."""
import os

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

from .env import vision

THINKER_ARCH = os.environ.get("SOPHON_THINKER_ARCH", "DuplexJevVisionForConditionalGeneration" if vision() else "DuplexJevForConditionalGeneration")
TAG_ARCH = "SophonTagTalker"
_CV = "vllm_omni.model_executor.stage_input_processors.cosyvoice3"

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
            stage_id=1, model_stage="cosyvoice3_talker", execution_type=StageExecutionType.LLM_AR, input_sources=(0,),
            engine_output_type="latent", model_subdir="talker", tokenizer_subdir="talker/CosyVoice-BlankEN", model_arch=TAG_ARCH,
            async_chunk_process_next_stage_input_func="sophon_realtime.proc.talker2code2wav_async_chunk",
            custom_process_next_stage_input_func=f"{_CV}.text2flow_full_payload",
            sampling_constraints={"detokenize": False, "stop_token_ids": [6562]},
            requires_full_payload_input=True,
        ),
        StagePipelineConfig(
            stage_id=2, model_stage="cosyvoice3_code2wav", execution_type=StageExecutionType.LLM_GENERATION, input_sources=(1,),
            final_output=True, final_output_type="audio", engine_output_type="latent", model_subdir="talker",
            tokenizer_subdir="talker/CosyVoice-BlankEN", model_arch=TAG_ARCH,
            sync_process_input_func=f"{_CV}.text2flow_token_only", requires_full_payload_input=True,
        ),
    ),
)
