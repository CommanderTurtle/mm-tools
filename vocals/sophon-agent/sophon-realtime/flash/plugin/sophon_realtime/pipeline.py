"""Three stages: thinker (DuplexJev, listens to audio and writes text) -> talker (Qwen3-TTS, text streamed in token by token) -> code2wav (vocoder)."""
import os

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

# thinker architecture: DuplexJevVision (Qwen3-VL vision tower + DuplexJev hearing) when the served thinker has a vision tower, else audio-only DuplexJev. SOPHON_VISION / SOPHON_THINKER_ARCH override.
from .env import vision

THINKER_ARCH = os.environ.get("SOPHON_THINKER_ARCH", "DuplexJevVisionForConditionalGeneration" if vision() else "DuplexJevForConditionalGeneration")

_TTS = "vllm_omni.model_executor.stage_input_processors.qwen3_tts"
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
            stage_id=1, model_stage="talker", execution_type=StageExecutionType.LLM_AR, input_sources=(0,),
            engine_output_type="latent", model_subdir="talker", tokenizer_subdir="talker", model_arch="SophonTTSTalker",
            async_chunk_process_next_stage_input_func=f"{_TTS}.talker2code2wav_async_chunk",
            custom_process_next_stage_input_func=f"{_TTS}.talker2code2wav_full_payload",
            sampling_constraints={"detokenize": False, "stop_token_ids": [2150]},
            requires_full_payload_input=True,
        ),
        StagePipelineConfig(
            stage_id=2, model_stage="code2wav", execution_type=StageExecutionType.LLM_GENERATION, input_sources=(1,),
            final_output=True, final_output_type="audio", engine_output_type="audio", model_subdir="talker", tokenizer_subdir="talker",
            model_arch="Qwen3TTSCode2Wav", sampling_constraints={"detokenize": True}, requires_full_payload_input=True,
        ),
    ),
)
