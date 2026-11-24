"""Sophon-Realtime-32B-Tag: DuplexJev (listen + write, with sound tags) -> tag-tts talker (CosyVoice3 LM, text streamed in
bistream style while it is written) -> CosyVoice3 flow + HiFT, served as one vLLM-Omni model.

vllm serve <Sophon-Realtime-32B-Tag dir> --omni --deploy-config <sophon_realtime_tag_1gpu.yaml>
"""
__version__ = "1.6.2"
ARCH = "SophonRealtimeForConditionalGeneration"
TALKER_ARCH = "SophonTagTalker"
_done = False


def register():
    global _done
    if _done:
        return
    _done = True
    from transformers import AutoConfig
    from .config import SophonRealtimeConfig
    try:
        AutoConfig.register(SophonRealtimeConfig.model_type, SophonRealtimeConfig)
    except ValueError:
        pass
    try:
        from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config
        AutoConfig.register("cosyvoice3", CosyVoice3Config)
    except Exception:  # noqa: BLE001  already registered by vllm-omni
        pass
    try:  # the talker re-emits the voice prompt (embed.*): keep the latest instead of concatenating per step
        from vllm_omni.outputs.output_modality import TensorAccumulationStrategy, register_key_accumulation_strategy
        for k in ("embed.speech_token", "embed.speech_feat", "embed.embedding", "embed.speech_token_len"):
            register_key_accumulation_strategy(k, TensorAccumulationStrategy.REPLACE)
    except Exception:  # noqa: BLE001
        pass
    try:
        import duplexjev_vllm
        duplexjev_vllm.register()
    except Exception:  # noqa: BLE001
        pass
    from vllm import ModelRegistry
    if TALKER_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(TALKER_ARCH, "sophon_realtime.talker_tag:SophonTagTalker")
    try:
        from vllm.model_executor.models.registry import _LazyRegisteredModel
        from vllm_omni.model_executor.models.registry import OmniModelRegistry
        OmniModelRegistry.models.setdefault(TALKER_ARCH, _LazyRegisteredModel(module_name="sophon_realtime.talker_tag", class_name="SophonTagTalker"))
        OmniModelRegistry.models.setdefault("DuplexJevForConditionalGeneration", _LazyRegisteredModel(module_name="duplexjev_vllm.model", class_name="DuplexJevForConditionalGeneration"))
        OmniModelRegistry.models.setdefault("DuplexJevVisionForConditionalGeneration", _LazyRegisteredModel(module_name="duplexjev_vllm.vision_model", class_name="DuplexJevVisionForConditionalGeneration"))
        from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline
        from .pipeline import SOPHON_REALTIME_PIPELINE
        if SOPHON_REALTIME_PIPELINE.model_type not in OMNI_PIPELINES:
            register_pipeline(SOPHON_REALTIME_PIPELINE)
    except ImportError:
        pass
