"""Sophon-Realtime: DuplexJev (listen + write) -> Qwen3-TTS talker (speaks token by token while text is written) -> Code2Wav, released as a single vLLM-Omni model.

Flash build (installed by flash/install.sh as `sophon-realtime-flash`): vllm serve <Sophon-Realtime dir> --omni --deploy-config <sophon_realtime.yaml>
"""
__version__ = "1.6.2"
ARCH = "SophonRealtimeForConditionalGeneration"
TALKER_ARCH = "SophonTTSTalker"
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
        import duplexjev_vllm
        duplexjev_vllm.register()
    except Exception:  # noqa: BLE001  if duplexjev-vllm is missing, let the caller raise a clearer error
        pass
    from vllm import ModelRegistry
    if TALKER_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(TALKER_ARCH, "sophon_realtime.talker:SophonTTSTalker")
    try:  # vLLM-Omni's own model registry
        from vllm.model_executor.models.registry import _LazyRegisteredModel
        from vllm_omni.model_executor.models.registry import OmniModelRegistry
        OmniModelRegistry.models.setdefault(TALKER_ARCH, _LazyRegisteredModel(module_name="sophon_realtime.talker", class_name="SophonTTSTalker"))
        OmniModelRegistry.models.setdefault("DuplexJevForConditionalGeneration", _LazyRegisteredModel(module_name="duplexjev_vllm.model", class_name="DuplexJevForConditionalGeneration"))
        OmniModelRegistry.models.setdefault("DuplexJevVisionForConditionalGeneration", _LazyRegisteredModel(module_name="duplexjev_vllm.vision_model", class_name="DuplexJevVisionForConditionalGeneration"))
        from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline
        from .pipeline import SOPHON_REALTIME_PIPELINE
        if SOPHON_REALTIME_PIPELINE.model_type not in OMNI_PIPELINES:
            register_pipeline(SOPHON_REALTIME_PIPELINE)
    except ImportError:
        pass
