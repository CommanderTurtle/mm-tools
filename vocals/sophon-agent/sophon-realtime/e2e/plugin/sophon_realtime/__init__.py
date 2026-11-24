"""Sophon-Realtime-32B-E2E as one vLLM-Omni model: thinker (DuplexJev, listens + writes the tagged reply)
-> talker (the E2E talker engine, driven one frame per decode step) -> decoder (2-step DiT + HiFT).

E2E build (installed by e2e/install.sh as `sophon-realtime-e2e`): vllm serve <Sophon-Realtime-32B-E2E dir> --omni --deploy-config <yaml>
"""
__version__ = "1.6.2"
ARCH = "SophonRealtimeForConditionalGeneration"
TALKER_ARCH = "SophonE2ETalker"
DECODER_ARCH = "SophonE2EDecoder"
_done = False


def register():
    global _done
    if _done:
        return
    _done = True
    from transformers import AutoConfig
    from .config import SophonE2EStageConfig, SophonRealtimeConfig
    for c in (SophonRealtimeConfig, SophonE2EStageConfig):
        try:
            AutoConfig.register(c.model_type, c)
        except ValueError:
            pass
    try:
        import duplexjev_vllm
        duplexjev_vllm.register()
    except Exception:  # noqa: BLE001  if duplexjev-vllm is missing, let the caller raise a clearer error
        pass
    from vllm import ModelRegistry
    archs = ((TALKER_ARCH, "sophon_realtime.talker_e2e"), (DECODER_ARCH, "sophon_realtime.decoder_e2e"))
    for arch, mod in archs:
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, f"{mod}:{arch}")
    try:  # vLLM-Omni's own model registry
        from vllm.model_executor.models.registry import _LazyRegisteredModel
        from vllm_omni.model_executor.models.registry import OmniModelRegistry
        for arch, mod in archs:
            OmniModelRegistry.models.setdefault(arch, _LazyRegisteredModel(module_name=mod, class_name=arch))
        OmniModelRegistry.models.setdefault("DuplexJevForConditionalGeneration", _LazyRegisteredModel(module_name="duplexjev_vllm.model", class_name="DuplexJevForConditionalGeneration"))
        OmniModelRegistry.models.setdefault("DuplexJevVisionForConditionalGeneration", _LazyRegisteredModel(module_name="duplexjev_vllm.vision_model", class_name="DuplexJevVisionForConditionalGeneration"))
        from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline
        from .pipeline import SOPHON_REALTIME_PIPELINE
        if SOPHON_REALTIME_PIPELINE.model_type not in OMNI_PIPELINES:
            register_pipeline(SOPHON_REALTIME_PIPELINE)
    except ImportError:
        pass
