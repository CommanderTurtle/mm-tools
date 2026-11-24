"""The model: Idol's denoiser on an adaptive token set, and how it is built and loaded.

    denoiser        Budget3DModel: Idol's LatentSeqMMFlowModel plus the Budget3D adapters
    adapters        the modules Budget3D adds to the denoiser, each zero at initialisation
    turbo           Idol-turbo's phase-LoRA experts, chosen per row
    encoders        the two frozen image encoders, exactly as Idol runs them
    load            Idol's released weights and decoders, trained Budget3D weights, the dense parity gate
    turbo_parity    the parity gate against Idol-turbo's own code
"""
