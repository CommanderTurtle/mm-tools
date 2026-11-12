"""Names and enabled-state reporting for inference CPU residency."""



OFFLOAD_OPTIONS = (
    "text_encoder_cpu_offload", "dit_cpu_offload", "dit_cpu_offload_for_decode",
)


def enabled_cpu_offloads(args):
    parallelism = getattr(args, "parallelism", None)
    enabled = [name for name in OFFLOAD_OPTIONS if getattr(parallelism, name, False)]
    if getattr(getattr(args, "vae", None), "cpu_offload", False):
        enabled.append("vae_cpu_offload")
    return enabled
