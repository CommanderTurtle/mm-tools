# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from scripts.tts_comparison_report.reporting.metrics.specs import DistributionMetricSpec, MetricSpec


MetricsRegistry: list[MetricSpec] = [
    MetricSpec("wer_cumulative", "WER (cumulative)", True, 2, "%", 100),
    MetricSpec("cer_cumulative", "CER (cumulative)", True, 2, "%", 100),
    MetricSpec("wer_filewise_avg", "WER (filewise avg)", True, 2, "%", 100),
    MetricSpec("cer_filewise_avg", "CER (filewise avg)", True, 2, "%", 100),
    MetricSpec("utmosv2_avg", "UTMOS v2", False, 3),
    MetricSpec("ssim_pred_gt_avg", "SSIM (pred vs GT)", False, 4, optional=True),
    MetricSpec("ssim_pred_context_avg", "SSIM (pred vs context)", False, 4, optional=True),
    MetricSpec("eou_cutoff_rate", "EoU cut-off rate", True, 3, "", 1, False, True),
    MetricSpec("eou_silence_rate", "EoU silence rate", True, 3, "", 1, False, True),
    MetricSpec("eou_noise_rate", "EoU noise rate", True, 3, "", 1, False, True),
    MetricSpec("eou_error_rate", "EoU error rate", True, 3, "", 1, False, True),
    MetricSpec("total_gen_audio_seconds", "Total audio (sec)", None, 1, "", 1, False),
]


DistributionMetricsRegistry: list[DistributionMetricSpec] = [
    DistributionMetricSpec("cer", "CER", True, True, (0.0, 0.3)),
    DistributionMetricSpec("utmosv2", "UTMOS v2", False),
    DistributionMetricSpec("pred_context_ssim", "SSIM (pred vs context)", False, optional=True),
]
