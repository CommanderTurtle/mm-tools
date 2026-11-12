# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from torchmetrics import Accuracy, AveragePrecision, F1Score, MatthewsCorrCoef, PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics.text import SacreBLEUScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.wer import WordErrorRate

from nemo.collections.common.metrics.classification_accuracy import ExactStringMatchMetric, TokenF1Score

__all__ = ['MetricStringToTorchMetric', 'TextMetricsSet', 'ClassificationMetricsSet']

# Dictionary that maps a metric string name to its corresponding torchmetric class.

MetricStringToTorchMetric = {
    'accuracy': Accuracy,
    'average_precision': AveragePrecision,
    'f1': F1Score,
    'token_f1': TokenF1Score,
    'pearson_corr_coef': PearsonCorrCoef,
    'spearman_corr_coef': SpearmanCorrCoef,
    'matthews_corr_coef': MatthewsCorrCoef,
    'exact_string_match': ExactStringMatchMetric,
    'rouge': ROUGEScore,
    'wer': WordErrorRate,
    'bleu': SacreBLEUScore,
}

TextMetricsSet = set(['rouge', 'wer', 'bleu'])

ClassificationMetricsSet = set(['accuracy', 'average_precision', 'f1', 'exact_string_match'])
