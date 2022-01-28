from Modularity.predict import predict
from seq2seq.helpers import sequence_accuracy
from Modularity.nn import get_exact_match

import torch
import torch.nn as nn
from typing import Iterator
from typing import Dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(data_iterator: Iterator, model: nn.Module, max_decoding_steps=-1, max_examples_to_evaluate=None) -> Dict[str, float]:
    metrics = {target: {"target_correct": 0, "target_total": 0} for target in model.target_keys}
    sequence_metrics = {target: {"accuracy": 0, "exact_match": 0, "total": 0} for target in model.target_keys_to_pad}
    for predictions_batch in predict(
            data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps,
            max_examples_to_evaluate=max_examples_to_evaluate):
        for target_name in model.target_keys:
            equal = torch.eq(predictions_batch["targets"]["%s_targets" % target_name].data,
                             predictions_batch["predictions"][target_name].data).long().sum().data.item()
            metrics[target_name]["target_correct"] += equal
            metrics[target_name]["target_total"] += len(predictions_batch["targets"]["%s_targets" % target_name])
        for target_name in model.target_keys_to_pad:
            sequence_preds = predictions_batch["predictions"][target_name + "_sequences"]
            sequence_pred_lengths = predictions_batch["predictions"][target_name + "_sequence_lengths"]
            target_lengths = torch.tensor(predictions_batch["targets"]["%s_lengths" % target_name], device=device,
                                          dtype=sequence_pred_lengths.dtype) - 1  # -1 because SOS gets removed
            sequence_targets = model.remove_start_of_sequence(predictions_batch["targets"]["%s_targets" % target_name])
            accuracy_per_sequence, exact_match_per_sequence = get_exact_match(sequence_preds, sequence_pred_lengths,
                                                                              sequence_targets, target_lengths)
            sequence_metrics[target_name]["accuracy"] += accuracy_per_sequence.mean().item()
            sequence_metrics[target_name]["exact_match"] += exact_match_per_sequence.mean().item()
            sequence_metrics[target_name]["total"] += 1

    final_metrics = {}
    for target_name in model.target_keys:
        final_metrics[target_name] = (metrics[target_name]["target_correct"]
                                      / metrics[target_name]["target_total"]) * 100.
    for target_name in model.target_keys_to_pad:
        final_metrics[target_name + "_accuracy"] = (sequence_metrics[target_name]["accuracy"]
                                                    / sequence_metrics[target_name]["total"]) * 100.
        final_metrics[target_name + "_exact_match"] = (sequence_metrics[target_name]["exact_match"]
                                                       / sequence_metrics[target_name]["total"]) * 100.
    return final_metrics
