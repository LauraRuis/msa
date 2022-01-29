import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Iterator, Dict
import time
import json

from Modularity.nn import sequence_mask, get_exact_match
from Modularity.dataset import ModularDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def predict_and_save_full_model(dataset: ModularDataset, model: nn.Module, output_file_path: str,
                                collapse_alo: bool, max_testing_examples=None, max_decoding_steps=-1, modular=True,
                                **kwargs):
    """
        Predict all data in dataset with a model and write the predictions to output_file_path.
        :param dataset: a dataset with test examples
        :param model: a trained model from model.py
        :param output_file_path: a path where a .json file with predictions will be saved.
        :param collapse_alo:
        :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
        :param max_decoding_steps:
        """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        all_metrics = {}
        for module_name, module in model.modules.items():
            all_metrics[module_name] = {}
            all_metrics[module_name]["metrics"] = {target: {"target_correct": 0, "target_total": 0} for
                                                   target in module.target_keys}
            all_metrics[module_name]["sequence_metrics"] = {target: {"accuracy": 0, "exact_match": 0, "total": 0} for
                                                            target in module.target_keys_to_pad}
            all_metrics[module_name]["ground_truth_metrics"] = {target: {"target_correct": 0, "target_total": 0} for
                                                   target in module.target_keys}
            all_metrics[module_name]["ground_truth_sequence_metrics"] = {target: {"accuracy": 0, "exact_match": 0, "total": 0} for
                                                            target in module.target_keys_to_pad}
        all_metrics["final"] = {"accuracy": 0, "exact_match": 0, "total": 0}
        with torch.no_grad():
            i = 0
            for all_predictions_batch, full_accuracy, full_exact_match, full_pred, full_target, final_vocab, final_module in predict_full(
                    dataset.get_data_iterator(input_keys=model.input_keys, input_keys_to_pad=model.input_keys_to_pad,
                                              target_keys=model.target_keys,
                                              target_keys_to_pad=model.target_keys_to_pad, batch_size=1,
                                              model_to_yield="full", shuffle=False, modular=modular), model=model,
                    max_decoding_steps=max_decoding_steps, modular=modular):
                all_metrics["final"]["accuracy"] += full_accuracy
                all_metrics["final"]["exact_match"] += full_exact_match
                all_metrics["final"]["total"] += 1
                i += 1

                final_module = "alo_transform" if not collapse_alo else "adverb_transform"
                final_target_tensor = "target_tensor" if not collapse_alo else "adverb_target_tensor"
                input_str_sequence = all_predictions_batch[final_module]["original_batch"]["extra_information"][0]["example_information"]["command"]
                target_str_sequence = all_predictions_batch[final_module]["original_batch"]["extra_information"][0]["example_information"]["target_command"]
                derivation_spec = all_predictions_batch[final_module]["original_batch"]["extra_information"][0]["example_information"][
                    "derivation_representation"]
                situation_spec = all_predictions_batch[final_module]["original_batch"]["extra_information"][0]["example_information"][
                    "situation_representation"]

                final_accuracy = all_predictions_batch[final_module]["metrics"][final_target_tensor]["accuracy"] if modular else full_accuracy
                exact_match = all_predictions_batch[final_module]["metrics"][final_target_tensor]["exact_match"] if modular else full_exact_match
                output_str_sequence = all_predictions_batch[final_module]["predictions"][final_target_tensor]
                output_str_sequence = dataset.array_to_sentence(output_str_sequence.squeeze(), "target")[1:-1]
                pred_str_sequence = dataset.array_to_sentence(full_pred.squeeze(),
                                                              vocabulary=final_vocab, module=final_module)[1:-1]
                exact_match_bool = True if exact_match == 1. else False
                if exact_match_bool:
                    assert len(target_str_sequence), "Empty target."
                    assert pred_str_sequence == target_str_sequence, "False exact match. \nFinal module: %s\nFinal vocab: %s\nOther Prediction: %s\nPrediction: %s\nTarget: %s" % (
                        final_module,
                        final_vocab,
                        ' '.join(pred_str_sequence),
                        ' '.join(output_str_sequence),
                        ' '.join(target_str_sequence))
                current_output = {"input": input_str_sequence, "prediction": pred_str_sequence,
                                  "derivation": derivation_spec,
                                  "target": target_str_sequence, "situation": situation_spec,
                                  "attention_weights_input": [],
                                  "attention_weights_situation": [],
                                  "accuracy": final_accuracy,
                                  "exact_match": exact_match_bool}
                for module_name in all_predictions_batch.keys():
                    if not model.modules[module_name].sequence_decoder:
                        for target_name in all_predictions_batch[module_name]["metrics"]:
                            correct = all_predictions_batch[module_name]["metrics"][target_name] == 1.
                            if correct:
                                all_metrics[module_name]["metrics"][target_name]["target_correct"] += 1
                            all_metrics[module_name]["metrics"][target_name]["target_total"] += 1
                            correct = all_predictions_batch[module_name]["ground_truth_metrics"][target_name] == 1.
                            if correct:
                                all_metrics[module_name]["ground_truth_metrics"][target_name]["target_correct"] += 1
                            all_metrics[module_name]["ground_truth_metrics"][target_name]["target_total"] += 1
                    else:
                        for target_name in all_predictions_batch[module_name]["metrics"]:
                            if target_name not in all_metrics[module_name]["sequence_metrics"]:
                                continue
                            all_metrics[module_name]["sequence_metrics"][target_name]["accuracy"] += \
                                all_predictions_batch[module_name]["metrics"][target_name]["accuracy"]
                            all_metrics[module_name]["sequence_metrics"][target_name]["exact_match"] += \
                            all_predictions_batch[module_name]["metrics"][target_name]["exact_match"]
                            all_metrics[module_name]["sequence_metrics"][target_name]["total"] += 1
                            all_metrics[module_name]["ground_truth_sequence_metrics"][target_name]["accuracy"] += \
                                all_predictions_batch[module_name]["ground_truth_metrics"][target_name]["accuracy"]
                            all_metrics[module_name]["ground_truth_sequence_metrics"][target_name]["exact_match"] += \
                                all_predictions_batch[module_name]["ground_truth_metrics"][target_name]["exact_match"]
                            all_metrics[module_name]["ground_truth_sequence_metrics"][target_name]["total"] += 1
                    # TODO write all predictions in some way
                output.append(current_output)

        logger.info("Wrote predictions for {} examples.".format(i))
        metric_output = {}
        for module_name, module in model.modules.items():
            logger.info("")
            logger.info("")
            logger.info("Results for module %s" % module_name)
            metric_output[module_name] = {}
            for target_name in module.target_keys:
                metric = (all_metrics[module_name]["metrics"][target_name]["target_correct"] / all_metrics[module_name]["metrics"][target_name]["target_total"]) * 100.
                ground_truth_metric = (all_metrics[module_name]["ground_truth_metrics"][target_name]["target_correct"] / all_metrics[module_name]["ground_truth_metrics"][target_name]["target_total"]) * 100.
                logger.info("Average %s: %5.2f" % (target_name, metric))
                logger.info("")
                logger.info("Ground truth %s: %5.2f" % (target_name, ground_truth_metric))
                metric_output[module_name][target_name] = {"accuracy": metric,
                                                           "ground_truth_accuracy": ground_truth_metric}

            for target_name in module.target_keys_to_pad:
                if not all_metrics[module_name]["sequence_metrics"][target_name]["total"]:
                    continue
                accuracy_metric = (all_metrics[module_name]["sequence_metrics"][target_name]["accuracy"] / all_metrics[module_name]["sequence_metrics"][target_name][
                    "total"]) * 100.
                ground_truth_accuracy_metric = (all_metrics[module_name]["ground_truth_sequence_metrics"][target_name]["accuracy"] /
                                   all_metrics[module_name]["ground_truth_sequence_metrics"][target_name][
                                       "total"]) * 100.
                exact_match = (all_metrics[module_name]["sequence_metrics"][target_name]["exact_match"] / all_metrics[module_name]["sequence_metrics"][target_name]["total"]) * 100.
                ground_truth_exact_match = (all_metrics[module_name]["ground_truth_sequence_metrics"][target_name]["exact_match"] /
                               all_metrics[module_name]["ground_truth_sequence_metrics"][target_name]["total"]) * 100.
                logger.info("Average accuracy %s: %5.2f" % (target_name, accuracy_metric))
                logger.info("Average exact match %s: %5.2f" % (target_name, exact_match))
                logger.info("")
                logger.info("Ground truth Average accuracy %s: %5.2f" % (target_name, ground_truth_accuracy_metric))
                logger.info("Ground truth Average exact match %s: %5.2f" % (target_name, ground_truth_exact_match))
                metric_output[module_name][target_name] = {"accuracy": accuracy_metric,
                                                           "ground_truth_accuracy": ground_truth_accuracy_metric,
                                                           "exact_match": exact_match,
                                                           "ground_truth_exact_match": ground_truth_exact_match}
        final_accuracy = all_metrics["final"]["accuracy"] / all_metrics["final"]["total"] * 100.
        final_exact_match = all_metrics["final"]["exact_match"] / all_metrics["final"]["total"] * 100.
        logger.info("")
        logger.info("Average full accuracy: %5.2f" % final_accuracy)
        logger.info("Average full exact match: %5.2f" % final_exact_match)
        metric_output["full"] = {"accuracy": final_accuracy, "exact_match": final_exact_match}
        json.dump(output, outfile, indent=4)
    return output_file_path, metric_output


def predict_and_save(dataset: ModularDataset, model: nn.Module, output_file_path: str, model_to_yield: str,
                     collapse_alo: bool, modular: bool, max_testing_examples=None, max_decoding_steps=-1, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param model_to_yield: which model of the modules to choose from (full, position, planner, transitive, final)
    :param collapse_alo:
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    :param max_decoding_steps:
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        metrics = {target: {"target_correct": 0, "target_total": 0} for target in model.target_keys}
        sequence_metrics = {target: {"accuracy": 0, "exact_match": 0, "total": 0} for target in
                            model.target_keys_to_pad}
        with torch.no_grad():
            i = 0
            for predictions_batch in predict(
                    dataset.get_data_iterator(input_keys=model.input_keys, input_keys_to_pad=model.input_keys_to_pad,
                                              target_keys=model.target_keys,
                                              target_keys_to_pad=model.target_keys_to_pad, batch_size=1,
                                              model_to_yield=model_to_yield, modular=modular), model=model,
                    max_decoding_steps=max_decoding_steps):
                i += 1
                average_correct = 0
                current_correct = {target: 0 for target in model.target_keys}
                current_correct_sequence = {target: 0 for target in model.target_keys_to_pad}
                total = 0

                predictions = {target_key: None for target_key in model.target_keys}
                for target_key_to_pad in model.target_keys_to_pad:
                    predictions[target_key_to_pad] = None
                for target_name in model.target_keys:
                    equal = torch.eq(predictions_batch["targets"]["%s_targets" % target_name].data,
                                     predictions_batch["predictions"][target_name].data).long().sum().data.item()
                    predictions[target_name] = predictions_batch["predictions"][target_name].data  # TODO check if correct with position module
                    metrics[target_name]["target_correct"] += equal
                    current_correct[target_name] += equal
                    average_correct += equal
                    metrics[target_name]["target_total"] += len(
                        predictions_batch["targets"]["%s_targets" % target_name])
                    total += len(
                        predictions_batch["targets"]["%s_targets" % target_name])
                    accuracy = (equal / total) * 100.

                for target_name in model.target_keys_to_pad:
                    sequence_preds = predictions_batch["predictions"][target_name + "_sequences"]
                    sequence_pred_lengths = predictions_batch["predictions"][target_name + "_sequence_lengths"]
                    target_lengths = torch.tensor(predictions_batch["targets"]["%s_lengths" % target_name],
                                                  device=device,
                                                  dtype=sequence_pred_lengths.dtype) - 1  # -1 because SOS gets removed
                    sequence_targets = model.remove_start_of_sequence(
                        predictions_batch["targets"]["%s_targets" % target_name])
                    accuracy_per_sequence, exact_match_per_sequence = get_exact_match(sequence_preds,
                                                                                      sequence_pred_lengths,
                                                                                      sequence_targets, target_lengths)
                    accuracy = accuracy_per_sequence.mean().item()
                    average_correct += exact_match_per_sequence.sum().item()
                    current_correct_sequence[target_name] += exact_match_per_sequence.sum()
                    total += len(exact_match_per_sequence)
                    sequence_metrics[target_name]["accuracy"] += accuracy
                    sequence_metrics[target_name]["exact_match"] += exact_match_per_sequence.mean().item()
                    sequence_metrics[target_name]["total"] += 1
                    translated_prediction = dataset.array_to_sentence(sequence_preds.squeeze()[:sequence_pred_lengths.item()],
                                                                      vocabulary=model.target_vocabulary)
                    predictions[target_name] = translated_prediction

                input_str_sequence = predictions_batch["extra_information"][0]["example_information"]["command"]
                target_str_sequence = predictions_batch["extra_information"][0]["example_information"]["target_command"]
                derivation_spec = predictions_batch["extra_information"][0]["example_information"][
                    "derivation_representation"]
                situation_spec = predictions_batch["extra_information"][0]["example_information"][
                    "situation_representation"]

                module_input_sequence = predictions_batch["inputs"][model.main_input_key].squeeze()[1:-1]
                module_inputs = dataset.array_to_sentence(module_input_sequence,
                                                          vocabulary=model.input_vocabulary)
                if len(model.target_keys_to_pad):
                    module_target_sequence = predictions_batch["targets"][model.target_keys_to_pad[0] + "_targets"].squeeze()[1:-1]
                    module_targets = dataset.array_to_sentence(module_target_sequence,
                                                               vocabulary=model.target_vocabulary)
                else:
                    module_targets = {}
                    for target_key in model.target_keys:
                        module_targets[target_key] = predictions_batch["targets"][target_key + "_targets"].item()

                exact_match = True if average_correct == total else False
                output_str_sequence = []  # TODO: fill
                current_output = {"input": input_str_sequence,
                                  "module_input": module_inputs,
                                  "module_targets": module_targets,
                                  "prediction": output_str_sequence,
                                  "derivation": derivation_spec,
                                  "target": target_str_sequence,
                                  "adverb": predictions_batch["extra_information"][0]["example_information"]["adverb"],
                                  "type_adverb": predictions_batch["extra_information"][0]["example_information"]["type_adverb"],
                                  "verb": predictions_batch["extra_information"][0]["example_information"]["verb_in_command"],
                                  "situation": situation_spec,
                                  "attention_weights_input": [],
                                  "attention_weights_situation": [],
                                  "accuracy": accuracy,
                                  "exact_match": exact_match}
                for key, pred in predictions.items():
                    current_output[key + "_prediction"] = pred
                for target_name in model.target_keys:
                    current_output[target_name + "_prediction"] = predictions_batch["predictions"][target_name].item()
                    current_output[target_name + "_correct"] = current_correct[target_name]
                output.append(current_output)

        logger.info("Wrote predictions for {} examples.".format(i))
        for target_name in model.target_keys:
            metric = (metrics[target_name]["target_correct"] / metrics[target_name]["target_total"]) * 100.
            logger.info("Average %s: %5.2f" % (target_name, metric))
        for target_name in model.target_keys_to_pad:
            accuracy_metric = (sequence_metrics[target_name]["accuracy"] / sequence_metrics[target_name]["total"]) * 100.
            exact_match = (sequence_metrics[target_name]["exact_match"] / sequence_metrics[target_name]["total"]) * 100.
            logger.info("Average accuracy %s: %5.2f" % (target_name, accuracy_metric))
            logger.info("Average exact match %s: %5.2f" % (target_name, exact_match))
        json.dump(output, outfile, indent=4)
    return output_file_path


def predict_step(model: nn.Module, batch: Dict[str, torch.Tensor]):
    scores = model(batch)
    predictions = {}
    for key, values in scores.items():
        pred = values.max(dim=-1)[1]
        if not len(pred.shape):
            pred = pred.unsqueeze(dim=0)
        predictions[key.split("_scores")[0]] = pred
    return predictions


def predict_sequence(model: nn.Module, batch: Dict[str, torch.Tensor], max_decoding_steps: int):
    # Encode the input sequence.
    encoder_output = model.encode_input(batch["inputs"])

    # Iteratively decode the output.
    tokens = torch.tensor([model.target_sos_idx] * batch["batch_size"], dtype=torch.long,
                          device=device).unsqueeze(dim=0)
    decoding_iteration = 0
    if encoder_output["attention_values_lengths"]:
        use_attention = True
        all_attention_weights = torch.zeros([batch["batch_size"], max_decoding_steps,
                                             int(max(encoder_output["attention_values_lengths"]))], dtype=torch.float32,
                                            device=device)
    else:
        use_attention = False
        all_attention_weights = None
    output_sequences = torch.zeros([batch["batch_size"], max_decoding_steps], dtype=torch.long, device=device)
    decoding_done = False
    sequences_ended = torch.zeros([1, batch["batch_size"]], dtype=torch.long, device=device)
    ending_indices = torch.tensor([max_decoding_steps] * batch["batch_size"], dtype=torch.long,
                                  device=device).unsqueeze(dim=0)
    while not decoding_done and decoding_iteration < max_decoding_steps:
        decoder_step_inputs = {model.target_keys_to_pad[0] + "_targets": tokens}
        output, hidden, attention_weights = model.decode_step(decoder_step_inputs, encoder_output)
        encoder_output["hidden"] = hidden
        output = F.log_softmax(output, dim=-1)
        tokens = output.max(dim=-1)[1]
        ending_sequences = (tokens == model.target_eos_idx).long()
        ending_sequences = ending_sequences.to(device)
        output_sequences[:, decoding_iteration] = tokens * (1 - sequences_ended)
        if use_attention:
            all_attention_weights[:, decoding_iteration, :] = attention_weights.squeeze(dim=1) * (1 - sequences_ended).unsqueeze(dim=2)
        # Only update ending indice if the sequence hasn't ended before
        currently_ending_sequences = torch.clip(ending_sequences - sequences_ended, 0)
        ending_indices = torch.where(currently_ending_sequences == 1, decoding_iteration + 1, ending_indices)
        sequences_ended = torch.logical_or(sequences_ended, ending_sequences).long()
        decoding_iteration += 1
        if (tokens == model.target_eos_idx).long().sum() == batch["batch_size"]:
            decoding_done = True
    return {"%s_sequences" % model.target_keys_to_pad[0]: output_sequences,
            "%s_sequence_lengths" % model.target_keys_to_pad[0]: ending_indices.squeeze()}, all_attention_weights


def predict(data_iterator: Iterator, model: nn.Module, max_decoding_steps=-1,
            max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps:
    :param max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    if model.sequence_decoder:
        assert max_decoding_steps > 0, "Please specify max_decoding_steps for sequence decoders."
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for batch in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break

        # Forward pass.
        if model.sequence_decoder:
            outputs, attention_weights = predict_sequence(model, batch, max_decoding_steps)
            predictions = outputs
        else:
            outputs = model(batch)
            predictions = {target_name: outputs[target_name + "_scores"].max(dim=-1)[1]
                           for target_name in model.target_keys}
        yield {
            "inputs": batch["inputs"],
            "predictions": predictions,
            "targets": batch["targets"],
            "extra_information": batch["extra_information"]
        }
    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))


def predict_full(data_iterator: Iterator, model: nn.Module, max_decoding_steps=-1,
                 max_examples_to_evaluate=None, modular=True) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps:
    :param max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    assert max_decoding_steps > 0, "Please specify max_decoding_steps for sequence decoders."
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for batch in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break

        # Forward pass.
        yield model(batch, max_decoding_steps, gold_forward_pass=modular)
    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))