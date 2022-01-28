import logging
import torch
import os
from torch.optim.lr_scheduler import LambdaLR
from typing import List

from Modularity.modules import NewInteractionModel, PlannerModel, PositionModel, AloTransform, AdverbTransform, DummyPlannerModel, DummySequencePlannerModel, DummyInteractionModel
from Modularity.dataset import ModularDataset
from Modularity.evaluate import evaluate
from seq2seq.helpers import log_parameters

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


def train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, adverb_vocab_path: str, adverb_input_vocab_path: str, transitive_target_vocab_path: str,
          grid_size: int,
          planner_target_vocab_path: str, adverb_target_vocab_path: str, embedding_dimension: int,
          num_encoder_layers: int, encoder_dropout_p: float, encoder_bidirectional: bool, training_batch_size: int,
          test_batch_size: int, cnn_kernel_size: int, cnn_dropout_p: float, module: str, cnn_hidden_num_channels: int,
          simple_situation_representation: bool, encoder_hidden_size: int, learning_rate: float, adam_beta_1: float,
          adam_beta_2: float, lr_decay: float, lr_decay_steps: int, resume_from_file: str, max_training_iterations: int,
          output_directory: str, decoder_hidden_size: int, decoder_dropout_p: float, num_decoder_layers: int,
          simplified_architecture: bool, use_attention: bool, use_conditional_attention: bool, type_attention: str,
          attention_values_key: str, conditional_attention_values_key: str, upsample_isolated: int, collapse_alo: bool,
          max_decoding_steps: int, print_every: int, evaluate_every: int, k: int, weight_decay: float, modular: bool,
          isolate_adverb_types: List[str], only_keep_adverbs: List[str],
          max_training_examples=None, seed=42, **kwargs):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    cfg = locals().copy()

    torch.manual_seed(seed)

    logger.info("Loading Training set...")
    training_set = ModularDataset(data_path, output_directory, split="train", input_vocabulary_file=input_vocab_path,
                                  target_vocabulary_file=target_vocab_path, adverb_vocabulary_file=adverb_vocab_path,
                                  adverb_input_vocabulary_file=adverb_input_vocab_path,
                                  transitive_target_vocabulary_file=transitive_target_vocab_path,
                                  adverb_target_vocabulary_file=adverb_target_vocab_path,
                                  planner_target_vocabulary_file=planner_target_vocab_path,
                                  generate_vocabulary=generate_vocabularies,
                                  k=k, upsample_isolated=100, isolate_adverb_types=isolate_adverb_types,
                                  only_keep_adverbs=only_keep_adverbs,
                                  seed=seed, isolate_examples_with="cautiously", collapse_alo=collapse_alo)
    logger.info("Done Loading Training set.")
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))

    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path, adverb_input_vocab_path,
                                       adverb_target_vocab_path, adverb_vocab_path, planner_target_vocab_path,
                                       transitive_target_vocab_path)
        logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    logger.info("Loading Dev. set...")
    test_set = ModularDataset(data_path, output_directory, split="dev", input_vocabulary_file=input_vocab_path,
                              target_vocabulary_file=target_vocab_path, adverb_vocabulary_file=adverb_vocab_path,
                              adverb_input_vocabulary_file=adverb_input_vocab_path,
                              transitive_target_vocabulary_file=transitive_target_vocab_path,
                              planner_target_vocabulary_file=planner_target_vocab_path,
                              adverb_target_vocabulary_file=adverb_target_vocab_path, generate_vocabulary=False, k=k,
                              upsample_isolated=1, isolate_adverb_types=isolate_adverb_types, only_keep_adverbs=only_keep_adverbs, seed=seed,
                              isolate_examples_with="cautiously", collapse_alo=collapse_alo)

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.
    test_set.shuffle_data()
    logger.info("Done Loading Dev. set.")

    if module == "position":
        model = PositionModel(input_vocabulary_size=training_set.input_vocabulary_size,
                              num_cnn_channels=training_set.image_channels,
                              input_padding_idx=training_set.input_vocabulary.pad_idx,
                              target_vocabulary="",
                              input_vocabulary="input",
                              main_input_key="input_tensor",
                              **cfg)
    elif module == "planner":
        model = PlannerModel(input_vocabulary_size=training_set.input_vocabulary_size,
                             input_padding_idx=training_set.input_vocabulary.pad_idx,
                             planner_target_eos_idx=training_set.planner_target_vocabulary.eos_idx,
                             planner_target_pad_idx=training_set.planner_target_vocabulary.pad_idx,
                             planner_target_sos_idx=training_set.planner_target_vocabulary.sos_idx,
                             planner_target_vocabulary_size=training_set.planner_target_vocabulary.size,
                             target_vocabulary="planner_target",
                             input_vocabulary="input",
                             main_input_key="input_tensor",
                             **cfg)
    elif module == "dummy_planner":
        model = DummyPlannerModel(input_vocabulary_size=training_set.input_vocabulary_size,
                                  input_padding_idx=training_set.input_vocabulary.pad_idx,
                                  planner_target_eos_idx=training_set.planner_target_vocabulary.eos_idx,
                                  planner_target_pad_idx=training_set.planner_target_vocabulary.pad_idx,
                                  planner_target_sos_idx=training_set.planner_target_vocabulary.sos_idx,
                                  planner_target_vocabulary_size=training_set.planner_target_vocabulary.size,
                                  target_vocabulary="plnner_target",
                                  input_vocabulary="input",
                                  main_input_key="input_tensor",
                                  **cfg)
    elif module == "dummy_sequence_planner":
        model = DummySequencePlannerModel(input_vocabulary_size=training_set.input_vocabulary_size,
                                          input_padding_idx=training_set.input_vocabulary.pad_idx,
                                          planner_target_eos_idx=training_set.planner_target_vocabulary.eos_idx,
                                          planner_target_pad_idx=training_set.planner_target_vocabulary.pad_idx,
                                          planner_target_sos_idx=training_set.planner_target_vocabulary.sos_idx,
                                          planner_target_vocabulary_size=training_set.planner_target_vocabulary.size,
                                          target_vocabulary="planner_target",
                                          input_vocabulary="input",
                                          main_input_key="input_tensor",
                                          **cfg)
    elif module == "interaction":
        model = NewInteractionModel(input_vocabulary_size=training_set.input_vocabulary_size,
                                    num_cnn_channels=training_set.image_channels,
                                    input_padding_idx=training_set.input_vocabulary.pad_idx,
                                    transitive_target_eos_idx=training_set.transitive_target_vocabulary.eos_idx,
                                    transitive_target_pad_idx=training_set.transitive_target_vocabulary.pad_idx,
                                    transitive_target_sos_idx=training_set.transitive_target_vocabulary.sos_idx,
                                    transitive_target_vocabulary_size=training_set.transitive_target_vocabulary.size,
                                    target_vocabulary="transitive_target",
                                    input_vocabulary="input",
                                    main_input_key="input_tensor",
                                    **cfg)
    elif module == "dummy_interaction":
        model = DummyInteractionModel(input_vocabulary_size=training_set.input_vocabulary_size,
                                      transitive_input_vocabulary_size=training_set.planner_target_vocabulary.size,
                                      transitive_input_padding_idx=training_set.planner_target_vocabulary.pad_idx,
                                      num_cnn_channels=training_set.image_channels,
                                      input_padding_idx=training_set.input_vocabulary.pad_idx,
                                      transitive_target_eos_idx=training_set.transitive_target_vocabulary.eos_idx,
                                      transitive_target_pad_idx=training_set.transitive_target_vocabulary.pad_idx,
                                      transitive_target_sos_idx=training_set.transitive_target_vocabulary.sos_idx,
                                      transitive_target_vocabulary_size=training_set.transitive_target_vocabulary.size,
                                      target_vocabulary="transitive_target",
                                      input_vocabulary="input",
                                      main_input_key="input_tensor",
                                      **cfg)
    elif module == "alo_transform":
        model = AloTransform(final_input_vocabulary_size=training_set.adverb_target_vocabulary.size,
                             final_input_padding_idx=training_set.adverb_target_vocabulary.pad_idx,
                             target_eos_idx=training_set.target_vocabulary.eos_idx,
                             target_pad_idx=training_set.target_vocabulary.pad_idx,
                             target_sos_idx=training_set.target_vocabulary.sos_idx,
                             target_vocabulary_size=training_set.target_vocabulary.size,
                             input_vocabulary_size=training_set.input_vocabulary.size,
                             input_padding_idx=training_set.input_vocabulary.pad_idx,
                             target_vocabulary="target",
                             input_vocabulary="adverb_target",
                             main_input_key="adverb_target_tensor",
                             **cfg)
    elif module == "adverb_transform":
        model = AdverbTransform(input_vocabulary_size=training_set.input_vocabulary_size,
                                adverb_input_vocabulary_size=training_set.adverb_input_vocabulary.size,
                                adverb_input_padding_idx=training_set.adverb_input_vocabulary.pad_idx,
                                input_padding_idx=training_set.input_vocabulary.pad_idx,
                                adverb_target_eos_idx=training_set.adverb_target_vocabulary.eos_idx,
                                adverb_target_pad_idx=training_set.adverb_target_vocabulary.pad_idx,
                                adverb_target_sos_idx=training_set.adverb_target_vocabulary.sos_idx,
                                adverb_target_vocabulary_size=training_set.adverb_target_vocabulary.size,
                                adverb_embedding_input_size=training_set.adverb_vocabulary.size,
                                target_vocabulary="adverb_target_vocabulary",
                                input_vocabulary="adverb_input",
                                main_input_key="adverb_input_tensor",
                                **cfg)
    else:
        raise NotImplementedError("Module %s is not implemented." % module)
    model = model.to(device)
    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2),
                                 weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_metric = 0
    best_loss = float('inf')
    if resume_from_file:
        assert os.path.isfile(resume_from_file), "No checkpoint found at {}".format(resume_from_file)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file))
        optimizer_state_dict = model.load_model(resume_from_file)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        best_metric = model.best_metric
        logger.info("Loaded checkpoint '%s' (iter %d, best %s %5.2f)" % (resume_from_file, start_iteration,
                                                                         model.metric, best_metric))

    its_to_save = [10000, 50000, 100000, 150000]
    next_it_idx_to_save = 0

    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        training_set.shuffle_data()
        for batch in training_set.get_data_iterator(input_keys=model.input_keys,
                                                    input_keys_to_pad=model.input_keys_to_pad,
                                                    target_keys=model.target_keys, shuffle=True,
                                                    target_keys_to_pad=model.target_keys_to_pad,
                                                    batch_size=training_batch_size, model_to_yield=module,
                                                    modular=modular):
            is_best = False
            model.train()

            # Forward pass.
            outputs = model(batch)
            loss = model.get_loss(**outputs,
                                  **batch["targets"])

            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.update_state(is_best=is_best)

            # Print current metrics.
            if training_iteration % print_every == 0:
                metrics = model.get_metrics(
                    **outputs, **batch["targets"])
                learning_rate = scheduler.get_lr()[0]
                metrics_string = ', '.join(["%s %5.2f" % (key, value) for key, value in metrics.items()])
                logger.info("Iteration %08d, loss %8.4f, learning_rate %.5f, %s" % (
                    training_iteration, loss, learning_rate, metrics_string))

            # Evaluate on test set.
            if training_iteration % evaluate_every == 0:
                with torch.no_grad():
                    model.eval()
                    logger.info("Evaluating..")
                    if model.sequence_decoder:
                        eval_metrics_dict = evaluate(
                            test_set.get_data_iterator(input_keys=model.input_keys,
                                                       input_keys_to_pad=model.input_keys_to_pad,
                                                       target_keys=model.target_keys,
                                                       target_keys_to_pad=model.target_keys_to_pad,
                                                       batch_size=test_batch_size, model_to_yield=module), model=model,
                            max_decoding_steps=max_decoding_steps,
                            max_examples_to_evaluate=kwargs["max_testing_examples"])
                    else:
                        eval_metrics_dict = evaluate(
                            test_set.get_data_iterator(input_keys=model.input_keys,
                                                       input_keys_to_pad=model.input_keys_to_pad,
                                                       target_keys=model.target_keys,
                                                       target_keys_to_pad=model.target_keys_to_pad,
                                                       batch_size=test_batch_size, model_to_yield=module), model=model,
                            max_examples_to_evaluate=kwargs["max_testing_examples"])
                    eval_metrics_string = ', '.join(
                        ["%s %5.2f" % (key, value) for key, value in eval_metrics_dict.items()])
                    logger.info("  Evaluation %s" % eval_metrics_string)
                    average_metric = sum(eval_metrics_dict.values()) / len(eval_metrics_dict)
                    if average_metric > best_metric:
                        is_best = True
                        best_metric = average_metric
                        model.update_state(metric=best_metric, is_best=is_best)
                    file_name = "checkpoint_iter_{}.pth.tar".format(str(training_iteration))
                    if next_it_idx_to_save < len(its_to_save):
                        if training_iteration > its_to_save[next_it_idx_to_save]:
                            model.save_checkpoint(file_name=file_name, is_best=is_best,
                                                  optimizer_state_dict=optimizer.state_dict())
                            next_it_idx_to_save += 1
                    if is_best:
                        model.save_checkpoint(file_name=file_name, is_best=is_best,
                                              optimizer_state_dict=optimizer.state_dict())

            training_iteration += 1
            if training_iteration > max_training_iterations:
                break
    logger.info("Finished training.")
