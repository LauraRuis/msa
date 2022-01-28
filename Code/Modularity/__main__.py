import argparse
import logging
import os
import torch

from Modularity.train import train
from Modularity.predict import predict_and_save, predict_and_save_full_model
from Modularity.dataset import ModularDataset
from Modularity.modules import NewInteractionModel, PlannerModel, PositionModel, AloTransform, AdverbTransform, FullModel, DummyPlannerModel, DummySequencePlannerModel, DummyInteractionModel
from Modularity.helpers import log_metrics

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger("Modular_gSCAN")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Meta Sequence to sequence models for Grounded SCAN")

# General arguments
parser.add_argument("--mode", type=str, default="train", help="")
parser.add_argument("--module", type=str, default="position", help="Which module to train or test. Options: "
                                                                   "position or planner.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_directory", type=str, default="output", help="Path where everything will be saved.")
parser.add_argument("--resume_from_file", type=str, default="", help="Full path to previously saved model to load.")
parser.add_argument("--output_folder_pattern", type=str, default="test_%s_model")

# Data arguments
parser.add_argument("--data_directory", type=str, default="data/gscan_modular_adverbs_data",
                    help="Path to folder with data.")
parser.add_argument("--dataset_file_name", type=str, default="dataset.txt",
                    help="The file name of the actual dataset that is in data_directory.")
parser.add_argument("--input_vocab_path", type=str, default="training_input_vocab.txt",
                    help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--target_vocab_path", type=str, default="training_target_vocab.txt",
                    help="Path to file with target vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--adverb_input_vocab_path", type=str, default="training_adverb_input_vocab.txt",
                    help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--planner_target_vocab_path", type=str, default="training_planner_target_vocab.txt",
                    help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--transitive_target_vocab_path", type=str, default="training_transitive_target_vocab.txt",
                    help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--adverb_target_vocab_path", type=str, default="training_adverb_target_vocab.txt",
                    help="Path to file with target vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--adverb_vocab_path", type=str, default="training_adverb_vocab.txt",
                    help="Path to file with adverb vocabulary as saved by Vocabulary class in dataset.py")
parser.add_argument("--generate_vocabularies", dest="generate_vocabularies", default=False, action="store_true",
                    help="Whether to generate vocabularies based on the data.")
parser.add_argument("--load_vocabularies", dest="generate_vocabularies", default=True, action="store_false",
                    help="Whether to use previously saved vocabularies.")

# Training and learning arguments
parser.add_argument("--training_batch_size", type=int, default=200)
parser.add_argument("--k", type=int, default=5, help="How many examples from the adverb_1 split to move to train.")
parser.add_argument("--upsample_isolated", type=int, default=100, help="How many times to upsample the isolated adverb examples.")
parser.add_argument("--test_batch_size", type=int, default=1, help="Currently only 1 supported due to decoder.")
parser.add_argument("--max_training_examples", type=int, default=None, help="If None all are used.")
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--lr_decay_steps', type=float, default=20000)
parser.add_argument("--adam_beta_1", type=float, default=0.9)
parser.add_argument("--adam_beta_2", type=float, default=0.999)
parser.add_argument("--weight_decay", type=float, default=0.)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--evaluate_every", type=int, default=1000, help="How often to evaluate the model by decoding the "
                                                                     "test set (without teacher forcing).")
parser.add_argument("--max_training_iterations", type=int, default=100000)

# Testing and predicting arguments
parser.add_argument("--max_testing_examples", type=int, default=None)
parser.add_argument("--splits", type=str, default="test", help="comma-separated list of splits to predict for.")
parser.add_argument("--max_decoding_steps", type=int, default=30, help="After 30 decoding steps, the decoding process "
                                                                       "is stopped regardless of whether an EOS token "
                                                                       "was generated.")
parser.add_argument("--output_file_name", type=str, default="predict.json")

# Situation Encoder arguments
parser.add_argument("--simple_situation_representation", dest="simple_situation_representation", default=True,
                    action="store_true", help="Represent the situation with 1 vector per grid cell. "
                                              "For more information, read grounded SCAN documentation.")
parser.add_argument("--image_situation_representation", dest="simple_situation_representation", default=False,
                    action="store_false", help="Represent the situation with the full gridworld RGB image. "
                                               "For more information, read grounded SCAN documentation.")
parser.add_argument("--cnn_hidden_num_channels", type=int, default=50)
parser.add_argument("--cnn_kernel_size", type=int, default=7, help="Size of the largest filter in the world state "
                                                                   "model.")
parser.add_argument("--cnn_dropout_p", type=float, default=0.1, help="Dropout applied to the output features of the "
                                                                     "world state model.")
parser.add_argument("--grid_size", type=int, default=6, help="Rows (or cols) of the grid world.")

# Command Encoder arguments
parser.add_argument("--embedding_dimension", type=int, default=25)
parser.add_argument("--num_encoder_layers", type=int, default=1)
parser.add_argument("--encoder_hidden_size", type=int, default=100)
parser.add_argument("--encoder_dropout_p", type=float, default=0.3, help="Dropout on instruction embeddings and LSTM.")
parser.add_argument("--encoder_bidirectional", dest="encoder_bidirectional", default=True, action="store_true")
parser.add_argument("--encoder_unidirectional", dest="encoder_bidirectional", default=False, action="store_false")

# Planner Decoder arguments
parser.add_argument("--num_decoder_layers", type=int, default=1)
parser.add_argument("--decoder_dropout_p", type=float, default=0.3, help="Dropout on decoder embedding and LSTM.")
parser.add_argument("--decoder_hidden_size", type=int, default=100)

# Overall architecture arguments
parser.add_argument("--simplified_architecture", type=str, default="yes", help="If yes, simplified architecture will be "
                                                                               "used.")
parser.add_argument("--use_attention", type=str, default="no", help="If yes, attn will "
                                                                     "be used.")
parser.add_argument("--use_conditional_attention", type=str, default="no", help="If yes, cond. attn will "
                                                                     "be used.")
parser.add_argument("--type_attention", type=str, default="luong", help="Options: bahdanau or luong.")
parser.add_argument("--attention_values_key", type=str, default="input_tensor", help="Which input to do attention over.")
parser.add_argument("--conditional_attention_values_key", type=str, default="world_state_tensor",
                    help="Which input to do conditional attention over.")
parser.add_argument("--collapse_alo", type=str, default="no", help="If yes, the adverb and alo model will be"
                                                                   " collapsed into one model from adverb input to "
                                                                   "final target.")
parser.add_argument("--modular", type=str, default="yes", help="If yes, the adverb and alo model will be"
                                                               " collapsed into one model from adverb input to "
                                                               "final target.")
parser.add_argument("--isolate_adverb_types", type=str, default="",
                    help="Which adverb types not to add to the training data.")
parser.add_argument("--only_keep_adverbs", type=str, default="",
                    help="Which adverb to keep in the training data.")
parser.add_argument("--model_size", type=str, default="small",
                    help="small or big, determines hidden sizes and dropout.")


def main(flags):

    if not flags["model_size"] in ["small", "big"]:
        raise ValueError("--model_size can only be small or big and not %s" % flags["model_size"])

    if flags["model_size"] == "big":
        flags["encoder_hidden_size"] = 400
        flags["decoder_hidden_size"] = 400
        flags["decoder_dropout_p"] = 0.2
        flags["encoder_dropout_p"] = 0.2
        flags["embedding_dimension"] = 50
    else:
        flags["encoder_hidden_size"] = 100
        flags["decoder_hidden_size"] = 100
        flags["decoder_dropout_p"] = 0.3
        flags["encoder_dropout_p"] = 0.3
        flags["embedding_dimension"] = 25

    assert flags["encoder_hidden_size"] == flags["decoder_hidden_size"], "Different encoder and decoder hidden sizes "\
                                                                         "not implemented."
    assert flags["simplified_architecture"] in ["yes", "no"], "Specify 'yes' or 'no' for --simplified_architecture."
    assert flags["use_attention"] in ["yes", "no"], "Specify 'yes' or 'no' for --use_attention."
    if flags["simplified_architecture"] == "yes":
        flags["simplified_architecture"] = True
    else:
        flags["simplified_architecture"] = False

    if flags["use_attention"] == "yes":
        flags["use_attention"] = True
    else:
        flags["use_attention"] = False

    if flags["use_conditional_attention"] == "yes":
        flags["use_conditional_attention"] = True
    else:
        flags["use_conditional_attention"] = False

    if flags["collapse_alo"] == "yes":
        flags["collapse_alo"] = True
    else:
        flags["collapse_alo"] = False

    if flags["modular"] == "yes":
        flags["modular"] = True
    else:
        flags["modular"] = False

    for argument, value in flags.items():
        logger.info("{}: {}".format(argument, value))

    if flags["isolate_adverb_types"]:
        if flags["isolate_adverb_types"] == "None":
            flags["isolate_adverb_types"] = ""
        flags["isolate_adverb_types"] = flags["isolate_adverb_types"].split(",")

    if flags["only_keep_adverbs"]:
        flags["only_keep_adverbs"] = flags["only_keep_adverbs"].split(",")

    # Some checks on the flags.
    if not os.path.exists(flags["output_directory"]):
        os.mkdir(os.path.join(os.getcwd(), flags["output_directory"]))

    data_path = os.path.join(flags["data_directory"], flags["dataset_file_name"])

    if not os.path.exists(data_path):
        raise ValueError("Cannot find data file at: %s" % data_path)

    if flags["module"] != "full":
        assert flags["module"] in flags["output_directory"], "Please specify output_directory with the module=%s in "\
                                                             "the name, for later retrieval purposes." % flags["module"]

    if flags["mode"] == "train":
        train(data_path=data_path, **flags)
    elif flags["mode"] == "test":
        splits = flags["splits"].split(",")
        if flags["module"] != "full":
            assert os.path.isfile(flags["resume_from_file"]), "No checkpoint found at {}".format(flags["resume_from_file"])
            checkpoint = torch.load(flags["resume_from_file"], map_location=torch.device('cpu'))
            for key, value in checkpoint.items():
                if key in flags:
                    flags[key] = value
            flags["decoder_hidden_size"] = flags["encoder_hidden_size"]
        else:
            checkpoints_per_module = {}
            for module_name in ["position", "planner", "interaction", "adverb_transform", "alo_transform"]:
                module_file_name = flags["resume_from_file"].split("/")[-1]
                logger.info("Module name: %s" % module_name)
                logger.info("Output folder pattern: %s" % flags["output_folder_pattern"])
                module_path_pattern = flags["output_folder_pattern"]
                if module_name != "adverb_transform":
                    module_path_pattern_split = flags["output_folder_pattern"].split("seed_")
                    module_path_pattern_seed = int(module_path_pattern_split[-1]) % 5
                    if not module_path_pattern_seed:
                        module_path_pattern_seed = 5
                    module_path_pattern = module_path_pattern_split[0] + "seed_" + str(module_path_pattern_seed)
                module_resume_from_file = os.path.join(module_path_pattern % module_name,
                                                       module_file_name)
                assert os.path.isfile(module_resume_from_file), "No checkpoint found at {}".format(
                    module_resume_from_file)
                logger.info("Loading checkpoint from %s" % module_resume_from_file)
                checkpoint = torch.load(module_resume_from_file, map_location=torch.device('cpu'))
                checkpoints_per_module[module_name] = checkpoint
        for split in splits:
            test_set = ModularDataset(data_path, flags["output_directory"], split=split,
                                      input_vocabulary_file=flags["input_vocab_path"],
                                      target_vocabulary_file=flags["target_vocab_path"],
                                      adverb_vocabulary_file=flags["adverb_vocab_path"],
                                      adverb_input_vocabulary_file=flags["adverb_input_vocab_path"],
                                      adverb_target_vocabulary_file=flags["adverb_target_vocab_path"],
                                      planner_target_vocabulary_file=flags["planner_target_vocab_path"],
                                      transitive_target_vocabulary_file=flags["transitive_target_vocab_path"],
                                      generate_vocabulary=False, k=5, upsample_isolated=1,
                                      isolate_adverb_types=None, only_keep_adverbs=None, module=flags["module"],
                                      seed=flags["seed"], output_folder_pattern=flags["output_folder_pattern"],
                                      isolate_examples_with="cautiously", collapse_alo=flags["collapse_alo"])
            if flags["module"] == "position":
                model = PositionModel(input_vocabulary_size=test_set.input_vocabulary_size,
                                      num_cnn_channels=test_set.image_channels,
                                      input_padding_idx=test_set.input_vocabulary.pad_idx,
                                      target_vocabulary="",
                                      input_vocabulary="input",
                                      main_input_key="input_tensor",
                                      **flags)
            elif flags["module"] == "planner":
                model = PlannerModel(input_vocabulary_size=test_set.input_vocabulary_size,
                                     input_padding_idx=test_set.input_vocabulary.pad_idx,
                                     planner_target_eos_idx=test_set.planner_target_vocabulary.eos_idx,
                                     planner_target_pad_idx=test_set.planner_target_vocabulary.pad_idx,
                                     planner_target_sos_idx=test_set.planner_target_vocabulary.sos_idx,
                                     planner_target_vocabulary_size=test_set.planner_target_vocabulary.size,
                                     target_vocabulary="planner_target",
                                     input_vocabulary="input",
                                     main_input_key="input_tensor",
                                     **flags)
            elif flags["module"] == "dummy_planner":
                model = DummyPlannerModel(input_vocabulary_size=test_set.input_vocabulary_size,
                                          input_padding_idx=test_set.input_vocabulary.pad_idx,
                                          planner_target_eos_idx=test_set.planner_target_vocabulary.eos_idx,
                                          planner_target_pad_idx=test_set.planner_target_vocabulary.pad_idx,
                                          planner_target_sos_idx=test_set.planner_target_vocabulary.sos_idx,
                                          planner_target_vocabulary_size=test_set.planner_target_vocabulary.size,
                                          target_vocabulary="planner_target",
                                          input_vocabulary="input",
                                          main_input_key="input_tensor",
                                          **flags)
            elif flags["module"] == "dummy_sequence_planner":
                model = DummySequencePlannerModel(input_vocabulary_size=test_set.input_vocabulary_size,
                                                  input_padding_idx=test_set.input_vocabulary.pad_idx,
                                                  planner_target_eos_idx=test_set.planner_target_vocabulary.eos_idx,
                                                  planner_target_pad_idx=test_set.planner_target_vocabulary.pad_idx,
                                                  planner_target_sos_idx=test_set.planner_target_vocabulary.sos_idx,
                                                  target_vocabulary="planner_target",
                                                  input_vocabulary="input",
                                                  main_input_key="input_tensor",
                                                  planner_target_vocabulary_size=test_set.planner_target_vocabulary.size,
                                                  **flags)
            elif flags["module"] == "interaction":
                model = NewInteractionModel(input_vocabulary_size=test_set.input_vocabulary_size,
                                            num_cnn_channels=test_set.image_channels,
                                            input_padding_idx=test_set.input_vocabulary.pad_idx,
                                            transitive_target_eos_idx=test_set.transitive_target_vocabulary.eos_idx,
                                            transitive_target_pad_idx=test_set.transitive_target_vocabulary.pad_idx,
                                            transitive_target_sos_idx=test_set.transitive_target_vocabulary.sos_idx,
                                            transitive_target_vocabulary_size=test_set.transitive_target_vocabulary.size,
                                            target_vocabulary="transitive_target",
                                            input_vocabulary="input",
                                            main_input_key="input_tensor",
                                            **flags)
            elif flags["module"] == "dummy_interaction":
                model = DummyInteractionModel(input_vocabulary_size=test_set.input_vocabulary_size,
                                              transitive_input_vocabulary_size=test_set.planner_target_vocabulary.size,
                                              transitive_input_padding_idx=test_set.planner_target_vocabulary.pad_idx,
                                              num_cnn_channels=test_set.image_channels,
                                              input_padding_idx=test_set.input_vocabulary.pad_idx,
                                              transitive_target_eos_idx=test_set.transitive_target_vocabulary.eos_idx,
                                              transitive_target_pad_idx=test_set.transitive_target_vocabulary.pad_idx,
                                              transitive_target_sos_idx=test_set.transitive_target_vocabulary.sos_idx,
                                              transitive_target_vocabulary_size=test_set.transitive_target_vocabulary.size,
                                              target_vocabulary="transitive_target",
                                              input_vocabulary="input",
                                              main_input_key="input_tensor",
                                              **flags)
            elif flags["module"] == "alo_transform":
                model = AloTransform(final_input_vocabulary_size=test_set.adverb_target_vocabulary.size,
                                     final_input_padding_idx=test_set.adverb_target_vocabulary.pad_idx,
                                     target_eos_idx=test_set.target_vocabulary.eos_idx,
                                     target_pad_idx=test_set.target_vocabulary.pad_idx,
                                     target_sos_idx=test_set.target_vocabulary.sos_idx,
                                     target_vocabulary_size=test_set.target_vocabulary.size,
                                     input_vocabulary_size=test_set.input_vocabulary.size,
                                     input_padding_idx=test_set.input_vocabulary.pad_idx,
                                     target_vocabulary="target",
                                     input_vocabulary="adverb_target",
                                     main_input_key="final_input_tensor",
                                     **flags)
            elif flags["module"] == "adverb_transform":
                model = AdverbTransform(input_vocabulary_size=test_set.input_vocabulary_size,
                                        adverb_input_vocabulary_size=test_set.adverb_input_vocabulary.size,
                                        adverb_input_padding_idx=test_set.adverb_input_vocabulary.pad_idx,
                                        input_padding_idx=test_set.input_vocabulary.pad_idx,
                                        adverb_target_eos_idx=test_set.adverb_target_vocabulary.eos_idx,
                                        adverb_target_pad_idx=test_set.adverb_target_vocabulary.pad_idx,
                                        adverb_target_sos_idx=test_set.adverb_target_vocabulary.sos_idx,
                                        adverb_target_vocabulary_size=test_set.adverb_target_vocabulary.size,
                                        adverb_embedding_input_size=test_set.adverb_vocabulary.size,
                                        target_vocabulary="adverb_target",
                                        input_vocabulary="adverb_input",
                                        main_input_key="adverb_input_tensor",
                                        **flags)
            elif flags["module"] == "full":
                model = FullModel(dataset=test_set,
                                  input_vocabulary=test_set.input_vocabulary,
                                  num_cnn_channels=test_set.image_channels,
                                  planner_target_vocabulary=test_set.planner_target_vocabulary,
                                  adverb_input_vocabulary=test_set.adverb_input_vocabulary,
                                  adverb_target_vocabulary=test_set.adverb_target_vocabulary,
                                  target_vocabulary=test_set.target_vocabulary,
                                  module_path_pattern=flags["output_folder_pattern"],
                                  checkpoints_per_module=checkpoints_per_module,
                                  **flags)
            else:
                raise NotImplementedError("Module %s is not implemented." % flags["module"])
            if not flags["module"] == "full":
                logger.info("Loading checkpoint from file at '{}'".format(flags["resume_from_file"]))
                model.load_model(flags["resume_from_file"])
                model = model.to(device)
                start_iteration = model.trained_iterations
                logger.info("Loaded checkpoint '{}' (iter {})".format(flags["resume_from_file"], start_iteration))
                output_file_path = os.path.join(flags["output_directory"], "predictions_module_%s_split_%s.json" % (
                    flags["module"], split))
                predict_and_save(test_set, model, output_file_path, model_to_yield=flags["module"],
                                 collapse_alo=flags["collapse_alo"],
                                 max_testing_examples=flags["max_testing_examples"],
                                 max_decoding_steps=flags["max_decoding_steps"], modular=flags["modular"])
            else:
                output_file_path = os.path.join(flags["output_directory"], "predictions_module_%s_split_%s.json" % (
                    flags["module"], split))
                _, metrics = predict_and_save_full_model(test_set, model, output_file_path, model_to_yield=flags["module"],
                                                         collapse_alo=flags["collapse_alo"],
                                                         max_testing_examples=flags["max_testing_examples"],
                                                         max_decoding_steps=flags["max_decoding_steps"],
                                                         modular=flags["modular"])
                metrics_output_file_path = os.path.join(flags["output_directory"], "all_metrics_module_%s_split_%s.txt" % (
                    flags["module"], split))
                log_metrics(metrics, metrics_output_file_path)


if __name__ == "__main__":
    input_flags = vars(parser.parse_args())
    main(flags=input_flags)


