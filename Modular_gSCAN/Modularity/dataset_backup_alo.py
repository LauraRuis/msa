import logging
import torch
import numpy as np
import os
import json
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

from seq2seq.gSCAN_dataset import GroundedScanDataset, Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class AdverbVocabulary(object):
    """
    Object that maps adverb string to class to be processed by numerical models.
    """

    def __init__(self):
        self._idx_to_word = []
        self._word_to_idx = {}
        self._word_frequencies = Counter()

    def word_to_idx(self, word: str) -> int:
        return self._word_to_idx[word]

    def idx_to_word(self, idx: int) -> str:
        return self._idx_to_word[idx]

    def add_word(self, word: List[str]):
        if word not in self._word_to_idx:
            self._word_to_idx[word] = self.size
            self._idx_to_word.append(word)
        self._word_frequencies[word] += 1

    def most_common(self, n=10):
        return self._word_frequencies.most_common(n=n)

    def contains_word(self, word: str) -> bool:
        if word in self._word_to_idx:
            return True
        else:
            return False

    @property
    def size(self):
        return len(self._idx_to_word)

    @classmethod
    def load(cls, path: str):
        assert os.path.exists(path), "Trying to load a vocabulary from a non-existing file {}".format(path)
        with open(path, 'r') as infile:
            all_data = json.load(infile)
            vocab = cls()
            vocab._idx_to_word = all_data["idx_to_word"]
            vocab._word_to_idx = defaultdict(int)
            for word, idx in all_data["word_to_idx"].items():
                vocab._word_to_idx[word] = idx
            vocab._word_frequencies = Counter(all_data["word_frequencies"])
        return vocab

    def to_dict(self) -> dict:
        return {
            "idx_to_word": self._idx_to_word,
            "word_to_idx": self._word_to_idx,
            "word_frequencies": self._word_frequencies
        }

    def save(self, path: str) -> str:
        with open(path, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)
        return path


class ModularDataset(GroundedScanDataset):

    def __init__(self, path_to_data: str, save_directory: str, k: int, split="train", input_vocabulary_file="",
                 adverb_input_vocabulary_file="", adverb_target_vocabulary_file="", planner_target_vocabulary_file="",
                 target_vocabulary_file="", adverb_vocabulary_file="", transitive_target_vocabulary_file="",
                 generate_vocabulary=False, upsample_isolated=1, isolate_adverb_types=None, only_keep_adverbs=None,
                 seed=1, isolate_examples_with="cautiously", module="", output_folder_pattern="", collapse_alo=False):
        if generate_vocabulary:
            self.adverb_vocabulary = AdverbVocabulary()
            self.adverb_input_vocabulary = Vocabulary()
            self.adverb_target_vocabulary = Vocabulary()
            self.planner_target_vocabulary = Vocabulary()
            self.transitive_target_vocabulary = Vocabulary()
        self.collapse_alo = collapse_alo
        if module == "full":
            assert output_folder_pattern, "Please provide the pattern of the output folders for all the modules."
        self.all_modules = ["position", "planner", "dummy_planner", "dummy_sequence_planner", "interaction",
                            "dummy_interaction", "adverb_transform", "alo_transform"]
        if isolate_adverb_types:
            assert isinstance(isolate_adverb_types, list), "Please provide list of strings to isolate_adverb_types."
            logger.info("Isolating adverb types: %s" % ','.join(isolate_adverb_types))
        if only_keep_adverbs:
            assert isinstance(only_keep_adverbs, list), "Please provide list of strings to only_keep_adverbs."
            logger.info("Only keeping adverbs: %s" % ','.join(only_keep_adverbs))
        if module == "full":
            old_save_directory = save_directory
            save_directory = output_folder_pattern % "position"
        super().__init__(path_to_data, save_directory, k, split, input_vocabulary_file,
                         target_vocabulary_file, generate_vocabulary, upsample_isolated=upsample_isolated,
                         isolate_adverb_types=isolate_adverb_types, only_keep_adverbs=only_keep_adverbs,
                         isolate_examples_with=isolate_examples_with, seed=seed)
        if not generate_vocabulary:
            self.adverb_vocabulary = AdverbVocabulary.load(os.path.join(save_directory,
                                                                        adverb_vocabulary_file))
            self.adverb_input_vocabulary = Vocabulary.load(os.path.join(save_directory,
                                                                        adverb_input_vocabulary_file))
            self.adverb_target_vocabulary = Vocabulary.load(os.path.join(save_directory,
                                                                         adverb_target_vocabulary_file))
            self.planner_target_vocabulary = Vocabulary.load(os.path.join(save_directory,
                                                                          planner_target_vocabulary_file))
            self.transitive_target_vocabulary = Vocabulary.load(os.path.join(save_directory,
                                                                             transitive_target_vocabulary_file))
            if module == "full":
                self.vocabulary_dict = {
                    "position": [(input_vocabulary_file, "input_tensor")],
                    "planner": [(input_vocabulary_file, "input_tensor"),
                                (planner_target_vocabulary_file, "planner_target_tensor")],
                    "interaction": [(input_vocabulary_file, "input_tensor"),
                                    (transitive_target_vocabulary_file, "transitive_target_tensor")],
                    "adverb_transform": [(input_vocabulary_file, "input_tensor"),
                                         (adverb_input_vocabulary_file, "adverb_input_tensor"),
                                         (adverb_target_vocabulary_file, "adverb_target_tensor")],
                    "alo_transform": [(adverb_target_vocabulary_file, "adverb_target_tensor"),
                                      (target_vocabulary_file, "target_tensor")]
                }
                self.full_vocabularies = self.load_full_vocabularies(
                    output_folder_pattern)
            logger.info("Also done loading adverb vocabulary.")
        if module == "full":
            save_directory = old_save_directory
        self.transitive_verbs = self.dataset._vocabulary.get_transitive_verbs()
        if isolate_adverb_types:
            logger.info("Isolated %d examples for adverb types." % len(self.dataset._data_pairs["isolated_adverb_types"]))
        logger.info("Occurrences of adverb types:")
        for adverb_type, count in self.dataset._data_statistics["train"]["type_adverbs"].items():
            logger.info("{}: {} occurrences.".format(adverb_type, count))
        self.adverb_vocabulary_size = self.adverb_vocabulary.size
        self._percentage_not_unk = {
            "input": {"total": 0, "not_unk": 0},
            "target": {"total": 0, "not_unk": 0},
            "adverb": {"total": 0, "not_unk": 0},
        }

    def sentence_to_array(self, sentence: List[str], vocabulary: str, module="") -> List[int]:
        """
        Convert each string word in a sentence to the corresponding integer from the vocabulary and append
        a start-of-sequence and end-of-sequence token.
        :param sentence: the sentence in words (strings)
        :param vocabulary: whether to use the input or target vocabulary.
        :param module: if specified, get the module-specific vocabulary
        :return: the sentence in integers.
        """
        vocab = self.get_vocabulary(vocabulary, module)
        sentence_array = [vocab.sos_idx]
        for word in sentence:
            sentence_array.append(vocab.word_to_idx(word))
        sentence_array.append(vocab.eos_idx)
        return sentence_array

    def array_to_sentence(self, sentence_array: List[int], vocabulary: str, module="") -> List[str]:
        """
        Translate each integer in a sentence array to the corresponding word.
        :param sentence_array: array with integers representing words from the vocabulary.
        :param vocabulary: whether to use the input or target vocabulary.
        :param module: if specified, get the module-specific vocabulary
        :return: the sentence in words.
        """
        vocab = self.get_vocabulary(vocabulary, module)
        return [vocab.idx_to_word(word_idx) for word_idx in sentence_array]

    def load_full_vocabularies(self, output_folder_pattern: str):
        full_vocabularies = {}
        for module, vocabs in self.vocabulary_dict.items():
            module_folder_path = output_folder_pattern % module
            full_vocabularies[module] = {}
            for vocab, tensor_name in vocabs:
                vocab_file = os.path.join(module_folder_path, vocab)
                vocabulary = Vocabulary.load(vocab_file)
                full_vocabularies[module][tensor_name] = vocabulary
                vocab_attribute = tensor_name.split("_tensor")[0] + "_vocabulary"
                modular_vocab_attribute = module + "_" + vocab_attribute
                setattr(self, vocab_attribute, vocabulary)
                setattr(self, modular_vocab_attribute, vocabulary)
        return full_vocabularies

    def read_vocabularies(self) -> {}:
        """
        Loop over all examples in the dataset and add the words in them to the vocabularies.
        """
        logger.info("Populating vocabulary...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split,
                                                                         simple_situation_representation=True,
                                                                         adverb_inputs=True)):
            self.adverb_input_vocabulary.add_sentence(example["adverb_input"])
            if example["verb_in_command"] == "push" or example["verb_in_command"] == "pull":
                planner_target = [command for command in example["adverb_input"] if command != example["verb_in_command"]]
                interactions = [command for command in example["adverb_input"] if command == example["verb_in_command"]]
            else:
                interactions = []
                planner_target = example["adverb_input"]
            self.planner_target_vocabulary.add_sentence(planner_target)
            self.adverb_vocabulary.add_word(example["adverb"])
            self.transitive_target_vocabulary.add_sentence(interactions)
            if self.collapse_alo:
                self.adverb_target_vocabulary.add_sentence(example["target_command"])
            else:
                self.adverb_target_vocabulary.add_sentence(example["adverb_target"])
            self.input_vocabulary.add_sentence(example["input_command"])
            self.target_vocabulary.add_sentence(example["target_command"])

    def get_vocabulary(self, vocabulary: str, module="") -> Vocabulary:
        vocabulary_attribute = vocabulary + "_vocabulary"
        if vocabulary_attribute == "final_input_vocabulary":
            vocabulary_attribute = "adverb_target_vocabulary"
        if module:
            vocabulary_attribute = module + "_" + vocabulary_attribute
        if hasattr(self, vocabulary_attribute):
            return getattr(self, vocabulary_attribute)
        else:
            raise ValueError("Specified unknown vocabulary in sentence_to_array: {}".format(vocabulary))

    def save_vocabularies(self, input_vocabulary_file: str, target_vocabulary_file: str,
                          adverb_input_vocabulary_file="", adverb_target_vocabulary_file="",
                          adverb_vocabulary_file="", planner_target_vocabulary_file="",
                          transitive_target_vocabulary_file=""):
        self.adverb_input_vocabulary.save(os.path.join(self.directory, adverb_input_vocabulary_file))
        self.adverb_target_vocabulary.save(os.path.join(self.directory, adverb_target_vocabulary_file))
        self.input_vocabulary.save(os.path.join(self.directory, input_vocabulary_file))
        self.target_vocabulary.save(os.path.join(self.directory, target_vocabulary_file))
        self.adverb_vocabulary.save(os.path.join(self.directory, adverb_vocabulary_file))
        self.planner_target_vocabulary.save(os.path.join(self.directory, planner_target_vocabulary_file))
        self.transitive_target_vocabulary.save(os.path.join(self.directory, transitive_target_vocabulary_file))

    def get_percentage_unk(self, vocabulary: str):
        if vocabulary not in self._percentage_not_unk:
            raise ValueError("Unknown vocabulary %s passed to get_percentage_unk(). "
                             "Options are 'input', 'target', 'adverb'." % vocabulary)
        return -1.
            #return 1. - (self._percentage_not_unk[vocabulary]["not_unk"] / self._percentage_not_unk[vocabulary]["total"])

    def remove_transitive_verbs(self, input_sequence: List[str]) -> List[str]:
        removed_transitive = [word for word in input_sequence if word not in self.transitive_verbs]
        return removed_transitive

    def convert_example_for_modularity(self, example):
        original_adverb_input = example["adverb_input"]
        original_adverb_target = example["adverb_target"]
        verb_in_command = example["verb_in_command"]
        planner_target = original_adverb_input.copy()
        adverb_target = original_adverb_target.copy()
        if verb_in_command == "push" or verb_in_command == "pull":
            planner_target = [command for command in planner_target if command != verb_in_command]
            interactions = [command for command in example["target_command"] if command == verb_in_command]
        else:
            interactions = []
        interaction_target = []
        adverb_input = original_adverb_input.copy()
        if verb_in_command not in interaction_target:
            interaction_target += interactions
        if verb_in_command not in adverb_input:
            adverb_input += interactions
        if example["type_adverb"] == "movement_rewrite" or example["type_adverb"] == "movement":
            planner_target = [command for command in original_adverb_target if command != verb_in_command]
            interaction_target = interactions
            adverb_input = original_adverb_target.copy() + interactions
            adverb_target = original_adverb_target.copy() + interactions
        if not example["adverb"]:
            planner_target = [command for command in original_adverb_target if command not in ["push", "pull"]]
            adverb_input = original_adverb_target
        example["adverb_input"] = adverb_input
        if self.collapse_alo:
            example["adverb_target"] = example["target_command"]
        else:
            example["adverb_target"] = adverb_target
        example["planner_target"] = planner_target
        example["interaction_target"] = interaction_target
        return example

    def make_dummy_sequence_planner_target(self, example, sequence=True):
        direction_to_target = example["situation_representation"]['direction_to_target']
        if direction_to_target in ['n', 'e', 's', 'w']:
            if example["adverb"] in ["hesitantly", "cautiously"]:
                direction_translation = {
                    "n": "turn left",
                    "e": "walk",
                    "s": "turn right",
                    "w": "turn left"
                }
            else:
                direction_translation = {
                    "n": "north",
                    "e": "east",
                    "s": "south",
                    "w": "west"
                }
            dummy_planner_target = direction_translation[direction_to_target]
        elif example["adverb"] == "while zigzagging":
            direction_translation = {
                "ne": "east",
                "sw": "west",
                "se": "east",
                "nw": "west"
            }
            dummy_planner_target = direction_translation[direction_to_target]
        else:
            if example["adverb"] in ["hesitantly", "cautiously"]:
                direction_translation = {
                    "ne": "walk",
                    "sw": "turn right",
                    "se": "turn right",
                    "nw": "turn left"
                }
            else:
                direction_translation = {
                    "ne": "north",
                    "sw": "south",
                    "se": "south",
                    "nw": "north"
                }
            dummy_planner_target = direction_translation[direction_to_target]
        if sequence:
            dummy_planner_target = [dummy_planner_target]
        else:
            int_translate = {
                "north": 0,
                "east": 1,
                "south": 2,
                "west": 3,
                "turn left": 4,
                "turn right": 5,
                "walk": 6
            }
            dummy_planner_target = int_translate[dummy_planner_target]
        return dummy_planner_target

    def get_data_iterator(self, input_keys=None, input_keys_to_pad=None, target_keys=None, target_keys_to_pad=None,
                          batch_size=None, max_examples=None, simple_situation_representation=True, shuffle=False,
                          model_to_yield="full", modular=True) -> {}:
        """
        Loop over the data examples in GroundedScan and convert them to tensors, also save the lengths
        for input and target sequences that are needed for padding.
        :param input_keys:
        :param input_keys_to_pad:
        :param target_keys:
        :param target_keys_to_pad:
        :param batch_size
        :param max_examples: how many examples to read maximally, read all if None.
        :param simple_situation_representation: whether to read the full situation image in RGB or the simplified
        :param shuffle:
        :param model_to_yield:
        :param modular:
        smaller representation.
        """
        assert isinstance(batch_size, int), "Provide a batch size."
        assert model_to_yield in ["full"] + self.all_modules

        logger.info("Converting dataset to tensors...")
        current_examples_batch = np.array([])
        current_lengths_batch = {
            "adverb_input_tensor_lengths": np.array([]),
            "adverb_target_tensor_lengths": np.array([]),
            "input_tensor_lengths": np.array([]),
            "target_tensor_lengths": np.array([]),
            "transitive_input_tensor_lengths": np.array([]),
            "transitive_target_tensor_lengths": np.array([]),
            "planner_target_tensor_lengths": np.array([]),
            "dummy_planner_target_tensor_lengths": np.array([]),
            "final_input_tensor_lengths": np.array([])
        }
        for i, example in enumerate(self.dataset.get_examples_with_image(
                self.split, shuffle=shuffle, simple_situation_representation=simple_situation_representation,
                adverb_inputs=modular)):
            if max_examples:
                if len(self._examples) > max_examples:
                    return
            empty_example = {}

            example = self.convert_example_for_modularity(example)
            if modular:
                #  Adverb model input and target TODO: if no adverb?
                adverb_input = example["adverb_input"]
                if (example["type_adverb"] == "movement_rewrite" or example["type_adverb"] == "movement") and not self.collapse_alo:
                    adverb_target = []
                else:
                    adverb_target = example["adverb_target"]
                if not example["adverb"]:
                    adverb_target = []

            final_input = example["adverb_target"]

            # Position model input and target
            input_commands = example["input_command"]
            situation_image = example["situation_image"]
            grid_size = example["situation_representation"]["grid_size"]
            agent_position = example["situation_representation"]["agent_position"]
            agent_direction = example["situation_representation"]["agent_direction"]
            agent_position_representation = (int(agent_position["row"]) * grid_size) + int(agent_position["column"])
            target_position = example["situation_representation"]["target_object"]["position"]
            target_position_representation = (int(target_position["row"]) * grid_size) + int(target_position["column"])

            if modular:
                # Planner model input=positions/direction and target
                # planner_target = self.remove_transitive_verbs(adverb_input)
                planner_target = example["planner_target"]
                # Transitive model input=planner_target and target
                transitive_target = example["interaction_target"] if example["verb_in_command"] in self.transitive_verbs \
                    else []
                dummy_planner_target = self.make_dummy_sequence_planner_target(example, sequence=False)
                dummy_sequence_planner_target = self.make_dummy_sequence_planner_target(example, sequence=True)
                # dummy_sequence_planner_target = example["planner_target"][0:1]
                if example["verb_in_command"] == "pull":
                    dummy_interaction_target = transitive_target
                elif example["verb_in_command"] == "push":
                    dummy_interaction_target = transitive_target
                else:
                    dummy_interaction_target = transitive_target

            # Final input=adverb_target and target
            target_commands = example["target_command"]
            if example["adverb"] == "cautiously" or example["adverb"] == "hesitantly":
                target_commands = []
            adverb_embedding = self.adverb_vocabulary.word_to_idx(example["adverb"])

            example_information = {
                "command": example["input_command"],
                "adverb": example["adverb"] if modular else None,
                "type_adverb": example["type_adverb"] if modular else None,
                "adverb_input": adverb_input if modular else None,
                "adverb_target": adverb_target if modular else None,
                "final_input": final_input if modular else None,
                "target_command": example["target_command"],
                "verb_in_command": example["verb_in_command"] if modular else None,
                "derivation_representation": example["derivation_representation"],
                "situation_representation": example["situation_representation"]
            }

            # Position model sequence
            input_array = self.sentence_to_array(input_commands, vocabulary="input")

            # Final model sequence
            target_array = self.sentence_to_array(target_commands, vocabulary="target")

            if modular:
                # Planner model sequence
                planner_target_array = self.sentence_to_array(planner_target, vocabulary="planner_target")
                dummy_planner_target_array = self.sentence_to_array(dummy_sequence_planner_target,
                                                              vocabulary="planner_target")

                transitive_input_array = self.sentence_to_array(planner_target, vocabulary="planner_target")

                # Transitive model sequence
                transitive_target_array = self.sentence_to_array(transitive_target, vocabulary="transitive_target")
                dummy_interaction_target_array = self.sentence_to_array(dummy_interaction_target,
                                                                        vocabulary="transitive_target")

                # Adverb model sequences
                adverb_state = self.adverb_vocabulary.word_to_idx(example["adverb"])
                adverb_input_array = self.sentence_to_array(adverb_input, vocabulary="adverb_input")
                adverb_target_array = self.sentence_to_array(adverb_target, vocabulary="adverb_target")

                final_input_array = self.sentence_to_array(final_input, vocabulary="adverb_target")

                num_not_unk_input = sum([int(self.get_vocabulary("input").contains_word(word))
                                         for word in adverb_input])
                self._percentage_not_unk["input"]["total"] += len(adverb_input)
                self._percentage_not_unk["input"]["not_unk"] += num_not_unk_input
                num_not_unk_adverb = int(self.get_vocabulary("adverb").contains_word(
                    example["adverb"]))
                self._percentage_not_unk["adverb"]["total"] += 1
                self._percentage_not_unk["adverb"]["not_unk"] += num_not_unk_adverb
                num_not_unk_target = sum([int(self.get_vocabulary("adverb_target").contains_word(word))
                                      for word in adverb_target])
                self._percentage_not_unk["target"]["total"] += len(adverb_target)
                self._percentage_not_unk["target"]["not_unk"] += num_not_unk_target
                # Planner model tensor
                empty_example["planner_target_tensor"] = torch.tensor(planner_target_array, dtype=torch.long,
                                                                      device=device).unsqueeze(
                    dim=0)
                empty_example["dummy_planner_target_tensor"] = torch.tensor(dummy_planner_target_array, dtype=torch.long,
                                                                            device=device).unsqueeze(
                    dim=0)

                # Transitive model tensor
                empty_example["transitive_input_tensor"] = torch.tensor(transitive_input_array, dtype=torch.long,
                                                                        device=device).unsqueeze(dim=0)
                empty_example["transitive_target_tensor"] = torch.tensor(transitive_target_array, dtype=torch.long,
                                                                         device=device).unsqueeze(dim=0)

                # Adverb model tensors
                empty_example["adverb_input_tensor"] = torch.tensor(adverb_input_array, dtype=torch.long,
                                                                    device=device).unsqueeze(
                    dim=0)
                empty_example["adverb_state"] = torch.tensor(adverb_state, dtype=torch.long, device=device).unsqueeze(
                    dim=0)
                empty_example["adverb_target_tensor"] = torch.tensor(adverb_target_array, dtype=torch.long,
                                                                     device=device).unsqueeze(
                    dim=0)
                empty_example["final_input_tensor"] = torch.tensor(final_input_array, dtype=torch.long,
                                                                   device=device).unsqueeze(dim=0)
                current_lengths_batch["adverb_input_tensor_lengths"] = np.append(current_lengths_batch["adverb_input_tensor_lengths"],
                                                                          [len(adverb_input_array)])
                current_lengths_batch["adverb_target_tensor_lengths"] = np.append(
                    current_lengths_batch["adverb_target_tensor_lengths"], [len(adverb_target_array)])
                current_lengths_batch["planner_target_tensor_lengths"] = np.append(
                    current_lengths_batch["planner_target_tensor_lengths"], [len(planner_target_array)])
                current_lengths_batch["dummy_planner_target_tensor_lengths"] = np.append(
                    current_lengths_batch["dummy_planner_target_tensor_lengths"], [len(dummy_planner_target_array)])
                current_lengths_batch["transitive_input_tensor_lengths"] = np.append(
                    current_lengths_batch["transitive_input_tensor_lengths"], [len(transitive_input_array)])
                current_lengths_batch["transitive_target_tensor_lengths"] = np.append(
                    current_lengths_batch["transitive_target_tensor_lengths"], [len(transitive_target_array)])
                current_lengths_batch["final_input_tensor_lengths"] = np.append(
                    current_lengths_batch["final_input_tensor_lengths"], [len(final_input_array)])
                empty_example["dummy_planner_target"] = torch.tensor(dummy_planner_target, dtype=torch.long,
                                                                     device=device).unsqueeze(dim=0)
                empty_example["dummy_interaction_target"] = torch.tensor(dummy_interaction_target_array, dtype=torch.long,
                                                                         device=device).unsqueeze(dim=0)

            # Position model tensors
            empty_example["input_tensor"] = torch.tensor(input_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)
            empty_example["world_state_tensor"] = torch.tensor(situation_image, dtype=torch.float,
                                                               device=device).unsqueeze(dim=0)
            empty_example["agent_position"] = torch.tensor(agent_position_representation, dtype=torch.long,
                                                           device=device).unsqueeze(dim=0)
            empty_example["target_position"] = torch.tensor(target_position_representation, dtype=torch.long,
                                                            device=device).unsqueeze(dim=0)
            empty_example["agent_direction"] = torch.tensor(agent_direction, dtype=torch.long,
                                                            device=device).unsqueeze(dim=0)
            empty_example["adverb_embedding"] = torch.tensor(adverb_embedding, dtype=torch.long,
                                                             device=device).unsqueeze(dim=0)

            # Final model tensor
            empty_example["target_tensor"] = torch.tensor(target_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)

            empty_example["example_information"] = example_information

            current_lengths_batch["input_tensor_lengths"] = np.append(
                current_lengths_batch["input_tensor_lengths"], [len(input_array)])
            current_lengths_batch["target_tensor_lengths"] = np.append(
                current_lengths_batch["target_tensor_lengths"], [len(target_array)])

            current_examples_batch = np.append(current_examples_batch, [empty_example])
            if len(current_examples_batch) == batch_size:
                if modular:
                    if model_to_yield != "full":
                        input_keys_to_pad_d = {input_key_to_pad: current_lengths_batch[input_key_to_pad + "_lengths"]
                                               for input_key_to_pad in input_keys_to_pad}
                        target_keys_to_pad_d = {target_key_to_pad: current_lengths_batch[target_key_to_pad + "_lengths"]
                                                for target_key_to_pad in target_keys_to_pad}
                        yield self.make_modular_batch(current_examples_batch, input_keys, input_keys_to_pad_d,
                                                      target_keys, target_keys_to_pad_d)
                    else:
                        batches = {}
                        for model_name in input_keys_to_pad.keys():
                            input_keys_to_pad_d = {
                                input_key_to_pad: current_lengths_batch[input_key_to_pad + "_lengths"]
                                for input_key_to_pad in input_keys_to_pad[model_name]}
                            target_keys_to_pad_d = {
                                target_key_to_pad: current_lengths_batch[target_key_to_pad + "_lengths"]
                                for target_key_to_pad in target_keys_to_pad[model_name]}
                            batch = self.make_modular_batch(current_examples_batch, input_keys[model_name],
                                                            input_keys_to_pad_d, target_keys[model_name],
                                                            target_keys_to_pad_d)
                            batches[model_name] = batch
                        yield batches
                else:
                    raise NotImplementedError("model_to_yield=%s not implemented." % model_to_yield)
                current_examples_batch = np.array([])
                current_lengths_batch = {
                    "adverb_input_tensor_lengths": np.array([]),
                    "adverb_target_tensor_lengths": np.array([]),
                    "input_tensor_lengths": np.array([]),
                    "target_tensor_lengths": np.array([]),
                    "transitive_input_tensor_lengths": np.array([]),
                    "transitive_target_tensor_lengths": np.array([]),
                    "planner_target_tensor_lengths": np.array([]),
                    "dummy_planner_target_tensor_lengths": np.array([]),
                    "final_input_tensor_lengths": np.array([])
                }

    def make_modular_batch(self, examples: Dict[str, torch.Tensor], input_keys: List[str],
                           input_keys_to_pad: Dict[str, np.ndarray], target_keys: List[str],
                           target_keys_to_pad: Dict[str, np.ndarray]):
        """
        Prepare a batch of inputs and targets for module=position.
        :param examples: a dict with inputs in tensor form and other information about the examples
        :param input_keys:
        :param input_keys_to_pad:
        :param target_keys:
        :param target_keys_to_pad:
        :return: a batch with inputs, targets, and extra information about the examples (needed for writing predictions)
        """
        all_input_keys = input_keys + list(input_keys_to_pad.keys())
        all_target_keys = target_keys + list(target_keys_to_pad.keys())
        inputs = {key: [] for key in all_input_keys}
        targets = {key + "_targets": [] for key in all_target_keys}
        extra_information_batch = []
        for example in examples:
            for key in input_keys_to_pad:
                max_input_length = np.max(input_keys_to_pad[key])
                to_pad_input = max_input_length - example[key].size(1)
                padded_input = torch.cat([
                    example[key],
                    torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                inputs[key].append(padded_input)
            for key in input_keys:
                inputs[key].append(example[key])
            for key in target_keys_to_pad:
                max_target_length = np.max(target_keys_to_pad[key])
                to_pad_target = max_target_length - example[key].size(1)
                padded_target = torch.cat([
                    example[key],
                    torch.zeros(int(to_pad_target), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                targets[key + "_targets"].append(padded_target)
            for key in target_keys:
                targets[key + "_targets"].append(example[key])
            extra_information_batch.append(example)

        for input_key, input_values in inputs.items():
            input_batch = torch.cat(input_values, dim=0)
            inputs[input_key] = input_batch
        for input_key, input_lengths in input_keys_to_pad.items():
            inputs[input_key + "_lengths"] = input_lengths
        for target_key, target_values in targets.items():
            target_batch = torch.cat(target_values, dim=0)
            targets[target_key] = target_batch
        for target_key, target_lengths in target_keys_to_pad.items():
            targets[target_key + "_lengths"] = target_keys_to_pad[target_key]
        return {
            "inputs": inputs,
            "targets": targets,
            "batch_size": len(examples),
            "extra_information": extra_information_batch
        }

    def make_position_model_batch(self, examples, input_lengths):
        """
        Prepare a batch of inputs and targets for module=position.
        :param examples: a dict with inputs in tensor form and other information about the examples
        :param input_lengths: the lengths of the input commands
        :return: a batch with inputs, targets, and extra information about the examples (needed for writing predictions)
        """
        max_input_length = np.max(input_lengths)
        input_batch = []
        agent_position_batch = []
        world_state_batch = []
        target_position_batch = []
        start_direction_batch = []
        extra_information_batch = []
        for example in examples:
            to_pad_input = max_input_length - example["input_tensor"].size(1)
            padded_input = torch.cat([
                example["input_tensor"],
                torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
            input_batch.append(padded_input)
            world_state_batch.append(example["world_state_tensor"])
            agent_position_batch.append(example["agent_position"])
            target_position_batch.append(example["target_position"])
            start_direction_batch.append(example["agent_direction"])
            extra_information_batch.append(example)
        input_batch = torch.cat(input_batch, dim=0)
        world_state_batch = torch.cat(world_state_batch, dim=0)
        agent_position_batch = torch.cat(agent_position_batch, dim=0)
        target_position_batch = torch.cat(target_position_batch, dim=0)
        start_direction_batch = torch.cat(start_direction_batch, dim=0)
        return {
            "inputs": {
                "input_tensor": input_batch,
                "input_tensor_lengths": input_lengths,
                "world_state_tensor": world_state_batch,
            },
            "targets": {
                "agent_direction_targets": start_direction_batch,
                "agent_position_targets": agent_position_batch,
                "target_position_targets": target_position_batch
            },
            "extra_information": extra_information_batch
        }

    def make_planner_batch(self, examples, planner_target_lengths):
        """

        :param examples:
        :param planner_target_lengths:
        :return:
        """
        max_target_length = np.max(planner_target_lengths)
        target_batch = []
        agent_position_batch = []
        target_position_batch = []
        start_direction_batch = []
        extra_information_batch = []
        for example in examples:
            to_pad_target = max_target_length - example["planner_target_tensor"].size(1)
            padded_target = torch.cat([
                example["planner_target_tensor"],
                torch.zeros(int(to_pad_target), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
            target_batch.append(padded_target)
            agent_position_batch.append(example["agent_position"])
            target_position_batch.append(example["target_position"])
            start_direction_batch.append(example["agent_direction"])
            extra_information_batch.append(example)
        target_batch = torch.cat(target_batch, dim=0)
        agent_position_batch = torch.cat(agent_position_batch, dim=0)
        target_position_batch = torch.cat(target_position_batch, dim=0)
        start_direction_batch = torch.cat(start_direction_batch, dim=0)
        return {
            "inputs": {
                "agent_directions": start_direction_batch,
                "agent_positions": agent_position_batch,
                "target_positions": target_position_batch
            },
            "targets": {
                "planner_target": target_batch
            },
            "extra_information": extra_information_batch
        }

    def make_transitive_batch(self, examples, planner_target_lengths, adverb_input_tensor_lengths):
        """

        :param examples:
        :param planner_target_lengths:
        :param adverb_input_tensor_lengths:
        :return:
        """
        pass

    def make_batch(self, examples, input_lengths, target_lengths, adverb_input_tensor_lengths, adverb_target_tensor_lengths,
                   transitive_target_lengths, planner_target_lengths) -> Tuple[torch.Tensor, List[int],
                                                                               torch.Tensor, List[dict],
                                                                               torch.Tensor, List[int]]:
        """
        Iterate over batches of example tensors, pad them to the max length in the batch and yield.
        :param examples:
        :param input_lengths:
        :param target_lengths:
        :param adverb_input_tensor_lengths:
        :param adverb_target_tensor_lengths:
        :param transitive_target_lengths:
        :param planner_target_lengths:
        :return:
        """
        raise NotImplementedError("TODO: implement full batch")
        max_input_length = np.max(input_lengths)
        max_target_length = np.max(target_lengths)
        input_batch = []
        adverb_batch = []
        target_batch = []
        situation_representation_batch = []
        derivation_representation_batch = []
        original_input_batch = []
        original_output_batch = []
        verb_in_command_batch = []
        adverb_type_batch = []
        gscan_final_targets_batch = []
        for example in examples:
            to_pad_input = max_input_length - example["input_tensor"].size(1)
            to_pad_target = max_target_length - example["target_tensor"].size(1)
            padded_input = torch.cat([
                example["input_tensor"],
                torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
            padded_target = torch.cat([
                example["target_tensor"],
                torch.zeros(int(to_pad_target), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
            input_batch.append(padded_input)
            target_batch.append(padded_target)
            adverb_batch.append(example["adverb_input"])
            situation_representation_batch.append(example["example_information"]["situation_representation"])
            adverb_type_batch.append(example["example_information"]["type_adverb"])
            derivation_representation_batch.append(example["example_information"]["derivation_representation"])
            original_input_batch.append(example["example_information"]["original_input"])
            original_output_batch.append(example["example_information"]["original_output"])
            verb_in_command_batch.append(example["example_information"]["verb_in_command"])
            gscan_final_targets_batch.append(example["example_information"]["gscan_final_target"])
        return (torch.cat(input_batch, dim=0), input_lengths, derivation_representation_batch,
               torch.cat(adverb_batch, dim=0), situation_representation_batch, torch.cat(target_batch, dim=0),
               target_lengths, adverb_type_batch, original_input_batch, original_output_batch,
               verb_in_command_batch, gscan_final_targets_batch)

