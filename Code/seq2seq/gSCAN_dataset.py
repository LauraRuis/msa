import os
from typing import List
from typing import Tuple
import logging
from collections import defaultdict
from collections import Counter
import json
import torch
import numpy as np

from GroundedScan.dataset import GroundedScan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class Vocabulary(object):
    """
    Object that maps words in string form to indices to be processed by numerical models.
    """

    def __init__(self, sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>"):
        """
        NB: <PAD> token is by construction idx 0.
        """
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self._idx_to_word = [pad_token, sos_token, eos_token]
        self._word_to_idx = defaultdict(lambda: self._idx_to_word.index(self.pad_token))
        self._word_to_idx[sos_token] = 1
        self._word_to_idx[eos_token] = 2
        self._word_frequencies = Counter()

    def word_to_idx(self, word: str) -> int:
        return self._word_to_idx[word]

    def idx_to_word(self, idx: int) -> str:
        return self._idx_to_word[idx]

    def contains_word(self, word: str) -> bool:
        if self._word_to_idx[word] != 0:
            return True
        else:
            return False

    def add_sentence(self, sentence: List[str]):
        for word in sentence:
            if word not in self._word_to_idx:
                self._word_to_idx[word] = self.size
                self._idx_to_word.append(word)
            self._word_frequencies[word] += 1

    def most_common(self, n=10):
        return self._word_frequencies.most_common(n=n)

    @property
    def pad_idx(self):
        return self.word_to_idx(self.pad_token)

    @property
    def sos_idx(self):
        return self.word_to_idx(self.sos_token)

    @property
    def eos_idx(self):
        return self.word_to_idx(self.eos_token)

    @property
    def size(self):
        return len(self._idx_to_word)

    @classmethod
    def load(cls, path: str):
        assert os.path.exists(path), "Trying to load a vocabulary from a non-existing file {}".format(path)
        with open(path, 'r') as infile:
            all_data = json.load(infile)
            sos_token = all_data["sos_token"]
            eos_token = all_data["eos_token"]
            pad_token = all_data["pad_token"]
            vocab = cls(sos_token=sos_token, eos_token=eos_token, pad_token=pad_token)
            vocab._idx_to_word = all_data["idx_to_word"]
            vocab._word_to_idx = defaultdict(int)
            for word, idx in all_data["word_to_idx"].items():
                vocab._word_to_idx[word] = idx
            vocab._word_frequencies = Counter(all_data["word_frequencies"])
        return vocab

    def to_dict(self) -> dict:
        return {
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "idx_to_word": self._idx_to_word,
            "word_to_idx": self._word_to_idx,
            "word_frequencies": self._word_frequencies
        }

    def save(self, path: str) -> str:
        with open(path, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)
        return path


class GroundedScanDataset(object):
    """
    Loads a GroundedScan instance from a specified location.
    """

    def __init__(self, path_to_data: str, save_directory: str, k: int, upsample_isolated=100, split="train",
                 input_vocabulary_file="", target_vocabulary_file="", generate_vocabulary=False,
                 isolate_examples_with="cautiously", simplified_objective=False):
        assert os.path.exists(path_to_data), "Trying to read a gSCAN dataset from a non-existing file {}.".format(
            path_to_data)
        self.simplified_objective = simplified_objective
        assert not simplified_objective, "Simplified objective for debugging purposes only."
        if not generate_vocabulary:
            assert os.path.exists(os.path.join(save_directory, input_vocabulary_file)) and os.path.exists(
                os.path.join(save_directory, target_vocabulary_file)), \
                "Trying to load vocabularies from non-existing files."
        if split == "test" and generate_vocabulary:
            logger.warning("WARNING: generating a vocabulary from the test set.")
        self.dataset = GroundedScan.load_dataset_from_file(path_to_data, save_directory=save_directory, k=k,
                                                           upsample_isolated=upsample_isolated,
                                                           isolate_examples_with=isolate_examples_with)
        if self.dataset._data_statistics.get("adverb_1"):
            logger.info("Verb-adverb combinations in training set: ")
            for adverb, items in self.dataset._data_statistics["train"]["verb_adverb_combinations"].items():
                logger.info("Verbs for adverb: {}".format(adverb))
                for key, count in items.items():
                    logger.info("   {}: {} occurrences.".format(key, count))
            logger.info("Verb-adverb combinations in dev set: ")
            for adverb, items in self.dataset._data_statistics["dev"]["verb_adverb_combinations"].items():
                logger.info("Verbs for adverb: {}".format(adverb))
                for key, count in items.items():
                    logger.info("   {}: {} occurrences.".format(key, count))
        actual_k = self.dataset._data_statistics["train"]["manners_in_command"][isolate_examples_with]
        expected_k = k * upsample_isolated
        if split in ["train", "dev"]:
            assert actual_k == expected_k, \
                "Chose k=%d and upsample=%d (expected k=%d) but actual number of examples with %s in training set is %d." % (
                    k, upsample_isolated, expected_k, isolate_examples_with, actual_k
                )
        self.image_dimensions = None
        self.image_channels = 16
        self.split = split
        self.directory = save_directory

        # Keeping track of data.
        self._examples = np.array([])
        self._input_lengths = np.array([])
        self._target_lengths = np.array([])
        if generate_vocabulary:
            logger.info("Generating vocabularies...")
            self.input_vocabulary = Vocabulary()
            self.target_vocabulary = Vocabulary()
            self.read_vocabularies()
            logger.info("Done generating vocabularies.")
        else:
            logger.info("Loading vocabularies...")
            self.input_vocabulary = Vocabulary.load(os.path.join(save_directory, input_vocabulary_file))
            self.target_vocabulary = Vocabulary.load(os.path.join(save_directory, target_vocabulary_file))
            logger.info("Done loading vocabularies.")

    def convert_target_to_simple(self, example):
        verb_in_command = example["input_command"][0]
        adverb_in_command = example["input_command"][-1]
        if adverb_in_command not in ["while spinning", "while zigzagging", "cautiously", "hesitantly"]:
            adverb_in_command = ""
        if verb_in_command == "push" or verb_in_command == "pull":
            interactions = [command for command in example["target_command"] if command == verb_in_command]
        else:
            interactions = []
        interaction_target = []
        if verb_in_command not in interaction_target:
            interaction_target += interactions
        if adverb_in_command == "while zigzagging":
            interaction_target = interactions
        return interaction_target

    def read_vocabularies(self) -> {}:
        """
        Loop over all examples in the dataset and add the words in them to the vocabularies.
        """
        logger.info("Populating vocabulary...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split,
                                                                         simple_situation_representation=True)):
            self.input_vocabulary.add_sentence(example["input_command"])
            if not self.simplified_objective:
                self.target_vocabulary.add_sentence(example["target_command"])
            else:
                interaction_target = self.convert_target_to_simple(example)
                self.target_vocabulary.add_sentence(interaction_target)

    def save_vocabularies(self, input_vocabulary_file: str, target_vocabulary_file: str):
        self.input_vocabulary.save(os.path.join(self.directory, input_vocabulary_file))
        self.target_vocabulary.save(os.path.join(self.directory, target_vocabulary_file))

    def get_vocabulary(self, vocabulary: str) -> Vocabulary:
        if vocabulary == "input":
            vocab = self.input_vocabulary
        elif vocabulary == "target":
            vocab = self.target_vocabulary
        else:
            raise ValueError("Specified unknown vocabulary in sentence_to_array: {}".format(vocabulary))
        return vocab

    def shuffle_data(self) -> {}:
        """
        Reorder the data examples and reorder the lengths of the input and target commands accordingly.
        """
        random_permutation = np.random.permutation(len(self._examples))
        self._examples = self._examples[random_permutation]
        self._target_lengths = self._target_lengths[random_permutation]
        self._input_lengths = self._input_lengths[random_permutation]

    def get_data_iterator(self, batch_size=None, max_examples=None,
                          simple_situation_representation=True, shuffle=False) -> {}:
        """
        Loop over the data examples in GroundedScan and convert them to tensors, also save the lengths
        for input and target sequences that are needed for padding.
        :param batch_size
        :param max_examples: how many examples to read maximally, read all if None.
        :param simple_situation_representation: whether to read the full situation image in RGB or the simplified
        :param shuffle:
        smaller representation.
        """
        assert isinstance(batch_size, int), "Provide a batch size."
        logger.info("Converting dataset to tensors...")
        current_examples_batch = np.array([])
        current_input_lengths = np.array([])
        current_target_lengths = np.array([])
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split,
                                                                         shuffle=shuffle,
                                                                         simple_situation_representation=simple_situation_representation,
                                                                         adverb_inputs=False)):
            if max_examples:
                if len(self._examples) > max_examples:
                    return
            empty_example = {}
            input_commands = example["input_command"]
            if not self.simplified_objective:
                target_commands = example["target_command"]
            else:
                target_commands = self.convert_target_to_simple(example)

            example_information = {
                # "adverb": example["adverb"],
                # "type_adverb": example["type_adverb"],
                "original_input": input_commands,
                "original_output": target_commands,
                "gscan_final_target": example["target_command"],
                # "verb_in_command": example["verb_in_command"],
                "derivation_representation": example["derivation_representation"],
                "situation_representation": example["situation_representation"]
            }
            input_array = self.sentence_to_array(input_commands, vocabulary="input")
            target_array = self.sentence_to_array(target_commands, vocabulary="target")
            empty_example["input_tensor"] = torch.tensor(input_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)
            empty_example["target_tensor"] = torch.tensor(target_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)
            empty_example["situation_image"] = torch.tensor(example["situation_image"],
                                                            dtype=torch.float, device=device).unsqueeze(dim=0)
            empty_example["example_information"] = example_information
            current_input_lengths = np.append(current_input_lengths, [len(input_array)])
            current_target_lengths = np.append(current_target_lengths, [len(target_array)])
            current_examples_batch = np.append(current_examples_batch, [empty_example])
            if len(current_examples_batch) == batch_size:
                yield self.make_batch(current_examples_batch, current_input_lengths, current_target_lengths)
                current_examples_batch = np.array([])
                current_input_lengths = np.array([])
                current_target_lengths = np.array([])

    def make_batch(self, examples, input_lengths, target_lengths) -> Tuple[torch.Tensor, List[int],
                                                                           torch.Tensor, List[dict],
                                                                           torch.Tensor, List[int]]:
        """
        Iterate over batches of example tensors, pad them to the max length in the batch and yield.
        :param batch_size: how many examples to put in each batch.
        :return: tuple of input commands batch, corresponding input lengths, adverb batch,
         target commands batch and corresponding target lengths.
        """
        max_input_length = np.max(input_lengths)
        max_target_length = np.max(target_lengths)
        input_batch = []
        adverb_batch = []
        target_batch = []
        situation_representation_batch = []
        derivation_representation_batch = []
        agent_positions_batch = []
        target_positions_batch = []
        situation_batch = []
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
            # adverb_batch.append(example["adverb_input"])
            situation_repr = example["example_information"]["situation_representation"]
            situation_representation_batch.append(situation_repr)
            situation_batch.append(example["situation_image"])
            agent_position = torch.tensor(
                (int(situation_repr["agent_position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["agent_position"]["column"]), dtype=torch.long,
                device=device).unsqueeze(dim=0)
            agent_positions_batch.append(agent_position)
            target_position = torch.tensor(
                (int(situation_repr["target_object"]["position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["target_object"]["position"]["column"]),
                dtype=torch.long, device=device).unsqueeze(dim=0)
            target_positions_batch.append(target_position)
            # adverb_type_batch.append(example["example_information"]["type_adverb"])
            derivation_representation_batch.append(example["example_information"]["derivation_representation"])
            # original_input_batch.append(example["example_information"]["original_input"])
            # original_output_batch.append(example["example_information"]["original_output"])
            # verb_in_command_batch.append(example["example_information"]["verb_in_command"])
            # gscan_final_targets_batch.append(example["example_information"]["gscan_final_target"])
        return (torch.cat(input_batch, dim=0), input_lengths, derivation_representation_batch,
               torch.cat(situation_batch, dim=0), situation_representation_batch, torch.cat(target_batch, dim=0),
               target_lengths, torch.cat(agent_positions_batch, dim=0), torch.cat(target_positions_batch, dim=0))

    def read_dataset(self, max_examples=None, simple_situation_representation=True) -> {}:
        """
        Loop over the data examples in GroundedScan and convert them to tensors, also save the lengths
        for input and target sequences that are needed for padding.
        :param max_examples: how many examples to read maximally, read all if None.
        :param simple_situation_representation: whether to read the full situation image in RGB or the simplified
        smaller representation.
        """
        logger.info("Converting dataset to tensors...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split, simple_situation_representation)):
            if max_examples:
                if len(self._examples) > max_examples:
                    return
            empty_example = {}
            input_commands = example["input_command"]
            target_commands = example["target_command"]
            #equivalent_target_commands = example["equivalent_target_command"]
            situation_image = example["situation_image"]
            if i == 0:
                self.image_dimensions = situation_image.shape[0]
                self.image_channels = situation_image.shape[-1]
            situation_repr = example["situation_representation"]
            input_array = self.sentence_to_array(input_commands, vocabulary="input")
            target_array = self.sentence_to_array(target_commands, vocabulary="target")
            #equivalent_target_array = self.sentence_to_array(equivalent_target_commands, vocabulary="target")
            empty_example["input_tensor"] = torch.tensor(input_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)
            empty_example["target_tensor"] = torch.tensor(target_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)
            #empty_example["equivalent_target_tensor"] = torch.tensor(equivalent_target_array, dtype=torch.long,
            #                                                         device=device).unsqueeze(dim=0)
            empty_example["situation_tensor"] = torch.tensor(situation_image, dtype=torch.float, device=device
                                                             ).unsqueeze(dim=0)
            empty_example["situation_representation"] = situation_repr
            empty_example["derivation_representation"] = example["derivation_representation"]
            empty_example["agent_position"] = torch.tensor(
                (int(situation_repr["agent_position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["agent_position"]["column"]), dtype=torch.long,
                device=device).unsqueeze(dim=0)
            empty_example["target_position"] = torch.tensor(
                (int(situation_repr["target_object"]["position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["target_object"]["position"]["column"]),
                dtype=torch.long, device=device).unsqueeze(dim=0)
            self._input_lengths = np.append(self._input_lengths, [len(input_array)])
            self._target_lengths = np.append(self._target_lengths, [len(target_array)])
            self._examples = np.append(self._examples, [empty_example])

    def sentence_to_array(self, sentence: List[str], vocabulary: str) -> List[int]:
        """
        Convert each string word in a sentence to the corresponding integer from the vocabulary and append
        a start-of-sequence and end-of-sequence token.
        :param sentence: the sentence in words (strings)
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in integers.
        """
        vocab = self.get_vocabulary(vocabulary)
        sentence_array = [vocab.sos_idx]
        for word in sentence:
            sentence_array.append(vocab.word_to_idx(word))
        sentence_array.append(vocab.eos_idx)
        return sentence_array

    def array_to_sentence(self, sentence_array: List[int], vocabulary: str) -> List[str]:
        """
        Translate each integer in a sentence array to the corresponding word.
        :param sentence_array: array with integers representing words from the vocabulary.
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in words.
        """
        vocab = self.get_vocabulary(vocabulary)
        return [vocab.idx_to_word(word_idx) for word_idx in sentence_array]

    @property
    def num_examples(self):
        return len(self._examples)

    @property
    def input_vocabulary_size(self):
        return self.input_vocabulary.size

    @property
    def target_vocabulary_size(self):
        return self.target_vocabulary.size
