import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Tuple, Union
import os
import shutil
import copy

from seq2seq.cnn_model import ConvolutionalNet
from seq2seq.seq2seq_model import EncoderRNN
from Modularity.dataset import Vocabulary
from Modularity.nn import LuongDecoder, MLP, LearnedAttention, get_exact_match, BOW, BahdanauAttentionDecoderRNN, BahdanauDecoder
from Modularity.predict import predict_sequence, predict_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


class SuperModule(nn.Module):
    """
    Super class used to initialize any one of the modules used in the modular gSCAN network.
    """

    def __init__(self, input_specs: Dict[str, Dict[str, Union[int, bool]]],
                 target_specs: Dict[str, Dict[str, Union[str, int, bool]]], metric: str,
                 embedding_dimension: int, encoder_hidden_size: int, target_vocabulary: str,
                 num_encoder_layers: int, encoder_dropout_p: float, encoder_bidirectional: bool,
                 output_directory: str, main_input_key: str, input_vocabulary: str, decoder_hidden_size=0,
                 num_decoder_layers=0, decoder_dropout_p=0.,
                 cnn_kernel_size=0, cnn_dropout_p=0., cnn_hidden_num_channels=0,
                 grid_size=0, gpu_util=True, **kwargs):
        super(SuperModule, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.encoder_bidirectional = encoder_bidirectional
        self.num_decoder_layers = num_decoder_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_hidden_num_channels = cnn_hidden_num_channels
        self.grid_size = grid_size

        self.input_keys = []
        self.input_keys_to_pad = []
        self.full_model = False
        self.gpu_util = gpu_util
        self.target_vocabulary = target_vocabulary
        self.main_input_key = main_input_key
        self.input_vocabulary = input_vocabulary
        if gpu_util:
            self.friend_one = torch.ones(10000, device=device).unsqueeze(dim=1)
            self.friend_two = torch.ones(10000, device=device).unsqueeze(dim=0)

        encoder_modules = {}

        mlp_inputs = []
        for input_key, input_args in input_specs.items():
            if input_args["data_type"] == "grid":
                self.input_keys.append(input_key)
                self.grid_size = grid_size
                # Input: [batch_size, image_width, image_width, num_channels]
                # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
                world_state_encoder = ConvolutionalNet(num_channels=input_args["num_input_channels"],
                                                       cnn_kernel_size=cnn_kernel_size,
                                                       num_conv_channels=cnn_hidden_num_channels,
                                                       dropout_probability=cnn_dropout_p)
                encoder_modules[input_key] = world_state_encoder
            elif input_args["data_type"] == "int":
                self.input_keys.append(input_key)
                mlp_inputs.append(input_key)
            elif input_args["data_type"] == "sequence":
                self.input_keys_to_pad.append(input_key)
                # Input: [batch_size, max_input_length]
                # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
                encoder = EncoderRNN(input_size=input_args["input_vocabulary_size"],
                                     embedding_dim=embedding_dimension,
                                     rnn_input_size=embedding_dimension,
                                     hidden_size=encoder_hidden_size, num_layers=num_encoder_layers,
                                     dropout_probability=encoder_dropout_p,
                                     bidirectional=encoder_bidirectional,
                                     padding_idx=input_args["input_padding_idx"])
                encoder_modules[input_key] = encoder
            elif input_args["data_type"] == "bow":
                encoder = BOW(input_size=input_args["input_vocabulary_size"],
                              embedding_dim=encoder_hidden_size,
                              dropout_probability=encoder_dropout_p,
                              padding_idx=input_args["input_padding_idx"])
                self.input_keys_to_pad.append(input_key)
                encoder_modules[input_key] = encoder
            elif input_args["data_type"] == "embedding":
                encoder = nn.Embedding(input_args["input_vocabulary_size"],
                                       embedding_dimension,
                                       padding_idx=input_args["input_padding_idx"])
                self.input_keys.append(input_key)
                encoder_modules[input_key] = encoder
            else:
                raise ValueError("Unrecognized data_type=%s for input_key=%s in constructor class for SuperModule" % (
                    input_args["data_type"], input_key
                ))

        if len(mlp_inputs):
            input_to_mlp = nn.Linear(len(mlp_inputs), encoder_hidden_size)
            mlp = MLP(input_dim=encoder_hidden_size, hidden_dim=encoder_hidden_size,
                      dropout_p=encoder_dropout_p)
            encoder_modules["mlp"] = nn.Sequential(input_to_mlp, mlp)

        self.encoder_module_dict = nn.ModuleDict(encoder_modules)

        self.target_keys = []
        self.target_keys_to_pad = []
        self.sequence_decoder = False
        self.target_eos_idx = None
        self.target_pad_idx = None
        self.target_sos_idx = None

        decoder_modules = {}
        for target_key, target_args in target_specs.items():
            if target_args["output_type"] == "sequence":
                self.target_keys_to_pad.append(target_key)
                # Input: [batch_size, max_target_length], initial hidden: ([batch_size, hidden_size], [batch_size, hidden_size])
                # Input for attention: [batch_size, max_input_length, hidden_size],
                #                      [batch_size, image_width * image_width, hidden_size]
                if target_args["type_attention"] == "luong":
                    attention_decoder = LuongDecoder(hidden_size=decoder_hidden_size,
                                                     output_size=target_args["target_vocabulary_size"],
                                                     num_layers=num_decoder_layers,
                                                     dropout_probability=decoder_dropout_p,
                                                     padding_idx=target_args["target_pad_idx"],
                                                     attention=target_args["attention"],
                                                     conditional_attention=target_args["conditional_attention"])
                elif target_args["type_attention"] == "bahdanau":
                    # attention = LearnedAttention(key_size=encoder_hidden_size, query_size=decoder_hidden_size,
                    #                              hidden_size=decoder_hidden_size)
                    # if target_args["conditional_attention"]:
                    #     conditional_attention = LearnedAttention(key_size=cnn_hidden_num_channels * 3,
                    #                                              query_size=decoder_hidden_size,
                    #                                              hidden_size=decoder_hidden_size)
                    # else:
                    #     conditional_attention = None
                    # attention_decoder = BahdanauDecoder(hidden_size=decoder_hidden_size,
                    #                                     output_size=target_args["target_vocabulary_size"],
                    #                                     num_layers=num_decoder_layers,
                    #                                     dropout_probability=decoder_dropout_p,
                    #                                     padding_idx=target_args["target_pad_idx"],
                    #                                     key_size=encoder_hidden_size,
                    #                                     attention=attention,
                    #                                     conditional_attention=conditional_attention)
                    # Attention over the output features of the ConvolutionalNet.
                    # Input: [bsz, 1, decoder_hidden_size], [bsz, image_width * image_width, cnn_hidden_num_channels * 3]
                    # Output: [bsz, 1, decoder_hidden_size], [bsz, 1, image_width * image_width]
                    if target_args["conditional_attention"]:
                        self.visual_attention = LearnedAttention(key_size=cnn_hidden_num_channels * 3,
                                                                 query_size=decoder_hidden_size,
                                                                 hidden_size=decoder_hidden_size)
                    else:
                        self.visual_attention = None
                    self.textual_attention = LearnedAttention(key_size=encoder_hidden_size,
                                                              query_size=decoder_hidden_size,
                                                              hidden_size=decoder_hidden_size)
                    attention_decoder = BahdanauAttentionDecoderRNN(hidden_size=decoder_hidden_size,
                                                                    output_size=target_args["target_vocabulary_size"],
                                                                    num_layers=num_decoder_layers,
                                                                    dropout_probability=decoder_dropout_p,
                                                                    padding_idx=target_args["target_pad_idx"],
                                                                    textual_attention=self.textual_attention,
                                                                    visual_attention=self.visual_attention,
                                                                    conditional_attention=target_args["conditional_attention"])
                else:
                    raise ValueError("Unknown type_attention=%s in target_specs, options are 'bahdanau' and "
                                     "'luong'" % target_args["type_attention"])
                decoder_modules[target_key] = attention_decoder
                self.sequence_decoder = True
                self.target_pad_idx = target_args["target_pad_idx"]
                self.target_sos_idx = target_args["target_sos_idx"]
                self.target_eos_idx = target_args["target_eos_idx"]
            elif target_args["output_type"] == "attention_map":
                self.target_keys.append(target_key)
                # Attention over the output features of the ConvolutionalNet.
                # Input: [bsz, 1, decoder_hidden_size], [bsz, image_width * image_width, cnn_hidden_num_channels * 3]
                # Output: [bsz, 1, decoder_hidden_size], [bsz, 1, image_width * image_width]
                world_attention = LearnedAttention(key_size=cnn_hidden_num_channels * 3, query_size=encoder_hidden_size,
                                                   hidden_size=encoder_hidden_size)
                decoder_modules[target_key] = nn.ModuleDict()
                decoder_modules[target_key]["attention"] = world_attention
                # decoder_modules[target_key]["projection"] = nn.Sequential(hidden_to_hidden, nn.Tanh(), hidden_to_output)
            elif target_args["output_type"] == "attention_map_projection":
                self.target_keys.append(target_key)
                # Attention over the output features of the ConvolutionalNet.
                # Input: [bsz, 1, decoder_hidden_size], [bsz, image_width * image_width, cnn_hidden_num_channels * 3]
                # Output: [bsz, 1, decoder_hidden_size], [bsz, 1, image_width * image_width]
                world_attention = LearnedAttention(key_size=cnn_hidden_num_channels * 3, query_size=encoder_hidden_size,
                                                   hidden_size=encoder_hidden_size)
                hidden_to_hidden = nn.Linear(encoder_hidden_size * 2,
                                             target_args["hidden_size"])
                hidden_to_output = nn.Linear(target_args["hidden_size"], target_args["output_size"])
                decoder_modules[target_key] = nn.ModuleDict()
                decoder_modules[target_key]["attention"] = world_attention
                decoder_modules[target_key]["projection"] = nn.Sequential(hidden_to_hidden, nn.Tanh(), hidden_to_output)
            elif target_args["output_type"] == "mlp":
                hidden_to_hidden = nn.Linear(decoder_hidden_size,
                                             target_args["hidden_size"])
                dropout = nn.Dropout(p=decoder_dropout_p)
                hidden_to_output = nn.Linear(target_args["hidden_size"], target_args["output_size"])
                decoder_modules[target_key] = nn.Sequential(hidden_to_hidden, nn.Tanh(), dropout, hidden_to_output)
            else:
                raise ValueError("Unrecognized data_type=%s for target_key=%s in constructor class for SuperModule" % (
                    target_args["data_type"], target_key
                ))

        if self.sequence_decoder:
            self.loss_criterion = nn.NLLLoss(ignore_index=self.target_pad_idx)
        else:
            self.loss_criterion = nn.NLLLoss()

        self.decoder_module_dict = nn.ModuleDict(decoder_modules)

        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_metric = 0
        self.metric = metric

    @staticmethod
    def remove_start_of_sequence(input_tensor: torch.Tensor) -> torch.Tensor:
        """Get rid of SOS-tokens in targets batch and append a padding token to each example in the batch."""
        batch_size, max_time = input_tensor.size()
        input_tensor = input_tensor[:, 1:]
        output_tensor = torch.cat([input_tensor, torch.zeros(batch_size, device=device, dtype=torch.long).unsqueeze(
            dim=1)], dim=1)
        return output_tensor

    def add_start_of_sequence(self, input_tensor: torch.Tensor):
        batch_size, max_time = input_tensor.size()
        sos_tensor = torch.tensor([self.target_sos_idx] * batch_size, device=device,
                                  dtype=input_tensor.dtype).unsqueeze(dim=1)
        return torch.cat([sos_tensor, input_tensor], dim=1)


    def get_prediction(self, target_scores: torch.Tensor):
        pred = target_scores.max(dim=-1)[1]
        if not len(pred.shape):
            pred = pred.unsqueeze(dim=0)
        return pred

    @staticmethod
    def get_single_accuracy(target_scores: torch.Tensor, targets: torch.Tensor) -> float:
        num_targets = targets.size(0)
        if len(target_scores.shape) == 1:
            target_scores = target_scores.unsqueeze(dim=0)
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(dim=0)
        with torch.no_grad():
            predicted_targets = target_scores.max(dim=1)[1]
            equal_targets = torch.eq(targets.data, predicted_targets.data).long().sum().data.item()
            accuracy = 100. * equal_targets / num_targets
        return accuracy

    def get_exact_match(self, scores: torch.Tensor, targets: torch.Tensor):
        """
        :param scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            targets = self.remove_start_of_sequence(targets)
            mask = (targets != self.target_pad_idx).long()
            total = mask.sum().data.item()
            predicted_targets = scores.max(dim=2)[1]
            equal_targets = torch.eq(targets.data, predicted_targets.data).long()
            match_targets = (equal_targets * mask)
            match_sum_per_example = match_targets.sum(dim=1)
            expected_sum_per_example = mask.sum(dim=1)
            batch_size = expected_sum_per_example.size(0)
            exact_match = 100. * (match_sum_per_example == expected_sum_per_example).sum().data.item() / batch_size
            match_targets_sum = match_targets.sum().data.item()
            accuracy = 100. * match_targets_sum / total
        return accuracy, exact_match

    def get_scores(self, output_scores: torch.Tensor) -> torch.Tensor:
        output_scores_normalized = F.log_softmax(output_scores, -1)
        return output_scores_normalized

    def get_single_loss(self, scores: torch.Tensor, targets: torch.Tensor):
        loss = self.loss_criterion(scores, targets.view(-1))
        return loss

    def get_sequence_loss(self, scores: torch.Tensor, targets: torch.Tensor):
        targets = self.remove_start_of_sequence(targets)

        # Calculate the loss.
        vocabulary_size = scores.size(2)
        target_scores_2d = scores.reshape(-1, vocabulary_size)
        loss = self.loss_criterion(target_scores_2d, targets.view(-1))
        return loss

    def get_metrics(self, **kwargs) -> Dict[str, float]:
        raise NotImplementedError()

    def get_loss(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def prepare_encoder_output_decoder(self, encoder_outputs: Dict[str, Union[torch.Tensor, List[int]]]):
        raise NotImplementedError()

    def encode_input(self, inputs: Dict[str, Union[torch.Tensor, List[int]]], **kwargs) -> Dict[str, torch.Tensor]:
        """Pass the inputs through encoders."""

        encoded_inputs = {}
        mlp_inputs = []
        for input_key, input_args in self.input_specs.items():
            if input_args["data_type"] == "sequence" or input_args["data_type"] == "bow":
                encoded_input = self.encoder_module_dict[input_key](inputs[input_key], inputs[input_key + "_lengths"])
                encoded_inputs[input_key] = encoded_input
            elif input_args["data_type"] == "int":
                mlp_inputs.append(inputs[input_key].unsqueeze(dim=1))
            else:
                encoded_input = self.encoder_module_dict[input_key](inputs[input_key])
                encoded_inputs[input_key] = encoded_input
        if len(mlp_inputs):
            mlp_inputs = torch.cat(mlp_inputs, dim=1) + 1  # TODO: keep this?
            encoded_inputs["mlp"] = self.encoder_module_dict["mlp"](mlp_inputs.float())
        if self.gpu_util:
            torch.mm(self.friend_one, self.friend_two)
        return self.prepare_encoder_output_decoder(encoded_inputs)

    def decode_input(self, encoder_output) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""

        decoded_inputs = {}
        for target_key, target_args in self.target_specs.items():
            if target_args["output_type"] == "attention_map" or target_args["output_type"] == "attention_map_projection":
                batch_size, image_num_memory, image_dim = encoder_output["world_state_tensor"].size()
                situation_lengths = [image_num_memory for _ in range(batch_size)]
                context, attention_weights = self.decoder_module_dict[target_key]["attention"].forward(
                    queries=encoder_output["hidden"].unsqueeze(dim=1),
                    keys=encoder_output["world_state_tensor"],
                    values=encoder_output["world_state_tensor"],
                    memory_lengths=situation_lengths)
                if target_args["output_type"] == "attention_map_projection":
                    context = torch.tanh(context)
                    final_hidden = torch.cat([context.squeeze(dim=1), encoder_output["hidden"]], dim=-1)
                    attention_weights = self.decoder_module_dict[target_key]["projection"](final_hidden)
                scores_normalized = self.get_scores(attention_weights)
            elif target_args["output_type"] == "mlp":
                output_states = self.decoder_module_dict[target_key](encoder_output["hidden"])
                scores_normalized = self.get_scores(output_states)
            else:
                raise ValueError("unrecognized output_type %s for target_key %s" % (target_args["output_type"],
                                                                                    target_key))
            decoded_inputs[target_key] = scores_normalized
        return decoded_inputs

    def decode_step(self, targets, encoder_outputs, **kwargs) -> Tuple[torch.Tensor,
                                                                     Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """One decoding step based on the previous hidden state of the decoder and the previous target token."""
        target_key = self.target_keys_to_pad[0]
        return self.decoder_module_dict[target_key].forward_step(
            target_tokens=targets[target_key + "_targets"],  **encoder_outputs)

    def decode_input_batched(self, targets: Dict[str, Union[torch.Tensor, List[int]]],
                             encoder_outputs: Dict[str, Union[torch.Tensor, List[int], tuple]],
                             **kwargs) -> Dict[str, torch.Tensor]:
        """Decode a batch of input sequences."""
        target_key = self.target_keys_to_pad[0]
        decoder_output_batched, _, attention_weights = self.decoder_module_dict[target_key](
            target_tokens=targets[target_key + "_targets"],
            target_lengths=targets[target_key + "_lengths"],
            **encoder_outputs)
        decoder_output_batched = F.log_softmax(decoder_output_batched, dim=-1)
        return {
            "decoder_output_batched": decoder_output_batched,
            "attention_weights": attention_weights
        }

    def forward(self, batch, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def update_state(self, is_best: bool, metric=None) -> {}:
        self.trained_iterations += 1
        if is_best:
            self.best_metric = metric
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint, map_location=torch.device('cpu'))
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_metric = checkpoint["best_metric"]
        self.metric = checkpoint["metric"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_metric": self.best_metric,
            "metric": self.metric,
            "simplified_architecture": self.simplified_architecture,
            "use_attention": self.use_attention,
            "use_conditional_attention": self.use_conditional_attention,
            "attention_values_key": self.attention_values_key,
            "conditional_attention_values_key": self.conditional_attention_values_key,
            "type_attention": self.type_attention,
            "embedding_dimension": self.embedding_dimension,
            "encoder_hidden_size": self.encoder_hidden_size,
            "decoder_hidden_size": self.decoder_hidden_size,
            "num_encoder_layers": self.num_encoder_layers,
            "encoder_bidirectional": self.encoder_bidirectional,
            "num_decoder_layers": self.num_decoder_layers,
            "cnn_kernel_size": self.cnn_kernel_size,
            "cnn_hidden_num_channels": self.cnn_hidden_num_channels,
            "grid_size": self.grid_size
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path


class InteractionModel(SuperModule):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 transitive_input_vocabulary_size: int, transitive_input_padding_idx: int, input_vocabulary: str,
                 num_encoder_layers: int, encoder_dropout_p: float, encoder_bidirectional: bool, input_padding_idx: int,
                 output_directory: str, decoder_hidden_size: int, transitive_target_pad_idx: int,
                 transitive_target_sos_idx: int, transitive_target_vocabulary_size: int, num_decoder_layers: int,
                 decoder_dropout_p: float, transitive_target_eos_idx: int, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, grid_size: int, simplified_architecture: bool,
                 use_attention: bool, use_conditional_attention: bool, target_vocabulary: str, main_input_key: str,
                 type_attention: str, attention_values_key: str,  conditional_attention_values_key: str, **kwargs):
        self.simplified_architecture = simplified_architecture
        assert type_attention in ["bahdanau", "luong"], "Unknown type_attention=%s. Options are 'bahdanau', 'luong'"
        if not simplified_architecture:
            self.input_specs = {
                "world_state_tensor": {
                    "data_type": "grid",
                    "num_input_channels": num_cnn_channels,
                },
                "input_tensor": {
                    "data_type": "sequence",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                },
                "transitive_input_tensor": {
                    "data_type": "sequence",
                    "input_vocabulary_size": transitive_input_vocabulary_size,
                    "input_padding_idx": transitive_input_padding_idx
                }
            }
            self.target_specs = {
                "transitive_target_tensor": {
                    "output_type": "sequence",
                    "type_attention": type_attention,
                    "attention": use_attention,
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": transitive_target_vocabulary_size,
                    "target_pad_idx": transitive_target_pad_idx,
                    "target_eos_idx": transitive_target_eos_idx,
                    "target_sos_idx": transitive_target_sos_idx
                }
            }
        else:
            self.input_specs = {
                "world_state_tensor": {
                    "data_type": "grid",
                    "num_input_channels": num_cnn_channels,
                },
                "input_tensor": {
                    "data_type": "bow",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                },
                "transitive_input_tensor": {
                    "data_type": "sequence",
                    "input_vocabulary_size": transitive_input_vocabulary_size,
                    "input_padding_idx": transitive_input_padding_idx
                }
            }
            self.target_specs = {
                "transitive_target_tensor": {
                    "output_type": "sequence",
                    "type_attention": type_attention,
                    "attention": use_attention,
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": transitive_target_vocabulary_size,
                    "target_pad_idx": transitive_target_pad_idx,
                    "target_eos_idx": transitive_target_eos_idx,
                    "target_sos_idx": transitive_target_sos_idx
                }
            }
        metric = "exact match"
        self.attention_values_key = attention_values_key
        self.conditional_attention_values_key = conditional_attention_values_key

        self.input_keys = ["world_state_tensor"]
        self.input_keys_to_pad = ["input_tensor", "transitive_input_tensor"]
        self.target_keys = []
        self.target_keys_to_pad = ["transitive_target_tensor"]
        self.sequence_decoder = True
        self.use_attention = use_attention
        self.use_conditional_attention = use_conditional_attention
        self.type_attention = type_attention

        super().__init__(input_specs=self.input_specs, target_specs=self.target_specs, metric=metric,
                         embedding_dimension=embedding_dimension, encoder_hidden_size=encoder_hidden_size,
                         num_encoder_layers=num_encoder_layers, encoder_dropout_p=encoder_dropout_p,
                         encoder_bidirectional=encoder_bidirectional, output_directory=output_directory,
                         decoder_hidden_size=decoder_hidden_size, num_decoder_layers=num_decoder_layers,
                         decoder_dropout_p=decoder_dropout_p, cnn_kernel_size=cnn_kernel_size,
                         cnn_dropout_p=cnn_dropout_p, cnn_hidden_num_channels=cnn_hidden_num_channels,
                         grid_size=grid_size, target_vocabulary=target_vocabulary, input_vocabulary=input_vocabulary,
                         main_input_key=main_input_key)

        # Module specific combination parameters
        if not simplified_architecture:
            self.world_state_to_decoder_hidden = nn.Linear(cnn_hidden_num_channels * 3,
                                                           decoder_hidden_size)
            self.encoder_hiddens_to_initial_hidden = nn.Linear(encoder_hidden_size * 2,
                                                               decoder_hidden_size)
        else:
            self.world_state_to_decoder_hidden = nn.Linear(cnn_hidden_num_channels * 3 * grid_size ** 2,
                                                           decoder_hidden_size)
            self.encoder_to_initial_hidden = nn.Linear(decoder_hidden_size + (encoder_hidden_size * 2),
                                                       decoder_hidden_size)

    def get_predictions(self, batch, max_decoding_steps, **kwargs):
        return predict_sequence(self, batch, max_decoding_steps)

    def get_metrics(self, transitive_target_tensor_scores: torch.Tensor, transitive_target_tensor_targets: torch.Tensor,
                    **kwargs) -> Dict[str, float]:
        """
        :param transitive_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param transitive_target_tensor_targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        accuracy, exact_match = self.get_exact_match(scores=transitive_target_tensor_scores,
                                                     targets=transitive_target_tensor_targets)
        return {
            "transitive_target_tensor_accuracy": accuracy,
            "transitive_target_tensor_exact_match": exact_match
        }

    def get_loss(self, transitive_target_tensor_scores: torch.Tensor, transitive_target_tensor_targets: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """
        :param transitive_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param transitive_target_tensor_targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        return self.get_sequence_loss(transitive_target_tensor_scores, transitive_target_tensor_targets)

    def prepare_encoder_output_decoder(self, encoder_output: Dict[str, Union[torch.Tensor, List[int], tuple]]):
        if not self.simplified_architecture:
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            hidden_state_adverb_input_tensor = encoder_output["transitive_input_tensor"][0]
            attention_values = encoder_output[self.attention_values_key][1]["encoder_outputs"]
            attention_values_lengths = encoder_output[self.attention_values_key][1]["sequence_lengths"]
            initial_hidden = torch.cat([hidden_state_input_tensor,
                                        hidden_state_adverb_input_tensor], dim=1)
            batch_size = initial_hidden.size(0)
            initial_hidden = self.tanh(self.encoder_hiddens_to_initial_hidden(initial_hidden))
            conditional_attention_values = torch.tanh(
                self.world_state_to_decoder_hidden(encoder_output[self.conditional_attention_values_key].transpose(0, 1)))
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
            conditional_attention_values_lengths = [self.grid_size ** 2] * batch_size
        else:
            encoded_world_state = encoder_output["world_state_tensor"]
            batch_size, image_num_memory, image_dim = encoded_world_state.size()
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            hidden_state_transitive_input_tensor = encoder_output["transitive_input_tensor"][0]
            context_world_state = self.world_state_to_decoder_hidden(
                encoded_world_state.reshape(batch_size, image_num_memory * image_dim))
            attention_values = None
            attention_values_lengths = None
            initial_hidden = torch.cat([context_world_state,
                                        hidden_state_transitive_input_tensor,
                                        hidden_state_input_tensor], dim=1)
            initial_hidden = self.tanh(self.encoder_to_initial_hidden(initial_hidden))
            conditional_attention_values = None
            conditional_attention_values_lengths = None
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
        return {"hidden": initial_hidden,
                "attention_keys": attention_values,
                "attention_values": attention_values,
                "attention_values_lengths": attention_values_lengths,
                "conditional_attention_values": conditional_attention_values,
                "conditional_attention_values_lengths": conditional_attention_values_lengths}

    def forward(self, batch, **kwargs) -> Dict[
        str, torch.Tensor]:
        encoder_output = self.encode_input(batch["inputs"])
        decoder_output = self.decode_input_batched(batch["targets"], encoder_output)
        # decoder_output_batched: [max_target_length, batch_size, output_size]
        # attention_weights: [batch_size, max_target_length, max_input_length]
        return {
            "transitive_target_tensor_scores": decoder_output["decoder_output_batched"].transpose(0, 1),
            "decoder_attention_weights": decoder_output["attention_weights"]
        }


class NewInteractionModel(SuperModule):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 input_vocabulary: str,
                 num_encoder_layers: int, encoder_dropout_p: float, encoder_bidirectional: bool, input_padding_idx: int,
                 output_directory: str, decoder_hidden_size: int, transitive_target_pad_idx: int,
                 transitive_target_sos_idx: int, transitive_target_vocabulary_size: int, num_decoder_layers: int,
                 decoder_dropout_p: float, transitive_target_eos_idx: int, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, grid_size: int, simplified_architecture: bool,
                 use_attention: bool, use_conditional_attention: bool, target_vocabulary: str, main_input_key: str,
                 type_attention: str, attention_values_key: str,  conditional_attention_values_key: str, **kwargs):
        self.simplified_architecture = simplified_architecture
        assert type_attention in ["bahdanau", "luong"], "Unknown type_attention=%s. Options are 'bahdanau', 'luong'"
        self.type_attention = type_attention
        if not simplified_architecture:
            self.input_specs = {
                "world_state_tensor": {
                    "data_type": "grid",
                    "num_input_channels": num_cnn_channels,
                },
                "input_tensor": {
                    "data_type": "sequence",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                },
                "target_position": {
                    "data_type": "int",
                }
            }
            self.target_specs = {
                "transitive_target_tensor": {
                    "output_type": "sequence",
                    "type_attention": type_attention,
                    "attention": use_attention,
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": transitive_target_vocabulary_size,
                    "target_pad_idx": transitive_target_pad_idx,
                    "target_eos_idx": transitive_target_eos_idx,
                    "target_sos_idx": transitive_target_sos_idx
                }
            }
        else:
            self.input_specs = {
                "world_state_tensor": {
                    "data_type": "grid",
                    "num_input_channels": num_cnn_channels,
                },
                "input_tensor": {
                    "data_type": "bow",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                },
                "target_position": {
                    "data_type": "int",
                }
            }
            self.target_specs = {
                "transitive_target_tensor": {
                    "output_type": "sequence",
                    "type_attention": type_attention,
                    "attention": use_attention,
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": transitive_target_vocabulary_size,
                    "target_pad_idx": transitive_target_pad_idx,
                    "target_eos_idx": transitive_target_eos_idx,
                    "target_sos_idx": transitive_target_sos_idx
                }
            }
        metric = "exact match"
        self.attention_values_key = attention_values_key
        self.conditional_attention_values_key = conditional_attention_values_key

        self.input_keys = ["world_state_tensor", "target_position"]
        self.input_keys_to_pad = ["input_tensor"]
        self.target_keys = []
        self.target_keys_to_pad = ["transitive_target_tensor"]
        self.sequence_decoder = True
        self.use_attention = use_attention
        self.use_conditional_attention = use_conditional_attention

        super().__init__(input_specs=self.input_specs, target_specs=self.target_specs, metric=metric,
                         embedding_dimension=embedding_dimension, encoder_hidden_size=encoder_hidden_size,
                         num_encoder_layers=num_encoder_layers, encoder_dropout_p=encoder_dropout_p,
                         encoder_bidirectional=encoder_bidirectional, output_directory=output_directory,
                         decoder_hidden_size=decoder_hidden_size, num_decoder_layers=num_decoder_layers,
                         decoder_dropout_p=decoder_dropout_p, cnn_kernel_size=cnn_kernel_size,
                         cnn_dropout_p=cnn_dropout_p, cnn_hidden_num_channels=cnn_hidden_num_channels,
                         grid_size=grid_size, target_vocabulary=target_vocabulary, input_vocabulary=input_vocabulary,
                         main_input_key=main_input_key)

        # Module specific combination parameters
        if not simplified_architecture:
            self.world_state_to_decoder_hidden = nn.Linear(cnn_hidden_num_channels * 3,
                                                           decoder_hidden_size)
            self.encoder_hiddens_to_initial_hidden = nn.Linear(encoder_hidden_size * 2,
                                                               decoder_hidden_size)
        else:
            self.world_state_to_decoder_hidden = nn.Linear(cnn_hidden_num_channels * 3 * grid_size ** 2,
                                                           decoder_hidden_size)
            self.encoder_to_initial_hidden = nn.Linear(decoder_hidden_size + (encoder_hidden_size * 2),
                                                       decoder_hidden_size)

    def get_predictions(self, batch, max_decoding_steps, **kwargs):
        return predict_sequence(self, batch, max_decoding_steps)

    def get_metrics(self, transitive_target_tensor_scores: torch.Tensor, transitive_target_tensor_targets: torch.Tensor,
                    **kwargs) -> Dict[str, float]:
        """
        :param transitive_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param transitive_target_tensor_targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        accuracy, exact_match = self.get_exact_match(scores=transitive_target_tensor_scores,
                                                     targets=transitive_target_tensor_targets)
        return {
            "transitive_target_tensor_accuracy": accuracy,
            "transitive_target_tensor_exact_match": exact_match
        }

    def get_loss(self, transitive_target_tensor_scores: torch.Tensor, transitive_target_tensor_targets: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """
        :param transitive_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param transitive_target_tensor_targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        return self.get_sequence_loss(transitive_target_tensor_scores, transitive_target_tensor_targets)

    def prepare_encoder_output_decoder(self, encoder_output: Dict[str, Union[torch.Tensor, List[int], tuple]]):
        if not self.simplified_architecture:
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            attention_values = encoder_output[self.attention_values_key][1]["encoder_outputs"]
            attention_values_lengths = encoder_output[self.attention_values_key][1]["sequence_lengths"]
            encoded_position = encoder_output["mlp"]
            initial_hidden = torch.cat([hidden_state_input_tensor,
                                        encoded_position], dim=1)
            batch_size = initial_hidden.size(0)
            initial_hidden = self.tanh(self.encoder_hiddens_to_initial_hidden(initial_hidden))
            if self.type_attention == "luong":
                conditional_attention_values = torch.tanh(
                    self.world_state_to_decoder_hidden(encoder_output[self.conditional_attention_values_key].transpose(0, 1)))
            else:
                conditional_attention_values = encoder_output[self.conditional_attention_values_key].transpose(0, 1)
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
            conditional_attention_values_lengths = [self.grid_size ** 2] * batch_size
        else:
            encoded_world_state = encoder_output["world_state_tensor"]
            batch_size, image_num_memory, image_dim = encoded_world_state.size()
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            encoded_position = encoder_output["mlp"]
            context_world_state = self.world_state_to_decoder_hidden(
                encoded_world_state.reshape(batch_size, image_num_memory * image_dim))
            attention_values = None
            attention_values_lengths = None
            initial_hidden = torch.cat([context_world_state,
                                        encoded_position,
                                        hidden_state_input_tensor], dim=1)
            initial_hidden = self.tanh(self.encoder_to_initial_hidden(initial_hidden))
            conditional_attention_values = None
            conditional_attention_values_lengths = None
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
        return {"hidden": initial_hidden,
                "attention_keys": attention_values,
                "attention_values": attention_values,
                "attention_values_lengths": attention_values_lengths,
                "conditional_attention_values": conditional_attention_values,
                "conditional_attention_values_lengths": conditional_attention_values_lengths}

    def forward(self, batch, **kwargs) -> Dict[
        str, torch.Tensor]:
        encoder_output = self.encode_input(batch["inputs"])
        decoder_output = self.decode_input_batched(batch["targets"], encoder_output)
        # decoder_output_batched: [max_target_length, batch_size, output_size]
        # attention_weights: [batch_size, max_target_length, max_input_length]
        return {
            "transitive_target_tensor_scores": decoder_output["decoder_output_batched"].transpose(0, 1),
            "decoder_attention_weights": decoder_output["attention_weights"]
        }


class DummyInteractionModel(SuperModule):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 transitive_input_vocabulary_size: int, transitive_input_padding_idx: int,
                 num_encoder_layers: int, encoder_dropout_p: float, encoder_bidirectional: bool, input_padding_idx: int,
                 output_directory: str, decoder_hidden_size: int, transitive_target_pad_idx: int,
                 transitive_target_sos_idx: int, transitive_target_vocabulary_size: int, num_decoder_layers: int,
                 decoder_dropout_p: float, transitive_target_eos_idx: int, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, grid_size: int, simplified_architecture: bool,
                 use_attention: bool, use_conditional_attention: bool, target_vocabulary: str,
                 main_input_key: str, input_vocabulary: str,
                 type_attention: str, attention_values_key: str,  conditional_attention_values_key: str, **kwargs):
        self.simplified_architecture = simplified_architecture
        assert type_attention in ["bahdanau", "luong"], "Unknown type_attention=%s. Options are 'bahdanau', 'luong'"
        if not simplified_architecture:
            self.input_specs = {
                "world_state_tensor": {
                    "data_type": "grid",
                    "num_input_channels": num_cnn_channels,
                },
                "input_tensor": {
                    "data_type": "sequence",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                }
            }
            self.target_specs = {
                "transitive_target_tensor": {
                    "output_type": "sequence",
                    "type_attention": type_attention,
                    "attention": use_attention,
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": transitive_target_vocabulary_size,
                    "target_pad_idx": transitive_target_pad_idx,
                    "target_eos_idx": transitive_target_eos_idx,
                    "target_sos_idx": transitive_target_sos_idx
                }
            }
        else:
            self.input_specs = {
                "world_state_tensor": {
                    "data_type": "grid",
                    "num_input_channels": num_cnn_channels,
                },
                "input_tensor": {
                    "data_type": "bow",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                }
            }
            self.target_specs = {
                "transitive_target_tensor": {
                    "output_type": "sequence",
                    "type_attention": type_attention,
                    "attention": use_attention,
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": transitive_target_vocabulary_size,
                    "target_pad_idx": transitive_target_pad_idx,
                    "target_eos_idx": transitive_target_eos_idx,
                    "target_sos_idx": transitive_target_sos_idx
                }
            }
        metric = "exact match"
        self.attention_values_key = attention_values_key
        self.conditional_attention_values_key = conditional_attention_values_key

        self.input_keys = ["world_state_tensor"]
        self.input_keys_to_pad = ["input_tensor"]
        self.target_keys = []
        self.target_keys_to_pad = ["transitive_target_tensor"]
        self.sequence_decoder = True
        self.use_attention = use_attention
        self.use_conditional_attention = use_conditional_attention

        super().__init__(input_specs=self.input_specs, target_specs=self.target_specs, metric=metric,
                         embedding_dimension=embedding_dimension, encoder_hidden_size=encoder_hidden_size,
                         num_encoder_layers=num_encoder_layers, encoder_dropout_p=encoder_dropout_p,
                         encoder_bidirectional=encoder_bidirectional, output_directory=output_directory,
                         decoder_hidden_size=decoder_hidden_size, num_decoder_layers=num_decoder_layers,
                         decoder_dropout_p=decoder_dropout_p, cnn_kernel_size=cnn_kernel_size,
                         cnn_dropout_p=cnn_dropout_p, cnn_hidden_num_channels=cnn_hidden_num_channels,
                         grid_size=grid_size, target_vocabulary=target_vocabulary, main_input_key=main_input_key,
                         input_vocabulary=input_vocabulary)
        # Module specific combination parameters
        if not simplified_architecture:
            # self.world_state_to_decoder_hidden = nn.Linear(cnn_hidden_num_channels * 3, decoder_hidden_size)
            self.encoder_hiddens_to_initial_hidden = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            self.encoder_to_initial_hidden = nn.Linear(cnn_hidden_num_channels * 3 + encoder_hidden_size,
                                                       decoder_hidden_size)

    def get_predictions(self, batch, max_decoding_steps, **kwargs):
        return predict_sequence(self, batch, max_decoding_steps)

    def get_metrics(self, transitive_target_tensor_scores: torch.Tensor, transitive_target_tensor_targets: torch.Tensor,
                    **kwargs) -> Dict[str, float]:
        """
        :param transitive_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param transitive_target_tensor_targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        accuracy, exact_match = self.get_exact_match(scores=transitive_target_tensor_scores,
                                                     targets=transitive_target_tensor_targets)
        return {
            "transitive_target_tensor_accuracy": accuracy,
            "transitive_target_tensor_exact_match": exact_match
        }

    def get_loss(self, transitive_target_tensor_scores: torch.Tensor, transitive_target_tensor_targets: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """
        :param transitive_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param transitive_target_tensor_targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        return self.get_sequence_loss(transitive_target_tensor_scores, transitive_target_tensor_targets)

    def prepare_encoder_output_decoder(self, encoder_output: Dict[str, Union[torch.Tensor, List[int], tuple]]):
        if not self.simplified_architecture:
            encoded_world_state = encoder_output["world_state_tensor"]
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            attention_values = encoder_output[self.attention_values_key][1]["encoder_outputs"]
            attention_values_lengths = encoder_output[self.attention_values_key][1]["sequence_lengths"]
            initial_hidden = hidden_state_input_tensor
            batch_size = initial_hidden.size(0)
            initial_hidden = self.tanh(self.encoder_hiddens_to_initial_hidden(initial_hidden))
            # conditional_attention_values = torch.tanh(
            #     self.world_state_to_decoder_hidden(encoder_output[self.conditional_attention_values_key].transpose(0, 1)))
            conditional_attention_values = encoder_output[self.conditional_attention_values_key].transpose(0, 1)
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
            conditional_attention_values_lengths = [self.grid_size ** 2] * batch_size
        else:
            raise NotImplementedError("Simplified dummy interaction module not implemented.")
            encoded_world_state = encoder_output["world_state_tensor"]
            batch_size, image_num_memory, image_dim = encoded_world_state.size()
            situation_lengths = [image_num_memory for _ in range(batch_size)]
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            input_tensor_embeddings = encoder_output["input_tensor"][1]["encoder_outputs"]
            input_tensor_lengths = encoder_output["input_tensor"][1]["sequence_lengths"]
            hidden_state_adverb_input_tensor = encoder_output["transitive_input_tensor"][0]

            context_world_state, attention_weights = self.attention(
                queries=hidden_state_adverb_input_tensor.unsqueeze(dim=1), keys=encoded_world_state,
                values=encoded_world_state, memory_lengths=situation_lengths)
            context_input, attention_weights_input = self.attention_two(
                queries=hidden_state_adverb_input_tensor.unsqueeze(dim=1), keys=input_tensor_embeddings,
                values=input_tensor_embeddings, memory_lengths=input_tensor_lengths)

            attention_values = None
            attention_values_lengths = None
            initial_hidden = torch.cat([context_world_state.squeeze(dim=1),
                                        context_input.squeeze(dim=1)], dim=1)
            batch_size = initial_hidden.size(0)
            initial_hidden = self.tanh(self.encoder_to_initial_hidden(initial_hidden))
            conditional_attention_values = None
            conditional_attention_values_lengths = None
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
        return {"hidden": initial_hidden,
                "attention_keys": attention_values,
                "attention_values": attention_values,
                "attention_values_lengths": attention_values_lengths,
                "conditional_attention_values": conditional_attention_values,
                "conditional_attention_values_lengths": conditional_attention_values_lengths}

    def forward(self, batch, **kwargs) -> Dict[
        str, torch.Tensor]:
        encoder_output = self.encode_input(batch["inputs"])
        decoder_output = self.decode_input_batched(batch["targets"], encoder_output)
        # decoder_output_batched: [max_target_length, batch_size, output_size]
        # attention_weights: [batch_size, max_target_length, max_input_length]
        return {
            "transitive_target_tensor_scores": decoder_output["decoder_output_batched"].transpose(0, 1),
            "decoder_attention_weights": decoder_output["attention_weights"]
        }


class PlannerModel(SuperModule):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, encoder_dropout_p: float, encoder_bidirectional: bool, input_padding_idx: int,
                 output_directory: str, decoder_hidden_size: int, planner_target_pad_idx: int, input_vocabulary: str,
                 planner_target_sos_idx: int, planner_target_vocabulary_size: int, num_decoder_layers: int,
                 decoder_dropout_p: float, planner_target_eos_idx: int, simplified_architecture: bool,
                 use_attention: bool, use_conditional_attention: bool, type_attention: str, attention_values_key: str,
                 conditional_attention_values_key: str, target_vocabulary: str, main_input_key: str, **kwargs):
        self.simplified_architecture = simplified_architecture
        assert type_attention in ["bahdanau", "luong"], "Unknown type_attention=%s. Options are 'bahdanau', 'luong'"
        assert not use_conditional_attention, "Cannot use conditional attention in PlannerModel."
        if not simplified_architecture:
            self.input_specs = {
                "input_tensor": {
                    "data_type": "sequence",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                },
                "agent_position": {
                    "data_type": "int",
                },
                "target_position": {
                    "data_type": "int",
                },
                "agent_direction": {
                    "data_type": "int",
                }
            }
            self.target_specs = {
                "planner_target_tensor": {
                    "output_type": "sequence",
                    "type_attention": type_attention,
                    "attention": use_attention,
                    "attention_values": "input_tensor",
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": planner_target_vocabulary_size,
                    "target_pad_idx": planner_target_pad_idx,
                    "target_eos_idx": planner_target_eos_idx,
                    "target_sos_idx": planner_target_sos_idx
                }
            }
        else:
            self.input_specs = {
                "input_tensor": {
                    "data_type": "bow",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                },
                "agent_position": {
                    "data_type": "int",
                },
                "target_position": {
                    "data_type": "int",
                },
                "agent_direction": {
                    "data_type": "int",
                }
            }
            self.target_specs = {
                "planner_target_tensor": {
                    "output_type": "sequence",
                    "type_attention": "luong",
                    "attention": use_attention,
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": planner_target_vocabulary_size,
                    "target_pad_idx": planner_target_pad_idx,
                    "target_eos_idx": planner_target_eos_idx,
                    "target_sos_idx": planner_target_sos_idx
                }
            }
        metric = "exact match"
        self.attention_values_key = attention_values_key
        self.conditional_attention_values_key = conditional_attention_values_key

        self.input_keys = ["agent_position", "target_position", "agent_direction"]
        self.input_keys_to_pad = ["input_tensor"]
        self.target_keys = []
        self.target_keys_to_pad = ["planner_target_tensor"]
        self.sequence_decoder = True
        self.use_attention = use_attention
        self.use_conditional_attention = use_conditional_attention
        self.type_attention = type_attention

        super().__init__(input_specs=self.input_specs, target_specs=self.target_specs, metric=metric,
                         embedding_dimension=embedding_dimension, encoder_hidden_size=encoder_hidden_size,
                         num_encoder_layers=num_encoder_layers, encoder_dropout_p=encoder_dropout_p,
                         encoder_bidirectional=encoder_bidirectional, output_directory=output_directory,
                         decoder_hidden_size=decoder_hidden_size, num_decoder_layers=num_decoder_layers,
                         decoder_dropout_p=decoder_dropout_p, target_vocabulary=target_vocabulary,
                         input_vocabulary=input_vocabulary, main_input_key=main_input_key)

        # Used to project the final encoder state to the decoder hidden state such that it can be initialized with it.
        self.encoder_hiddens_to_initial_hidden = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)

        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_metric = 0
        self.metric = "exact match"

        self.input_keys = ["agent_position", "target_position", "agent_direction"]
        self.input_keys_to_pad = ["input_tensor"]
        self.target_keys = []
        self.target_keys_to_pad = ["planner_target_tensor"]
        self.sequence_decoder = True

    def get_predictions(self, batch, max_decoding_steps, **kwargs):
        return predict_sequence(self, batch, max_decoding_steps)

    def get_metrics(self, planner_target_tensor_scores: torch.Tensor, planner_target_tensor_targets: torch.Tensor,
                    **kwargs) -> Dict[str, float]:
        """
        :param planner_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param planner_target_tensor_targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        accuracy, exact_match = self.get_exact_match(scores=planner_target_tensor_scores,
                                                     targets=planner_target_tensor_targets)
        return {
            "planner_target_tensor_accuracy": accuracy,
            "planner_target_tensor_exact_match": exact_match
        }

    def get_loss(self, planner_target_tensor_scores: torch.Tensor, planner_target_tensor_targets: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """
        :param planner_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param planner_target_tensor_targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        return self.get_sequence_loss(planner_target_tensor_scores, planner_target_tensor_targets)

    def prepare_encoder_output_decoder(self, encoder_output: Dict[str, Union[torch.Tensor, List[int], tuple]]):
        if not self.simplified_architecture:
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            if self.use_attention:
                attention_values = encoder_output[self.attention_values_key][1]["encoder_outputs"]
                attention_values_lengths = encoder_output[self.attention_values_key][1]["sequence_lengths"]
            else:
                attention_values = None
                attention_values_lengths = None
            encoded_positions = encoder_output["mlp"]
            initial_hidden = torch.cat([hidden_state_input_tensor,
                                        encoded_positions], dim=1)
            initial_hidden = self.tanh(self.encoder_hiddens_to_initial_hidden(initial_hidden))
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
        else:
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            if not self.use_attention:
                attention_values = None
                attention_values_lengths = None
            else:
                attention_values = encoder_output[self.attention_values_key][1]["encoder_outputs"].transpose(0, 1)
                attention_values_lengths = encoder_output[self.attention_values_key][1]["sequence_lengths"]
            encoded_positions = encoder_output["mlp"]
            initial_hidden = torch.cat([hidden_state_input_tensor,
                                        encoded_positions], dim=1)
            initial_hidden = self.tanh(self.encoder_hiddens_to_initial_hidden(initial_hidden))
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
        return {"hidden": initial_hidden,
                "attention_keys": attention_values,
                "attention_values": attention_values,
                "attention_values_lengths": attention_values_lengths,
                "conditional_attention_values": None,
                "conditional_attention_values_lengths": None}

    def forward(self, batch, **kwargs) -> Dict[str, torch.Tensor]:
        encoder_output = self.encode_input(batch["inputs"])
        decoder_output = self.decode_input_batched(batch["targets"], encoder_output)
        # decoder_output: [max_target_length, batch_size, output_size]
        # attention_weights: [batch_size, max_target_length, max_input_length]
        return {
            "planner_target_tensor_scores": decoder_output["decoder_output_batched"].transpose(0, 1),
            "decoder_attention_weights": decoder_output["attention_weights"],
        }


class DummyPlannerModel(SuperModule):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, encoder_dropout_p: float, encoder_bidirectional: bool, input_padding_idx: int,
                 output_directory: str, decoder_hidden_size: int, planner_target_pad_idx: int,
                 planner_target_sos_idx: int, planner_target_vocabulary_size: int, num_decoder_layers: int,
                 decoder_dropout_p: float, planner_target_eos_idx: int, simplified_architecture: bool,
                 use_attention: bool, use_conditional_attention: bool, type_attention: str, attention_values_key: str,
                 conditional_attention_values_key: str, grid_size: int, **kwargs):
        self.simplified_architecture = simplified_architecture
        assert type_attention in ["bahdanau", "luong"], "Unknown type_attention=%s. Options are 'bahdanau', 'luong'"
        assert not use_conditional_attention, "Cannot use conditional attention in DummyPlannerModel."
        if not simplified_architecture:
            self.input_specs = {
                "input_tensor": {
                    "data_type": "sequence",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                },
                "agent_position": {
                    "data_type": "int",
                },
                "target_position": {
                    "data_type": "int",
                },
                "agent_direction": {
                    "data_type": "int",
                }
            }
            self.target_specs = {
                "dummy_planner_target": {
                "output_type": "mlp",
                "hidden_size": encoder_hidden_size,
                "output_size": 7
                }
            }
        else:
            self.input_specs = {
                "input_tensor": {
                    "data_type": "bow",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                },
                "agent_position": {
                    "data_type": "int",
                },
                "target_position": {
                    "data_type": "int",
                },
                "agent_direction": {
                    "data_type": "int",
                }
            }
            self.target_specs = {
                "dummy_planner_target": {
                    "output_type": "mlp",
                    "hidden_size": encoder_hidden_size,
                    "output_size": 7
                }
            }
        metric = "exact match"
        self.attention_values_key = attention_values_key
        self.conditional_attention_values_key = conditional_attention_values_key

        self.input_keys = ["agent_position", "target_position", "agent_direction"]
        self.input_keys_to_pad = ["input_tensor"]
        self.target_keys = ["dummy_planner_target"]
        self.target_keys_to_pad = []
        self.sequence_decoder = False
        self.use_attention = use_attention

        super().__init__(input_specs=self.input_specs, target_specs=self.target_specs, metric=metric,
                         embedding_dimension=embedding_dimension, encoder_hidden_size=encoder_hidden_size,
                         num_encoder_layers=num_encoder_layers, encoder_dropout_p=encoder_dropout_p,
                         encoder_bidirectional=encoder_bidirectional, output_directory=output_directory,
                         decoder_hidden_size=decoder_hidden_size, num_decoder_layers=num_decoder_layers,
                         decoder_dropout_p=decoder_dropout_p)

        if self.simplified_architecture:
            self.attention = LearnedAttention(key_size=encoder_hidden_size,
                                              query_size=encoder_hidden_size,
                                              hidden_size=decoder_hidden_size)

        # Used to project the final encoder state to the decoder hidden state such that it can be initialized with it.
        self.encoder_hiddens_to_initial_hidden = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)

        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_metric = 0
        self.metric = "exact match"

        self.input_keys = ["agent_position", "target_position", "agent_direction"]
        self.input_keys_to_pad = ["input_tensor"]
        self.target_keys = ["dummy_planner_target"]
        self.target_keys_to_pad = []
        self.sequence_decoder = False

    def get_predictions(self, batch, max_decoding_steps, **kwargs):
        return predict_sequence(self, batch, max_decoding_steps)

    def get_metrics(self, dummy_planner_target_scores: torch.Tensor, dummy_planner_target_targets: torch.Tensor,
                    **kwargs) -> Dict[str, float]:
        """
        :param dummy_planner_target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param dummy_planner_target_targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        accuracy = self.get_single_accuracy(target_scores=dummy_planner_target_scores,
                                            targets=dummy_planner_target_targets)
        return {
            "dummy_planner_target_accuracy": accuracy
        }

    def get_loss(self, dummy_planner_target_scores: torch.Tensor, dummy_planner_target_targets: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """
        :param dummy_planner_target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param dummy_planner_target_targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        return self.get_single_loss(dummy_planner_target_scores, dummy_planner_target_targets)

    def prepare_encoder_output_decoder(self, encoder_output: Dict[str, Union[torch.Tensor, List[int], tuple]]):
        if not self.simplified_architecture:
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            attention_values = None
            attention_values_lengths = None
            encoded_positions = encoder_output["mlp"]
            initial_hidden = torch.cat([hidden_state_input_tensor,
                                        encoded_positions], dim=1)
            initial_hidden = self.tanh(self.encoder_hiddens_to_initial_hidden(initial_hidden))
        else:
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            attention_values = None
            attention_values_lengths = None
            encoded_positions = encoder_output["mlp"]
            initial_hidden = torch.cat([hidden_state_input_tensor,
                                        encoded_positions], dim=1)
            initial_hidden = self.tanh(self.encoder_hiddens_to_initial_hidden(initial_hidden))
        return {"hidden": initial_hidden,
                "attention_keys": attention_values,
                "attention_values": attention_values,
                "attention_values_lengths": attention_values_lengths,
                "conditional_attention_values": None,
                "conditional_attention_values_lengths": None}

    def forward(self, batch, **kwargs) -> Dict[str, torch.Tensor]:
        encoder_output = self.encode_input(batch["inputs"])
        decoder_output = self.decode_input(encoder_output)
        # decoder_output: [max_target_length, batch_size, output_size]
        # attention_weights: [batch_size, max_target_length, max_input_length]
        return {
            "dummy_planner_target_scores": decoder_output["dummy_planner_target"]
        }


class DummySequencePlannerModel(SuperModule):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, encoder_dropout_p: float, encoder_bidirectional: bool, input_padding_idx: int,
                 output_directory: str, decoder_hidden_size: int, planner_target_pad_idx: int,
                 planner_target_sos_idx: int, planner_target_vocabulary_size: int, num_decoder_layers: int,
                 decoder_dropout_p: float, planner_target_eos_idx: int, simplified_architecture: bool,
                 use_attention: bool, use_conditional_attention: bool, type_attention: str, attention_values_key: str,
                 conditional_attention_values_key: str, grid_size: int, **kwargs):
        self.simplified_architecture = simplified_architecture
        assert type_attention in ["bahdanau", "luong"], "Unknown type_attention=%s. Options are 'bahdanau', 'luong'"
        assert not use_conditional_attention, "Cannot use conditional attention in DummyPlannerModel."
        if not simplified_architecture:
            self.input_specs = {
                "input_tensor": {
                    "data_type": "sequence",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                },
                "agent_position": {
                    "data_type": "int",
                },
                "target_position": {
                    "data_type": "int",
                },
                "agent_direction": {
                    "data_type": "int",
                }
            }
            self.target_specs = {
                "dummy_planner_target_tensor": {
                    "output_type": "sequence",
                    "type_attention": type_attention,
                    "attention": use_attention,
                    "attention_values": "input_tensor",
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": planner_target_vocabulary_size,
                    "target_pad_idx": planner_target_pad_idx,
                    "target_eos_idx": planner_target_eos_idx,
                    "target_sos_idx": planner_target_sos_idx
                }
            }
        else:
            self.input_specs = {
                "input_tensor": {
                    "data_type": "bow",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                },
                "agent_position": {
                    "data_type": "int",
                },
                "target_position": {
                    "data_type": "int",
                },
                "agent_direction": {
                    "data_type": "int",
                }
            }
            self.target_specs = {
                "dummy_planner_target_tensor": {
                    "output_type": "sequence",
                    "type_attention": type_attention,
                    "attention": use_attention,
                    "attention_values": "input_tensor",
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": planner_target_vocabulary_size,
                    "target_pad_idx": planner_target_pad_idx,
                    "target_eos_idx": planner_target_eos_idx,
                    "target_sos_idx": planner_target_sos_idx
                }
            }
        metric = "exact match"
        self.attention_values_key = attention_values_key
        self.conditional_attention_values_key = conditional_attention_values_key

        self.input_keys = ["agent_position", "target_position", "agent_direction"]
        self.input_keys_to_pad = ["input_tensor"]
        self.target_keys = []
        self.target_keys_to_pad = ["dummy_planner_target_tensor"]
        self.sequence_decoder = True
        self.use_attention = use_attention

        super().__init__(input_specs=self.input_specs, target_specs=self.target_specs, metric=metric,
                         embedding_dimension=embedding_dimension, encoder_hidden_size=encoder_hidden_size,
                         num_encoder_layers=num_encoder_layers, encoder_dropout_p=encoder_dropout_p,
                         encoder_bidirectional=encoder_bidirectional, output_directory=output_directory,
                         decoder_hidden_size=decoder_hidden_size, num_decoder_layers=num_decoder_layers,
                         decoder_dropout_p=decoder_dropout_p)

        if self.simplified_architecture:
            self.attention = LearnedAttention(key_size=encoder_hidden_size,
                                              query_size=encoder_hidden_size,
                                              hidden_size=decoder_hidden_size)

        # Used to project the final encoder state to the decoder hidden state such that it can be initialized with it.
        self.encoder_hiddens_to_initial_hidden = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)

        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_metric = 0
        self.metric = "exact match"

        self.input_keys = ["agent_position", "target_position", "agent_direction"]
        self.input_keys_to_pad = ["input_tensor"]
        self.target_keys = []
        self.target_keys_to_pad = ["dummy_planner_target_tensor"]
        self.sequence_decoder = True

    def get_predictions(self, batch, max_decoding_steps, **kwargs):
        return predict_sequence(self, batch, max_decoding_steps)

    def get_metrics(self, dummy_planner_target_tensor_scores: torch.Tensor, dummy_planner_target_tensor_targets: torch.Tensor,
                    **kwargs) -> Dict[str, float]:
        """
        :param dummy_planner_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param dummy_planner_target_tensor_targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        accuracy, exact_match = self.get_exact_match(scores=dummy_planner_target_tensor_scores,
                                                     targets=dummy_planner_target_tensor_targets)
        return {
            "dummy_planner_target_tensor_accuracy": accuracy,
            "dummy_planner_target_tensor_exact_match": exact_match
        }

    def get_loss(self, dummy_planner_target_tensor_scores: torch.Tensor, dummy_planner_target_tensor_targets: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """
        :param dummy_planner_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param dummy_planner_target_tensor_targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        return self.get_sequence_loss(dummy_planner_target_tensor_scores, dummy_planner_target_tensor_targets)

    def prepare_encoder_output_decoder(self, encoder_output: Dict[str, Union[torch.Tensor, List[int], tuple]]):
        if not self.simplified_architecture:
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            if not self.use_attention:
                attention_values = None
                attention_values_lengths = None
            else:
                attention_values = encoder_output[self.attention_values_key][1]["encoder_outputs"]
                attention_values_lengths = encoder_output[self.attention_values_key][1]["sequence_lengths"]
            encoded_positions = encoder_output["mlp"]
            initial_hidden = torch.cat([hidden_state_input_tensor,
                                        encoded_positions], dim=1)
            initial_hidden = self.tanh(self.encoder_hiddens_to_initial_hidden(initial_hidden))
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
        else:
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            if not self.use_attention:
                attention_values = None
                attention_values_lengths = None
            else:
                attention_values = encoder_output[self.attention_values_key][1]["encoder_outputs"].transpose(0, 1)
                attention_values_lengths = encoder_output[self.attention_values_key][1]["sequence_lengths"]
            encoded_positions = encoder_output["mlp"]
            initial_hidden = torch.cat([hidden_state_input_tensor,
                                        encoded_positions], dim=1)
            initial_hidden = self.tanh(self.encoder_hiddens_to_initial_hidden(initial_hidden))
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
        return {"hidden": initial_hidden,
                "attention_keys": attention_values,
                "attention_values": attention_values,
                "attention_values_lengths": attention_values_lengths,
                "conditional_attention_values": None,
                "conditional_attention_values_lengths": None}

    def forward(self, batch, **kwargs) -> Dict[str, torch.Tensor]:
        encoder_output = self.encode_input(batch["inputs"])
        decoder_output = self.decode_input_batched(batch["targets"], encoder_output)
        # decoder_output: [max_target_length, batch_size, output_size]
        # attention_weights: [batch_size, max_target_length, max_input_length]
        return {
            "dummy_planner_target_tensor_scores": decoder_output["decoder_output_batched"].transpose(0, 1),
            "decoder_attention_weights": decoder_output["attention_weights"]
        }


class PositionModel(SuperModule):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, encoder_dropout_p: float, target_vocabulary: str, input_vocabulary: str,
                 main_input_key: str, encoder_bidirectional: bool, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int,
                 output_directory: str, grid_size: int, **kwargs):
        self.simplified_architecture = False
        self.input_specs = {
            "world_state_tensor": {
                "data_type": "grid",
                "num_input_channels": num_cnn_channels,
            },
            "input_tensor": {
                "data_type": "sequence",
                "input_vocabulary_size": input_vocabulary_size,
                "input_padding_idx": input_padding_idx
            }
        }
        self.target_specs = {
            "agent_position": {
                "output_type": "attention_map",
                "hidden_size": encoder_hidden_size,
                "output_size": grid_size**2
            },
            "target_position": {
                "output_type": "attention_map",
                "hidden_size": encoder_hidden_size,
                "output_size": grid_size**2
            },
            "agent_direction": {
                "output_type": "attention_map_projection",
                "hidden_size": encoder_hidden_size,
                "output_size": 4
            }
        }
        metric = "accuracy"

        self.input_keys = ["world_state_tensor"]
        self.input_keys_to_pad = ["input_tensor"]
        self.target_keys = ["agent_position", "target_position", "agent_direction"]
        self.target_keys_to_pad = []
        self.sequence_decoder = False
        self.use_attention = False
        self.attention_values_key = ""
        self.conditional_attention_values_key = ""
        self.use_conditional_attention = False
        self.type_attention = ""

        super().__init__(input_specs=self.input_specs, target_specs=self.target_specs, metric=metric,
                         embedding_dimension=embedding_dimension, encoder_hidden_size=encoder_hidden_size,
                         num_encoder_layers=num_encoder_layers, encoder_dropout_p=encoder_dropout_p,
                         encoder_bidirectional=encoder_bidirectional, output_directory=output_directory,
                         cnn_kernel_size=cnn_kernel_size, cnn_dropout_p=cnn_dropout_p, input_vocabulary=input_vocabulary,
                         target_vocabulary=target_vocabulary, cnn_hidden_num_channels=cnn_hidden_num_channels,
                         grid_size=grid_size, main_input_key=main_input_key)

    def get_predictions(self, batch, **kwargs):
        return predict_step(self, batch)

    def get_metrics(self, agent_direction_scores: torch.Tensor, agent_position_scores: torch.Tensor,
                    target_position_scores: torch.Tensor, agent_position_targets: torch.Tensor,
                    agent_direction_targets: torch.Tensor, target_position_targets: torch.Tensor):
        direction_accuracy = self.get_single_accuracy(agent_direction_scores, agent_direction_targets)
        agent_position_accuracy = self.get_single_accuracy(agent_position_scores, agent_position_targets)
        target_position_accuracy = self.get_single_accuracy(target_position_scores, target_position_targets)
        return {"agent_direction_accuracy": direction_accuracy,
                "agent_position_accuracy": agent_position_accuracy,
                "target_position_accuracy": target_position_accuracy}

    def get_loss(self, agent_direction_scores: torch.Tensor, agent_position_scores: torch.Tensor,
                 target_position_scores: torch.Tensor, agent_position_targets: torch.Tensor,
                 agent_direction_targets: torch.Tensor, target_position_targets: torch.Tensor):
        direction_loss = self.get_single_loss(agent_direction_scores, agent_direction_targets)
        agent_position_loss = self.get_single_loss(agent_position_scores, agent_position_targets)
        target_position_loss = self.get_single_loss(target_position_scores, target_position_targets)
        return torch.tensor(0.01, device=device) * direction_loss + torch.tensor(0.01, device=device) * agent_position_loss + torch.tensor(0.98, device=device) * target_position_loss

    def prepare_encoder_output_decoder(self, encoder_output: Dict[str, Union[torch.Tensor, List[int], tuple]]):
        encoded_world_state = encoder_output["world_state_tensor"]
        hidden_state_input_tensor = encoder_output["input_tensor"][0]
        return {"hidden": hidden_state_input_tensor,
                "world_state_tensor": encoded_world_state,
                "attention_keys": None,
                "attention_values": None,
                "attention_values_lengths": None,
                "conditional_attention_values": None,
                "conditional_attention_values_lengths": None}

    def forward(self, batch, **kwargs) -> Dict[str, torch.Tensor]:
        encoder_output = self.encode_input(batch["inputs"])
        decoder_output = self.decode_input(encoder_output)
        return {"agent_direction_scores": decoder_output["agent_direction"].squeeze(),
                "agent_position_scores": decoder_output["agent_position"].squeeze(),
                "target_position_scores": decoder_output["target_position"].squeeze()}


class AdverbTransform(SuperModule):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 adverb_input_vocabulary_size: int, adverb_input_padding_idx: int, input_vocabulary: str,
                 num_encoder_layers: int, encoder_dropout_p: float, encoder_bidirectional: bool, input_padding_idx: int,
                 output_directory: str, decoder_hidden_size: int, adverb_target_pad_idx: int, main_input_key: str,
                 adverb_target_sos_idx: int, adverb_target_vocabulary_size: int, num_decoder_layers: int,
                 decoder_dropout_p: float, adverb_target_eos_idx: int, type_attention: str, simplified_architecture: bool,
                 use_attention: bool, use_conditional_attention: bool, target_vocabulary: str,
                 attention_values_key: str, conditional_attention_values_key: str, adverb_embedding_input_size: int,
                 **kwargs):
        assert type_attention in ["bahdanau", "luong"], "Unknown type_attention=%s. Options are 'bahdanau', 'luong'"
        self.simplified_architecture = simplified_architecture
        if not self.simplified_architecture:
            self.input_specs = {
                "input_tensor": {
                    "data_type": "sequence",
                    "input_vocabulary_size": input_vocabulary_size,
                    "input_padding_idx": input_padding_idx
                },
                "adverb_input_tensor": {
                    "data_type": "sequence",
                    "input_vocabulary_size": adverb_input_vocabulary_size,
                    "input_padding_idx": adverb_input_padding_idx
                }
            }
            self.target_specs = {
                "adverb_target_tensor": {
                    "output_type": "sequence",
                    "attention": use_attention,
                    "type_attention": type_attention,
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": adverb_target_vocabulary_size,
                    "target_pad_idx": adverb_target_pad_idx,
                    "target_eos_idx": adverb_target_eos_idx,
                    "target_sos_idx": adverb_target_sos_idx
                }
            }
        else:
            self.input_specs = {
                "adverb_embedding": {
                    "data_type": "embedding",
                    "input_vocabulary_size": adverb_embedding_input_size,
                    "input_padding_idx": None
                },
                "adverb_input_tensor": {
                    "data_type": "sequence",
                    "input_vocabulary_size": adverb_input_vocabulary_size,
                    "input_padding_idx": adverb_input_padding_idx
                }
            }
            self.target_specs = {
                "adverb_target_tensor": {
                    "output_type": "sequence",
                    "attention": use_attention,
                    "type_attention": type_attention,
                    "conditional_attention": use_conditional_attention,
                    "target_vocabulary_size": adverb_target_vocabulary_size,
                    "target_pad_idx": adverb_target_pad_idx,
                    "target_eos_idx": adverb_target_eos_idx,
                    "target_sos_idx": adverb_target_sos_idx
                }
            }
        metric = "exact match"
        self.attention_values_key = attention_values_key
        self.conditional_attention_values_key = conditional_attention_values_key

        if not self.simplified_architecture:
            self.input_keys = []
            self.input_keys_to_pad = ["input_tensor", "adverb_input_tensor"]
        else:
            self.input_keys = ["adverb_embedding"]
            self.input_keys_to_pad = ["adverb_input_tensor"]
        self.target_keys = []
        self.target_keys_to_pad = ["adverb_target_tensor"]
        self.sequence_decoder = True
        self.use_attention = use_attention
        self.use_conditional_attention = use_conditional_attention
        self.type_attention = type_attention

        super().__init__(input_specs=self.input_specs, target_specs=self.target_specs, metric=metric,
                         embedding_dimension=embedding_dimension, encoder_hidden_size=encoder_hidden_size,
                         num_encoder_layers=num_encoder_layers, encoder_dropout_p=encoder_dropout_p,
                         encoder_bidirectional=encoder_bidirectional, output_directory=output_directory,
                         decoder_hidden_size=decoder_hidden_size, num_decoder_layers=num_decoder_layers,
                         decoder_dropout_p=decoder_dropout_p, target_vocabulary=target_vocabulary,
                         input_vocabulary=input_vocabulary, main_input_key=main_input_key)

        # Module specific combination parameters
        # self.encoder_hiddens_to_initial_hidden = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)
        # self.encoder_hiddens_to_initial_hidden = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        if self.simplified_architecture:
            self.adverb_to_dec_hidden = nn.Linear(embedding_dimension, decoder_hidden_size)

    def get_predictions(self, batch, max_decoding_steps, **kwargs):
        return predict_sequence(self, batch, max_decoding_steps)

    def get_metrics(self, adverb_target_tensor_scores: torch.Tensor, adverb_target_tensor_targets: torch.Tensor,
                    **kwargs) -> Dict[str, float]:
        """
        :param adverb_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param adverb_target_tensor_targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        accuracy, exact_match = self.get_exact_match(scores=adverb_target_tensor_scores,
                                                     targets=adverb_target_tensor_targets)
        return {
            "adverb_target_tensor_accuracy": accuracy,
            "adverb_target_tensor_exact_match": exact_match
        }

    def get_loss(self, adverb_target_tensor_scores: torch.Tensor, adverb_target_tensor_targets: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """
        :param adverb_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param adverb_target_tensor_targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        return self.get_sequence_loss(adverb_target_tensor_scores, adverb_target_tensor_targets)

    def prepare_encoder_output_decoder(self, encoder_output: Dict[str, Union[torch.Tensor, List[int], tuple]]):
        if not self.simplified_architecture:
            hidden_state_input_tensor = encoder_output["input_tensor"][0]
            hidden_state_adverb_input_tensor = encoder_output["adverb_input_tensor"][0]
            if self.use_attention:
                attention_values = encoder_output[self.attention_values_key][1]["encoder_outputs"]
                attention_values_lengths = encoder_output[self.attention_values_key][1]["sequence_lengths"]
            else:
                raise ValueError("No attention in adverb transform is not a good model in AdverbTransform")
                attention_values = None
                attention_values_lengths = None
            if self.use_conditional_attention:
                conditional_attention_values = encoder_output[self.conditional_attention_values_key][1]["encoder_outputs"]
                conditional_attention_values_lengths = encoder_output[self.conditional_attention_values_key][1]["sequence_lengths"]
            else:
                conditional_attention_values = None
                conditional_attention_values_lengths = None
            # initial_hidden = torch.cat([hidden_state_input_tensor,
            #                             hidden_state_adverb_input_tensor], dim=1)
            initial_hidden = hidden_state_input_tensor
            # initial_hidden = self.tanh(self.encoder_hiddens_to_initial_hidden(initial_hidden))
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
        else:
            adverb_embedding = encoder_output["adverb_embedding"]
            hidden_state_input_tensor = self.adverb_to_dec_hidden(adverb_embedding)
            if self.use_attention:
                attention_values = encoder_output[self.attention_values_key][1]["encoder_outputs"]
                attention_values_lengths = encoder_output[self.attention_values_key][1]["sequence_lengths"]
            else:
                raise ValueError("No attention in adverb transform is not a good model in AdverbTransform")
            initial_hidden = hidden_state_input_tensor
            target_key = self.target_keys_to_pad[0]
            initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
                initial_hidden)
        return {"hidden": initial_hidden,
                "attention_keys": attention_values,
                "attention_values": attention_values,
                "attention_values_lengths": attention_values_lengths,
                "conditional_attention_values": None,
                "conditional_attention_values_lengths": None}

    def forward(self, batch, **kwargs) -> Dict[
        str, torch.Tensor]:
        encoder_output = self.encode_input(batch["inputs"])
        decoder_output = self.decode_input_batched(batch["targets"], encoder_output)
        # decoder_output_batched: [max_target_length, batch_size, output_size]
        # attention_weights: [batch_size, max_target_length, max_input_length]
        return {
            "adverb_target_tensor_scores": decoder_output["decoder_output_batched"].transpose(0, 1),
            "decoder_attention_weights": decoder_output["attention_weights"]
        }


class AloTransform(SuperModule):

    def __init__(self, embedding_dimension: int, encoder_hidden_size: int, target_vocabulary: str,
                 final_input_vocabulary_size: int, final_input_padding_idx: int, input_vocabulary: str,
                 num_encoder_layers: int, encoder_dropout_p: float, encoder_bidirectional: bool,
                 simplified_architecture: bool, input_vocabulary_size: int, input_padding_idx: int,
                 output_directory: str, decoder_hidden_size: int, target_pad_idx: int, target_sos_idx: int,
                 target_vocabulary_size: int, num_decoder_layers: int, decoder_dropout_p: float, target_eos_idx: int,
                 type_attention: str, use_attention: bool, use_conditional_attention: bool,
                 attention_values_key: str, conditional_attention_values_key: str, main_input_key: str, **kwargs):
        self.simplified_architecture = simplified_architecture
        assert type_attention in ["bahdanau", "luong"], "Unknown type_attention=%s. Options are 'bahdanau', 'luong'"
        assert not use_conditional_attention, "Cannot use conditional attention in AloTransform"
        self.input_specs = {
            "agent_direction": {
                "data_type": "int",
            },
            "final_input_tensor": {
                "data_type": "sequence",
                "input_vocabulary_size": final_input_vocabulary_size,
                "input_padding_idx": final_input_padding_idx
            }
        }
        self.target_specs = {
            "target_tensor": {
                "output_type": "sequence",
                "attention": use_attention,
                "type_attention": type_attention,
                "conditional_attention": use_conditional_attention,
                "target_vocabulary_size": target_vocabulary_size,
                "target_pad_idx": target_pad_idx,
                "target_eos_idx": target_eos_idx,
                "target_sos_idx": target_sos_idx
            }
        }
        metric = "exact match"
        self.attention_values_key = attention_values_key
        self.conditional_attention_values_key = conditional_attention_values_key

        self.input_keys = ["agent_direction"]
        self.input_keys_to_pad = ["adverb_target_tensor"]
        self.target_keys = []
        self.target_keys_to_pad = ["target_tensor"]
        self.sequence_decoder = True
        self.use_attention = use_attention
        self.use_conditional_attention = use_conditional_attention
        self.type_attention = type_attention

        super().__init__(input_specs=self.input_specs, target_specs=self.target_specs, metric=metric,
                         embedding_dimension=embedding_dimension, encoder_hidden_size=encoder_hidden_size,
                         num_encoder_layers=num_encoder_layers, encoder_dropout_p=encoder_dropout_p,
                         encoder_bidirectional=encoder_bidirectional, output_directory=output_directory,
                         decoder_hidden_size=decoder_hidden_size, num_decoder_layers=num_decoder_layers,
                         decoder_dropout_p=decoder_dropout_p, target_vocabulary=target_vocabulary,
                         input_vocabulary=input_vocabulary, main_input_key=main_input_key)

        # Module specific combination parameters
        self.encoder_hiddens_to_initial_hidden = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)

    def get_predictions(self, batch, max_decoding_steps, **kwargs):
        return predict_sequence(self, batch, max_decoding_steps)

    def get_metrics(self, target_tensor_scores: torch.Tensor, target_tensor_targets: torch.Tensor,
                    **kwargs) -> Dict[str, float]:
        """
        :param target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param target_tensor_targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        accuracy, exact_match = self.get_exact_match(scores=target_tensor_scores,
                                                     targets=target_tensor_targets)
        return {
            "target_tensor_accuracy": accuracy,
            "target_tensor_exact_match": exact_match
        }

    def get_loss(self, target_tensor_scores: torch.Tensor, target_tensor_targets: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """
        :param target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param target_tensor_targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        return self.get_sequence_loss(target_tensor_scores, target_tensor_targets)

    def prepare_encoder_output_decoder(self, encoder_output: Dict[str, Union[torch.Tensor, List[int], tuple]]):

        hidden_state_adverb_target_tensor = encoder_output["final_input_tensor"][0]
        encoded_direction = encoder_output["mlp"]
        if self.use_attention:
            attention_values = encoder_output[self.attention_values_key][1]["encoder_outputs"]
            attention_values_lengths = encoder_output[self.attention_values_key][1]["sequence_lengths"]
        else:
            attention_values = None
            attention_values_lengths = None
        initial_hidden = torch.cat([encoded_direction, hidden_state_adverb_target_tensor], dim=1)
        initial_hidden = self.tanh(self.encoder_hiddens_to_initial_hidden(initial_hidden))
        target_key = self.target_keys_to_pad[0]
        initial_hidden = self.decoder_module_dict[target_key].initialize_hidden(
            initial_hidden)
        return {"hidden": initial_hidden,
                "attention_keys": attention_values,
                "attention_values": attention_values,
                "attention_values_lengths": attention_values_lengths}

    def forward(self, batch, **kwargs) -> Dict[str, torch.Tensor]:
        encoder_output = self.encode_input(batch["inputs"])
        decoder_output = self.decode_input_batched(batch["targets"], encoder_output)
        # decoder_output_batched: [max_target_length, batch_size, output_size]
        # attention_weights: [batch_size, max_target_length, max_input_length]
        return {
            "target_tensor_scores": decoder_output["decoder_output_batched"].transpose(0, 1),
            "decoder_attention_weights": decoder_output["attention_weights"]
        }


class FullModel(nn.Module):
    """

    """
    def __init__(self, dataset, input_vocabulary: Vocabulary, num_cnn_channels: int, module_path_pattern: str,
                 planner_target_vocabulary: Vocabulary, adverb_input_vocabulary: Vocabulary,
                 adverb_target_vocabulary: Vocabulary, target_vocabulary: Vocabulary, checkpoints_per_module,
                 collapse_alo, **kwargs):
        super(FullModel, self).__init__()
        self.dataset = dataset
        for key, value in checkpoints_per_module["position"].items():
            if key in kwargs:
                kwargs[key] = value
        kwargs["decoder_hidden_size"] = kwargs["encoder_hidden_size"]
        self.position_module = PositionModel(input_vocabulary_size=self.dataset.full_vocabularies["position"]["input_tensor"].size,
                                             num_cnn_channels=num_cnn_channels,
                                             input_padding_idx=self.dataset.full_vocabularies["position"]["input_tensor"].pad_idx,
                                             target_vocabulary="",
                                             input_vocabulary="input",
                                             main_input_key="input_tensor",
                                             **kwargs)
        for key, value in checkpoints_per_module["planner"].items():
            if key in kwargs:
                kwargs[key] = value
        self.planner_module = PlannerModel(input_vocabulary_size=self.dataset.full_vocabularies["planner"]["input_tensor"].size,
                                           input_padding_idx=self.dataset.full_vocabularies["planner"]["input_tensor"].pad_idx,
                                           planner_target_eos_idx=self.dataset.full_vocabularies["planner"]["planner_target_tensor"].eos_idx,
                                           planner_target_pad_idx=self.dataset.full_vocabularies["planner"]["planner_target_tensor"].pad_idx,
                                           planner_target_sos_idx=self.dataset.full_vocabularies["planner"]["planner_target_tensor"].sos_idx,
                                           planner_target_vocabulary_size=self.dataset.full_vocabularies["planner"]["planner_target_tensor"].size,
                                           target_vocabulary="planner_target",
                                           input_vocabulary="input",
                                           main_input_key="input_tensor",
                                           **kwargs)
        for key, value in checkpoints_per_module["interaction"].items():
            if key in kwargs:
                kwargs[key] = value
        self.interaction_module = NewInteractionModel(input_vocabulary_size=self.dataset.full_vocabularies["interaction"]["input_tensor"].size,
                                                      transitive_input_vocabulary_size=self.dataset.full_vocabularies["planner"]["planner_target_tensor"].size,
                                                      transitive_input_padding_idx=self.dataset.full_vocabularies["planner"]["planner_target_tensor"].pad_idx,
                                                      num_cnn_channels=num_cnn_channels,
                                                      input_padding_idx=self.dataset.full_vocabularies["interaction"]["input_tensor"].pad_idx,
                                                      transitive_target_eos_idx=self.dataset.full_vocabularies["interaction"]["transitive_target_tensor"].eos_idx,
                                                      transitive_target_pad_idx=self.dataset.full_vocabularies["interaction"]["transitive_target_tensor"].pad_idx,
                                                      transitive_target_sos_idx=self.dataset.full_vocabularies["interaction"]["transitive_target_tensor"].sos_idx,
                                                      transitive_target_vocabulary_size=self.dataset.full_vocabularies["interaction"]["transitive_target_tensor"].size,
                                                      target_vocabulary="transitive_target",
                                                      input_vocabulary="input",
                                                      main_input_key="input_tensor",
                                                      **kwargs)
        for key, value in checkpoints_per_module["adverb_transform"].items():
            logger.info("Key: %s, Value: %s" % (key, value))
            if key in kwargs:
                kwargs[key] = value
        kwargs["decoder_hidden_size"] = kwargs["encoder_hidden_size"]
        self.adverb_module = AdverbTransform(input_vocabulary_size=self.dataset.full_vocabularies["adverb_transform"]["input_tensor"].size,
                                             adverb_embedding_input_size=kwargs["embedding_dimension"],
                                             adverb_input_vocabulary_size=self.dataset.full_vocabularies["adverb_transform"]["adverb_input_tensor"].size,
                                             adverb_input_padding_idx=self.dataset.full_vocabularies["adverb_transform"]["adverb_input_tensor"].pad_idx,
                                             input_padding_idx=self.dataset.full_vocabularies["adverb_transform"]["input_tensor"].pad_idx,
                                             adverb_target_eos_idx=self.dataset.full_vocabularies["adverb_transform"]["adverb_target_tensor"].eos_idx,
                                             adverb_target_pad_idx=self.dataset.full_vocabularies["adverb_transform"]["adverb_target_tensor"].pad_idx,
                                             adverb_target_sos_idx=self.dataset.full_vocabularies["adverb_transform"]["adverb_target_tensor"].sos_idx,
                                             adverb_target_vocabulary_size=self.dataset.full_vocabularies["adverb_transform"]["adverb_target_tensor"].size,
                                             target_vocabulary="adverb_target",
                                             input_vocabulary="adverb_input",
                                             main_input_key="adverb_input_tensor",
                                             **kwargs)
        self.collapse_alo = collapse_alo
        if not collapse_alo:
            for key, value in checkpoints_per_module["alo_transform"].items():
                if key in kwargs:
                    kwargs[key] = value
            kwargs["decoder_hidden_size"] = kwargs["encoder_hidden_size"]
            self.alo_transform = AloTransform(final_input_vocabulary_size=self.dataset.full_vocabularies["alo_transform"]["adverb_target_tensor"].size,
                                              final_input_padding_idx=self.dataset.full_vocabularies["alo_transform"]["adverb_target_tensor"].pad_idx,
                                              target_eos_idx=self.dataset.full_vocabularies["alo_transform"]["target_tensor"].eos_idx,
                                              target_pad_idx=self.dataset.full_vocabularies["alo_transform"]["target_tensor"].pad_idx,
                                              target_sos_idx=self.dataset.full_vocabularies["alo_transform"]["target_tensor"].sos_idx,
                                              target_vocabulary_size=self.dataset.full_vocabularies["alo_transform"]["target_tensor"].size,
                                              target_vocabulary="target",
                                              input_vocabulary="adverb_target",
                                              main_input_key="final_input_tensor",
                                              **kwargs)

        self.modules = {"position": self.position_module,
                        "planner": self.planner_module,
                        "interaction": self.interaction_module,
                        "adverb_transform": self.adverb_module}
        if not self.collapse_alo:
            self.modules["alo_transform"] = self.alo_transform

        self.module_input_keys = {"position": "input_tensor",
                                  "planner": "input_tensor",
                                  "interaction": "transitive_input_tensor",
                                  "adverb_transform": "adverb_input_tensor"}
        if not self.collapse_alo:
            self.module_input_keys["alo_transform"] = "final_input_tensor"

        self.output_to_input_translate = {
            "agent_direction": {"new_input_key": "agent_direction",
                                "prediction_module": "position",
                                "prediction_key": "agent_direction",
                                "concatenate": False,
                                "concatenate_with_module": "",
                                "concatenate_with_key": ""},
            "agent_position": {"new_input_key": "agent_position",
                               "prediction_module": "position",
                               "prediction_key": "agent_position",
                               "concatenate": False,
                               "concatenate_with_module": "",
                               "concatenate_with_key": ""},
            "target_position": {"new_input_key": "target_position",
                                "prediction_module": "position",
                                "prediction_key": "target_position",
                                "concatenate": False,
                                "concatenate_with_module": "",
                                "concatenate_with_key": ""},
            "transitive_target_tensor": {"new_input_key": "adverb_input_tensor",
                                         "prediction_module": "planner",
                                         "prediction_key": "planner_target_tensor",
                                         "concatenate": True,
                                         "concatenate_with_module": "interaction",
                                         "concatenate_with_key": "transitive_target_tensor"},
            "transitive_target_tensor_lengths": "adverb_input_tensor_lengths",
            "adverb_target_tensor": {"new_input_key": "final_input_tensor",
                                     "prediction_module": "adverb_transform",
                                     "prediction_key": "adverb_target_tensor",
                                     "concatenate": False,
                                     "concatenate_with_module": "",
                                     "concatenate_with_key": ""},
            "adverb_target_tensor_lengths": "final_input_tensor_lengths",
            "planner_target_tensor": {"new_input_key": "target_position",
                                      "prediction_module": "position",
                                      "prediction_key": "target_position",
                                      "concatenate": False,
                                      "concatenate_with_module": "",
                                      "concatenate_with_key": ""},
            "planner_target_tensor_lengths": "transitive_input_tensor_lengths",
        }

        for module_name, module in self.modules.items():
            if module_name != "adverb_transform":
                module_path_pattern_split = module_path_pattern.split("seed_")
                module_path_pattern_seed = int(module_path_pattern_split[-1]) % 5
                if not module_path_pattern_seed:
                    module_path_pattern_seed = 5
                adjusted_module_path_pattern = module_path_pattern_split[0] + "seed_" + str(module_path_pattern_seed)
                module_resume_from_file = os.path.join(adjusted_module_path_pattern % module_name, "model_best.pth.tar")
            else:
                module_resume_from_file = os.path.join(module_path_pattern % module_name, "model_best.pth.tar")
            logger.info("Loading checkpoint from file at '{}'".format(module_resume_from_file))
            module.load_model(module_resume_from_file)
            module = module.to(device)
            start_iteration = module.trained_iterations
            logger.info("Loaded checkpoint '{}' (iter {})".format(module_resume_from_file, start_iteration))
            self.modules[module_name] = module

        self.input_keys = {module_name: module.input_keys for module_name, module in self.modules.items()}
        self.input_keys_to_pad = {module_name: module.input_keys_to_pad for module_name, module in self.modules.items()}
        self.target_keys = {module_name: module.target_keys for module_name, module in self.modules.items()}
        self.target_keys_to_pad = {module_name: module.target_keys_to_pad for module_name, module in self.modules.items()}
        self.sequence_decoder = False
        self.full_model = True
        # self.sequence_decoders = {module_name: module.sequence_decoder for module_name, module in self.modules.items()}

    def translate_to_vocab(self, sequence, from_vocab: str, to_vocab: str, from_module: str, to_module: str):
        dtype = sequence.dtype
        str_sequence = self.dataset.array_to_sentence(sequence.squeeze(), vocabulary=from_vocab, module=from_module)
        return torch.tensor(self.dataset.sentence_to_array(str_sequence[1:-1], vocabulary=to_vocab,
                                                           module=to_module),
                            device=device, dtype=dtype).unsqueeze(dim=0)

    @staticmethod
    def is_special_ending_sequence(sequence: torch.Tensor) -> bool:
        # The output sequence a module should produce if it shouldn't change the input.
        special_ending_tokens = torch.tensor([[1, 2]], dtype=torch.long, device=device)
        if sequence.shape == special_ending_tokens.shape:
            if torch.all(sequence == special_ending_tokens):
                return True
        return False

    def forward(self, batch, max_decoding_steps, gold_forward_pass=True, **kwargs):

        # The output sequence a module should produce if it shouldn't change the input.
        special_ending_tokens = torch.tensor([[1, 2]], dtype=torch.long, device=device)

        # Initialize some trackers
        previous_module = None
        all_predictions = {}
        predictions_per_target_key = {}
        final_predictions = None
        final_predictions_vocabulary = None
        final_predictions_module = None
        final_module = "alo_transform" if not self.collapse_alo else "adverb_transform"
        final_target_vocabulary = "target" if not self.collapse_alo else "adverb_target"

        # A running batch of examples that is updated with the predictions of the previous module for every new module.
        running_batch = copy.deepcopy(batch)

        # Loop over all the modules.
        for module_name, module in self.modules.items():

            # Translate the inputs from the master vocabulary to the module-specific vocabularies.
            for input_key_to_pad in self.input_keys_to_pad[module_name]:
                input_sequence = running_batch[module_name]["inputs"][input_key_to_pad]
                if not len(input_sequence):
                    continue
                from_vocab = input_key_to_pad.split("_tensor")[0]
                to_vocab = from_vocab  # The translation here is for the same vocabulary but between different modules.
                running_batch[module_name]["inputs"][input_key_to_pad] = self.translate_to_vocab(input_sequence,
                                                                                                 from_vocab,
                                                                                                 to_vocab,
                                                                                                 from_module="",
                                                                                                 to_module=module_name)
                original_input_sequence = batch[module_name]["inputs"][input_key_to_pad]
                batch[module_name]["inputs"][input_key_to_pad] = self.translate_to_vocab(original_input_sequence,
                                                                                         from_vocab,
                                                                                         to_vocab,
                                                                                         from_module="",
                                                                                         to_module=module_name)

            # Replace ground-truth input with predictions by previous module.
            for target_key, prediction in predictions_per_target_key.items():

                # Get the dict that specifies which inputs need to be translated from previous to next module.
                input_info = self.output_to_input_translate[target_key]
                if "lengths" not in target_key:  # Only translate predictions and not length tensors.
                    assert input_info["new_input_key"] in running_batch[module_name]["inputs"].keys(), \
                        "Input key %s not part of input batch for module %s" % (input_info["new_input_key"], module_name)

                    # If the previous module predicted special ending tokens, copy the input from the previous module.
                    prediction = all_predictions[input_info["prediction_module"]]["predictions"][input_info["prediction_key"]]
                    if "_tensor" in input_info["prediction_key"]:
                        from_vocab = input_info["prediction_key"].split("_tensor")[0]
                        to_vocab = input_info["new_input_key"].split("_tensor")[0]
                        prediction = self.translate_to_vocab(prediction, from_vocab, to_vocab,
                                                             from_module=input_info["prediction_module"],
                                                             to_module=module_name)
                    if input_info["concatenate"]:  # E.g., for the adverb module concat planner + interaction output
                        pred2 = all_predictions[input_info["concatenate_with_module"]]["predictions"][input_info["concatenate_with_key"]]
                        if "_tensor" in input_info["prediction_key"]:
                            from_vocab = input_info["concatenate_with_key"].split("_tensor")[0]
                            to_vocab = input_info["new_input_key"].split("_tensor")[0]
                            pred2 = self.translate_to_vocab(pred2, from_vocab, to_vocab,
                                                            from_module=input_info["concatenate_with_module"],
                                                            to_module=module_name)
                        prediction = torch.cat([prediction[0, :-1], pred2[0, 1:]]).unsqueeze(dim=0)  # Concatenate without extra <SOS> and <EOS>
                    if self.is_special_ending_sequence(prediction):
                        prediction = running_batch[previous_module]["inputs"][self.module_input_keys[previous_module]]
                        from_vocab = self.module_input_keys[previous_module].split("_tensor")[0]
                        to_vocab = self.module_input_keys[module_name].split("_tensor")[0]
                        prediction = self.translate_to_vocab(prediction, from_vocab, to_vocab,
                                                             from_module=previous_module,
                                                             to_module=module_name)
                    else:
                        final_predictions = prediction
                        final_predictions_vocabulary = to_vocab
                        final_predictions_module = module_name
                        if input_info["prediction_module"] != "position":
                            final_predictions_translated = self.translate_to_vocab(prediction, to_vocab, final_target_vocabulary,
                                                                                   from_module=module_name,
                                                                                   to_module=final_module)
                    running_batch[module_name]["inputs"][input_info["new_input_key"]] = prediction
                    if input_info["new_input_key"] + "_lengths" in running_batch[module_name]["inputs"]:
                        running_batch[module_name]["inputs"][input_info["new_input_key"] + "_lengths"] = [prediction.size(1)]

            if module.sequence_decoder:
                predictions_batch, attn_w = module.get_predictions(running_batch[module_name], max_decoding_steps)
                if gold_forward_pass:
                    ground_truth_input_predictions_batch, non_full_attn_w = module.get_predictions(batch[module_name],
                                                                                                   max_decoding_steps)
                target_name = module.target_keys_to_pad[0]
                sequence_preds = predictions_batch[target_name + "_sequences"]
                sequence_pred_lengths = predictions_batch[target_name + "_sequence_lengths"]
                preds = module.add_start_of_sequence(sequence_preds)
                predictions_per_target_key = {
                    target_name: preds[0, :(sequence_pred_lengths + 1).item()].unsqueeze(dim=0),
                    target_name + "_lengths": (sequence_pred_lengths + 1).unsqueeze(dim=0).cpu().numpy()
                }
                if gold_forward_pass:
                    non_full_sequence_preds = ground_truth_input_predictions_batch[target_name + "_sequences"]
                    non_full_sequence_pred_lengths = ground_truth_input_predictions_batch[target_name + "_sequence_lengths"]
                target_lengths = torch.tensor(running_batch[module_name]["targets"]["%s_lengths" % target_name],
                                              device=device,
                                              dtype=sequence_pred_lengths.dtype) - 1  # -1 because SOS gets removed
                original_targets = running_batch[module_name]["targets"]["%s_targets" % target_name]
                if len(original_targets):
                    from_vocab = target_name.split("_tensor")[0]
                    to_vocab = from_vocab
                    translated_targets = self.translate_to_vocab(original_targets,
                                                                 from_vocab,
                                                                 to_vocab,
                                                                 from_module="",
                                                                 to_module=module_name)
                    sequence_targets = module.remove_start_of_sequence(translated_targets)
                    accuracy_per_sequence, exact_match_per_sequence = get_exact_match(sequence_preds, sequence_pred_lengths,
                                                                                      sequence_targets, target_lengths)
                    metrics = {target_name: {"accuracy": float(accuracy_per_sequence.item()),
                                             "exact_match": float(exact_match_per_sequence.item())}}
                if gold_forward_pass:
                    non_full_accuracy_per_sequence, non_full_exact_match_per_sequence = get_exact_match(
                        non_full_sequence_preds, non_full_sequence_pred_lengths, sequence_targets, target_lengths)
                    ground_truth_input_metrics = {
                        target_name: {"accuracy": float(non_full_accuracy_per_sequence.item()),
                                      "exact_match": float(non_full_exact_match_per_sequence.item())}}
            else:
                predictions_batch = module.get_predictions(running_batch[module_name])
                ground_truth_input_predictions_batch = module.get_predictions(batch[module_name])
                metrics = {}
                ground_truth_input_metrics = {}
                for target_name in module.target_keys:
                    equal = torch.eq(running_batch[module_name]["targets"]["%s_targets" % target_name].data,
                                     predictions_batch[target_name].data).long().sum().data.item()
                    total = len(
                        running_batch[module_name]["targets"]["%s_targets" % target_name])
                    accuracy = (equal / total) * 1.
                    metrics[target_name] = accuracy
                    ground_truth_input_equal = torch.eq(
                        running_batch[module_name]["targets"]["%s_targets" % target_name].data,
                        ground_truth_input_predictions_batch[target_name].data).long().sum().data.item()
                    ground_truth_input_metrics[target_name] = (ground_truth_input_equal / total) * 1.
                predictions_per_target_key = predictions_batch

            previous_module = module_name
            all_predictions[module_name] = {}
            all_predictions[module_name]["original_batch"] = batch[module_name]
            all_predictions[module_name]["predictions"] = predictions_per_target_key
            all_predictions[module_name]["metrics"] = metrics
            all_predictions[module_name]["ground_truth_metrics"] = ground_truth_input_metrics

        # Also update the final predictions for the last module
        for key, prediction in predictions_per_target_key.items():
            if "lengths" not in key:
                if self.is_special_ending_sequence(prediction):
                    prediction = running_batch[previous_module]["inputs"][self.module_input_keys[previous_module]]
                else:
                    final_predictions = prediction
                    final_predictions_vocabulary = "target" if not self.collapse_alo else "adverb_target"
                    final_predictions_module = module_name
                    if previous_module != "position":
                        final_predictions_translated = self.translate_to_vocab(prediction, to_vocab,
                                                                               final_target_vocabulary,
                                                                               from_module=module_name,
                                                                               to_module=final_module)

        final_targets = batch["position"]["extra_information"][0]["example_information"]["target_command"]
        final_targets = torch.tensor(self.dataset.sentence_to_array(final_targets, final_predictions_vocabulary,
                                                                    final_predictions_module),
                                     device=device,
                                     dtype=final_predictions.dtype).unsqueeze(dim=0)
        sequence_preds = final_predictions
        if final_predictions.size(-1) < final_targets.size(-1):
            padding_preds = torch.zeros([final_targets.size(0),
                                         final_targets.size(-1) - final_predictions.size(-1)], dtype=torch.long,
                                         device=device)
            sequence_preds = torch.cat([sequence_preds, padding_preds], dim=1)
        sequence_pred_lengths = torch.tensor(final_predictions.size(-1), device=device,
                                             dtype=sequence_pred_lengths.dtype)
        target_lengths = torch.tensor(final_targets.size(-1), device=device,
                                      dtype=sequence_pred_lengths.dtype)
        sequence_targets = final_targets
        full_accuracy, full_exact_match = get_exact_match(sequence_preds, sequence_pred_lengths,
                                                          sequence_targets, target_lengths)
        full_prediction = sequence_preds
        full_target = sequence_targets
        return (all_predictions, full_accuracy.squeeze().item(), full_exact_match.squeeze().item(), full_prediction,
                full_target, final_predictions_vocabulary, final_predictions_module)