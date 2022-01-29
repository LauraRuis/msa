import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Tuple, Union
import os
import shutil

from seq2seq.seq2seq_model import EncoderRNN, Attention
from Modularity.nn import MLP, LuongDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


class PlannerModel(nn.Module):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, encoder_dropout_p: float, encoder_bidirectional: bool, input_padding_idx: int,
                 output_directory: str, decoder_hidden_size: int, target_pad_idx: int, target_sos_idx: int,
                 target_vocabulary_size: int, num_decoder_layers: int, decoder_dropout_p: float, target_eos_idx: int,
                 **kwargs):
        super(PlannerModel, self).__init__()

        self.loss_criterion = nn.NLLLoss(ignore_index=target_pad_idx)

        self.input_to_position = nn.Linear(3, encoder_hidden_size)
        self.encode_position = MLP(input_dim=encoder_hidden_size, hidden_dim=encoder_hidden_size,
                                   dropout_p=encoder_dropout_p)

        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.encoder = EncoderRNN(input_size=input_vocabulary_size,
                                  embedding_dim=embedding_dimension,
                                  rnn_input_size=embedding_dimension,
                                  hidden_size=encoder_hidden_size, num_layers=num_encoder_layers,
                                  dropout_probability=encoder_dropout_p, bidirectional=encoder_bidirectional,
                                  padding_idx=input_padding_idx)

        # Used to project the final encoder state to the decoder hidden state such that it can be initialized with it.
        self.enc_hidden_to_dec_hidden = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)

        # Input: [batch_size, max_target_length], initial hidden: ([batch_size, hidden_size], [batch_size, hidden_size])
        # Input for attention: [batch_size, max_input_length, hidden_size],
        #                      [batch_size, image_width * image_width, hidden_size]
        # Output: [max_target_length, batch_size, target_vocabulary_size]
        self.attention_decoder = LuongDecoder(hidden_size=decoder_hidden_size,
                                              output_size=target_vocabulary_size,
                                              num_layers=num_decoder_layers,
                                              dropout_probability=decoder_dropout_p,
                                              padding_idx=target_pad_idx,
                                              conditional_attention=False)
        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.target_sos_idx = target_sos_idx
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

    @staticmethod
    def remove_start_of_sequence(input_tensor: torch.Tensor) -> torch.Tensor:
        """Get rid of SOS-tokens in targets batch and append a padding token to each example in the batch."""
        batch_size, max_time = input_tensor.size()
        input_tensor = input_tensor[:, 1:]
        output_tensor = torch.cat([input_tensor, torch.zeros(batch_size, device=device, dtype=torch.long).unsqueeze(
            dim=1)], dim=1)
        return output_tensor

    def get_metrics(self, planner_target_tensor_scores: torch.Tensor, planner_target_tensor_targets: torch.Tensor,
                    **kwargs) -> Dict[str, float]:
        """
        :param planner_target_tensor_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param planner_target_tensor_targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            targets = self.remove_start_of_sequence(planner_target_tensor_targets)
            mask = (targets != self.target_pad_idx).long()
            total = mask.sum().data.item()
            predicted_targets = planner_target_tensor_scores.max(dim=2)[1]
            equal_targets = torch.eq(targets.data, predicted_targets.data).long()
            match_targets = (equal_targets * mask)
            match_sum_per_example = match_targets.sum(dim=1)
            expected_sum_per_example = mask.sum(dim=1)
            batch_size = expected_sum_per_example.size(0)
            exact_match = 100. * (match_sum_per_example == expected_sum_per_example).sum().data.item() / batch_size
            match_targets_sum = match_targets.sum().data.item()
            accuracy = 100. * match_targets_sum / total
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
        targets = self.remove_start_of_sequence(planner_target_tensor_targets)

        # Calculate the loss.
        vocabulary_size = planner_target_tensor_scores.size(2)
        target_scores_2d = planner_target_tensor_scores.reshape(-1, vocabulary_size)
        loss = self.loss_criterion(target_scores_2d, targets.view(-1))
        return loss

    def prepare_inputs(self, agent_position: torch.Tensor, agent_direction: torch.Tensor,
                       target_position: torch.Tensor):
        input_tensor = torch.cat([agent_direction.unsqueeze(dim=1),
                                  agent_position.unsqueeze(dim=1),
                                  target_position.unsqueeze(dim=1)], dim=1)
        return input_tensor

    def encode_input(self, input_tensor: torch.LongTensor, input_tensor_lengths: List[int],
                     agent_position: torch.Tensor, agent_direction: torch.Tensor,
                     target_position: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[int]]]:
        """Pass the input commands through an RNN encoder and the other input through an MLP."""
        position_inputs = self.prepare_inputs(agent_position, agent_direction, target_position)
        position_inputs = self.input_to_position(position_inputs.float())
        encoded_positions = self.encode_position(position_inputs)
        hidden, encoder_outputs = self.encoder(input_tensor, input_tensor_lengths)
        initial_hidden = torch.cat([hidden, encoded_positions], dim=1)
        return {"encoded_commands": encoder_outputs, "hidden_states": hidden,
                "encoded_positions": encoded_positions, "initial_hidden": initial_hidden,
                "attention_values": encoder_outputs["encoder_outputs"],
                "attention_values_lengths": encoder_outputs["sequence_lengths"]}

    def decode_step(self, planner_target_tensor: torch.LongTensor, hidden: Tuple[torch.Tensor, torch.Tensor],
                    attention_values: torch.Tensor,
                    attention_values_lengths: List[int], **kwargs) -> Tuple[torch.Tensor,
                                                                            Tuple[torch.Tensor, torch.Tensor],
                                                                            torch.Tensor]:
        """One decoding step based on the previous hidden state of the decoder and the previous target token."""
        return self.attention_decoder.forward_step(target_tokens=planner_target_tensor, last_hidden=hidden,
                                                   attention_keys=attention_values,
                                                   attention_values_lengths=attention_values_lengths,
                                                   attention_values=attention_values)

    def decode_input_batched(self, planner_target_tensor: torch.LongTensor, planner_target_tensor_lengths: List[int],
                             initial_hidden: torch.Tensor, attention_values: torch.Tensor,
                             attention_values_lengths: List[int], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode a batch of input sequences."""
        initial_hidden = self.attention_decoder.initialize_hidden(
            self.tanh(self.enc_hidden_to_dec_hidden(initial_hidden)))
        decoder_output_batched, _, attention_weights = self.attention_decoder(
            target_tokens=planner_target_tensor, target_lengths=planner_target_tensor_lengths,
            init_hidden=initial_hidden, attention_keys=attention_values, attention_values=attention_values,
            attention_values_lengths=attention_values_lengths)
        decoder_output_batched = F.log_softmax(decoder_output_batched, dim=-1)
        return decoder_output_batched, attention_weights

    def forward(self, input_tensor: torch.LongTensor, input_tensor_lengths: List[int], agent_position: torch.Tensor,
                agent_direction: torch.Tensor, target_position: torch.Tensor, planner_target_tensor: torch.LongTensor,
                planner_target_tensor_lengths: List[int]) -> Dict[str, torch.Tensor]:
        encoder_output = self.encode_input(input_tensor=input_tensor, input_tensor_lengths=input_tensor_lengths,
                                           agent_position=agent_position, agent_direction=agent_direction,
                                           target_position=target_position)
        decoder_output, attention_weights = self.decode_input_batched(
            planner_target_tensor=planner_target_tensor, planner_target_tensor_lengths=planner_target_tensor_lengths,
            initial_hidden=encoder_output["initial_hidden"],
            attention_values=encoder_output["attention_values"],
            attention_values_lengths=encoder_output["attention_values_lengths"])
        # decoder_output: [max_target_length, batch_size, output_size]
        # attention_weights: [batch_size, max_target_length, max_input_length]
        return {
            "planner_target_tensor_scores": decoder_output.transpose(0, 1),
            "decoder_attention_weights": attention_weights
        }

    def update_state(self, is_best: bool, metric=None) -> {}:
        self.trained_iterations += 1
        if is_best:
            self.best_metric = metric
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_metric = checkpoint["best_metric"]
        self.metric = "exact match"
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_metric": self.best_metric,
            "metric": "exact match"
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
