import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List
from typing import Dict
from typing import Tuple
import os
import shutil

from seq2seq.cnn_model import ConvolutionalNet, DownSamplingConvolutionalNet
from seq2seq.seq2seq_model import EncoderRNN, Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


class PositionModel(nn.Module):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, encoder_dropout_p: float,
                 encoder_bidirectional: bool, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int,
                 output_directory: str, grid_size: int,
                 simple_situation_representation: bool, **kwargs):
        super(PositionModel, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        # Attention over the output features of the ConvolutionalNet.
        # Input: [bsz, 1, decoder_hidden_size], [bsz, image_width * image_width, cnn_hidden_num_channels * 3]
        # Output: [bsz, 1, decoder_hidden_size], [bsz, 1, image_width * image_width]
        self.agent_attention = Attention(key_size=cnn_hidden_num_channels * 3, query_size=encoder_hidden_size,
                                         hidden_size=encoder_hidden_size)

        self.target_attention = Attention(key_size=cnn_hidden_num_channels * 3, query_size=encoder_hidden_size,
                                          hidden_size=encoder_hidden_size)

        self.hidden_to_hidden = nn.Linear(cnn_hidden_num_channels * 3 + encoder_hidden_size, encoder_hidden_size)
        self.hidden_to_output = nn.Linear(encoder_hidden_size, grid_size*grid_size)

        self.loss_criterion = nn.NLLLoss()

        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.encoder = EncoderRNN(input_size=input_vocabulary_size,
                                  embedding_dim=embedding_dimension,
                                  rnn_input_size=embedding_dimension,
                                  hidden_size=encoder_hidden_size, num_layers=num_encoder_layers,
                                  dropout_probability=encoder_dropout_p, bidirectional=encoder_bidirectional,
                                  padding_idx=input_padding_idx)

        self.situation_to_directions = nn.Linear(cnn_hidden_num_channels * 3, 4)
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_metric = 0
        self.metric = "accuracy"

        self.input_keys = ["world_state_tensor"]
        self.input_keys_to_pad = ["input_tensor"]
        self.target_keys = ["agent_position", "target_position", "agent_direction"]
        self.target_keys_to_pad = []
        self.sequence_decoder = False

    @staticmethod
    def get_single_accuracy(target_scores: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            predicted_targets = target_scores.max(dim=1)[1]
            equal_targets = torch.eq(targets.data, predicted_targets.data).long().sum().data.item()
            accuracy = 100. * equal_targets / len(targets)
        return accuracy

    def get_metrics(self, agent_direction_scores: torch.Tensor, agent_position_scores: torch.Tensor,
                    target_position_scores: torch.Tensor, agent_position_targets: torch.Tensor,
                    agent_direction_targets: torch.Tensor, target_position_targets: torch.Tensor):
        direction_accuracy = self.get_single_accuracy(agent_direction_scores, agent_direction_targets)
        agent_position_accuracy = self.get_single_accuracy(agent_position_scores, agent_position_targets)
        target_position_accuracy = self.get_single_accuracy(target_position_scores, target_position_targets)
        return {"agent_direction_accuracy": direction_accuracy,
                "agent_position_accuracy": agent_position_accuracy,
                "target_position_accuracy": target_position_accuracy}

    def get_single_loss(self, scores: torch.Tensor, targets: torch.Tensor):
        loss = self.loss_criterion(scores, targets.view(-1))
        return loss

    def get_loss(self, agent_direction_scores: torch.Tensor, agent_position_scores: torch.Tensor,
                 target_position_scores: torch.Tensor, agent_position_targets: torch.Tensor,
                 agent_direction_targets: torch.Tensor, target_position_targets: torch.Tensor):
        direction_loss = self.get_single_loss(agent_direction_scores, agent_direction_targets)
        agent_position_loss = self.get_single_loss(agent_position_scores, agent_position_targets)
        target_position_loss = self.get_single_loss(target_position_scores, target_position_targets)
        return torch.tensor(0.01, device=device) * direction_loss + torch.tensor(0.01, device=device) * agent_position_loss + torch.tensor(0.98, device=device) * target_position_loss

    def get_scores(self, output_scores: torch.Tensor) -> torch.Tensor:
        output_scores_normalized = F.log_softmax(output_scores, -1)
        return output_scores_normalized

    def encode_input(self, input_tensor: torch.LongTensor, input_tensor_lengths: List[int],
                     world_state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            world_state_tensor = self.downsample_image(world_state_tensor)

        # [batch_size, max_input_length] -> [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        hidden, encoder_outputs = self.encoder(input_tensor, input_tensor_lengths)

        # [batch_size, image_width, image_width, num_channels] ->
        # [batch_size, image_width * image_width, num_conv_channels * 3]
        encoded_image = self.situation_encoder(world_state_tensor)
        batch_size, image_num_memory, image_dim = encoded_image.size()
        situation_lengths = [image_num_memory for _ in range(batch_size)]
        projected_keys_agent = self.agent_attention.key_layer(encoded_image)  # [bsz, situation_length, dec_hidden_dim]
        context_situation, attention_weights_agent = self.agent_attention.forward(queries=hidden.unsqueeze(dim=1),
                                                                                  projected_keys=projected_keys_agent,
                                                                                  values=encoded_image,
                                                                                  memory_lengths=situation_lengths)

        projected_keys_target = self.target_attention.key_layer(encoded_image)
        context_visual, attention_weights_target = self.target_attention.forward(queries=hidden.unsqueeze(dim=1),
                                                                                 projected_keys=projected_keys_target,
                                                                                 values=encoded_image,
                                                                                 memory_lengths=situation_lengths)
        final_hidden = torch.cat([context_visual.squeeze(dim=1), hidden], dim=-1)
        final_hidden = torch.tanh(self.hidden_to_hidden(final_hidden))
        output_states = self.hidden_to_output(final_hidden)
        return context_situation.unsqueeze(dim=1), attention_weights_agent, output_states

    def forward(self, input_tensor: torch.LongTensor, input_tensor_lengths: List[int], world_state_tensor: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        context_situation, agent_weights, target_states = self.encode_input(
            input_tensor=input_tensor, input_tensor_lengths=input_tensor_lengths, world_state_tensor=world_state_tensor)
        agent_direction_scores = self.situation_to_directions(context_situation)
        agent_direction_scores_normalized = self.get_scores(agent_direction_scores)
        agent_position_scores_normalized = self.get_scores(agent_weights.exp())
        target_position_scores_normalized = self.get_scores(target_states)
        return {"agent_direction_scores": agent_direction_scores_normalized.squeeze(),
                "agent_position_scores": agent_position_scores_normalized.squeeze(),
                "target_position_scores": target_position_scores_normalized.squeeze()}

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
        self.metric = "accuracy"
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_metric": self.best_metric,
            "metric": "accuracy"
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
