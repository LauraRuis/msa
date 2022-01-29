import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Simple MLP class."""

    def __init__(self, input_dim=0, hidden_dim=0, output_dim=0, depth=1, activation=F.leaky_relu, dropout_p=0.):
        super(MLP, self).__init__()

        self.depth = depth
        self.inner = nn.Linear(input_dim, hidden_dim)
        if depth > 1:
            self.outer = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=dropout_p)
        self.activation = activation

    def forward(self, x):
        if self.depth > 1:
            return self.outer(self.activation(self.inner(x)))
        else:
            return self.activation(self.inner(x))


def get_exact_match(predictions, prediction_lengths, targets_no_sos, target_lengths_no_sos):
    sequence_preds = predictions
    sequence_pred_lengths = prediction_lengths
    target_lengths = target_lengths_no_sos
    max_lengths = torch.where(sequence_pred_lengths > target_lengths, sequence_pred_lengths, target_lengths)
    sequence_targets = targets_no_sos
    max_decoding_length = sequence_preds.size(1)
    if max_decoding_length < sequence_targets.size(1):
        raise ValueError("--max_decoding_steps=%d too low for targets of >= %d" % (max_decoding_length,
                                                                                   sequence_targets.size(1)))
    padding_targets = torch.zeros([sequence_targets.size(0),
                                   max_decoding_length - sequence_targets.size(1)], dtype=torch.long,
                                  device=device)
    padded_targets = torch.cat([sequence_targets, padding_targets], dim=1)
    padded_targets = torch.where(padded_targets == 0, -1, padded_targets)
    accuracy_per_sequence = torch.eq(sequence_preds, padded_targets).float().sum(dim=1) / max_lengths
    exact_match_per_sequence = (accuracy_per_sequence == 1.).float()
    return accuracy_per_sequence, exact_match_per_sequence


def sequence_mask(sequence_length, max_len=None):
    #
    # Create a binary mask of size (batch,max_len)
    #
    #  Input
    #    sequence_length : (batch,) length of each batch sequence
    #
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len, device=device).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    # if sequence_length.is_cuda:
    #     seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


class LearnedAttention(nn.Module):

    def __init__(self, key_size: int, query_size: int, hidden_size: int):
        super(LearnedAttention, self).__init__()
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                memory_lengths: List[int]):
        """
        Key-value memory which takes queries and retrieves weighted combinations of values
          This version masks out certain memories, so that you can differing numbers of memories per batch.

        :param queries: [batch_size, 1, query_dim]
        :param keys: [batch_size, num_memory, query_dim]
        :param values: [batch_size, num_memory, value_dim]
        :param memory_lengths: [batch_size] actual number of keys in each batch
        :return:
            soft_values_retrieval : soft-retrieval of values; [batch_size, 1, value_dim]
            attention_weights : soft-retrieval of values; [batch_size, 1, n_memory]
        """
        projected_keys = self.key_layer(keys)
        values = self.key_layer(values)
        batch_size = projected_keys.size(0)
        assert len(memory_lengths) == batch_size
        memory_lengths = torch.tensor(memory_lengths, dtype=torch.long, device=device)

        # Project queries down to the correct dimension.
        # [bsz, 1, query_dimension] X [bsz, query_dimension, hidden_dim] = [bsz, 1, hidden_dim]
        queries = self.query_layer(queries)

        # [bsz, 1, query_dim] X [bsz, query_dim, num_memory] = [bsz, num_memory, 1]
        scores = self.energy_layer(torch.tanh(queries + projected_keys))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out keys that are on a padding location.encoded_commands
        mask = sequence_mask(memory_lengths)  # [batch_size, num_memory]
        mask = mask.unsqueeze(1)  # [batch_size, 1, num_memory]
        scores = scores.masked_fill(mask == 0, float('-inf'))  # fill with large negative numbers
        attention_weights = F.softmax(scores, dim=2)  # [batch_size, 1, num_memory]

        # [bsz, 1, num_memory] X [bsz, num_memory, value_dim] = [bsz, 1, value_dim]
        soft_values_retrieval = torch.bmm(attention_weights, values)
        return soft_values_retrieval, attention_weights


class Attention(nn.Module):

    def __init__(self, rescale=False):
        """
        :param rescale: if True, then rescale/divide the scores by the sqrt(query_drim) as with the Transformer
        """
        super(Attention, self).__init__()
        self.rescale = rescale

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                                torch.Tensor]:
        """
        Key-value memory which takes queries and retrieves weighted combinations of values

        :param queries: [batch_size, n_queries, query_dim]
        :param keys: [batch_size, n_memory, query_dim]
        :param values: [batch_size, n_memory, value_dim]
        :return: tuple of tensors:
         retrieved_values: soft-retrieval of values; [batch_size, n_queries. value_dim]
         attention_weights : dimension used for retrieval, sum to 1 over n_memory; [batch_size, n_queries, n_memory]
        """
        # [batch_size, n_queries, query_dim] X [batch_size, query_dim, n_memory] -> [batch_size, n_queries, n_memory]
        attention_weights = torch.bmm(queries, keys.transpose(1, 2))
        if self.rescale:
            query_dim = torch.tensor(float(queries.size(2)), device=device)
            attention_weights = torch.div(attention_weights, torch.sqrt(query_dim))
        # [batch_size, n_queries, n_memory]
        attention_weights = F.softmax(attention_weights, dim=2)
        # [batch_size, n_queries, n_memory] X [batch_size, n_memory, value_dim] -> [batch_size, n_queries, value_dim]
        retrieved_values = torch.bmm(attention_weights, values)
        return retrieved_values, attention_weights

    def forward_mask(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                     memory_length: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Key-value memory which takes queries and retrieves weighted combinations of values
          This version masks out certain memories, so that you can differing numbers of memories per batch.
        :param queries: batch_size x n_queries x query_dim
        :param keys: batch_size x n_memory x query_dim
        :param values: batch_size x n_memory x value_dim
        :param memory_length: [batch_size] actual number of keys in each batch
        :return:
         retrieved_values : soft-retrieval of values; batch_size x n_queries x value_dim
         attention_weights : soft-retrieval of values; batch_size x n_queries x n_memory
        """
        query_dim = torch.tensor(float(queries.size(2)), device=device)
        n_memory = keys.size(1)
        batch_size = keys.size(0)
        assert len(memory_length) == batch_size, "Not enough lengths (%d) provided for batch size (%d) in "\
                                                 "Attention"".forward_mask()" % (len(memory_length), batch_size)
        memory_length = torch.tensor(memory_length, dtype=torch.long, device=device)
        # [batch_size, n_queries, query_dim] x [batch_size, query_dim, n_memory] -> [batch_size, n_queries, n_memory]
        attention_weights = torch.bmm(queries, keys.transpose(1, 2))
        if self.rescale:
            attention_weights = torch.div(attention_weights, torch.sqrt(query_dim))

        # Mask out keys that aren't there in each batch.
        # [batch_size, n_memory]
        mask = sequence_mask(memory_length, max_len=n_memory)
        # [batch_size, n_queries, n_memory]
        mask = mask.unsqueeze(1).expand_as(attention_weights)
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=2)
        # [batch_size, n_queries, n_memory] x [batch_size, n_memory, value_dim] ->  [batch_size, n_queries, value_dim]
        retrieved_values = torch.bmm(attention_weights, values)
        return retrieved_values, attention_weights


class LuongDecoder(nn.Module):  # TODO: make universal decoder (so input is continuous tensor, output also)
    """One-step batch decoder with Luong attention"""

    def __init__(self, hidden_size: int, output_size: int, num_layers: int, attention: bool,
                 conditional_attention: bool,
                 dropout_probability=0.1, padding_idx=0):
        """
        :param hidden_size: number of hidden units in RNN, and embedding size for output symbols
        :param output_size: number of output symbols
        :param num_layers: number of hidden layers
        :param attention: whether or not to apply attention
        :param conditional_attention: whether or not to apply conditional attention
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(LuongDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.use_attention = attention
        self.conditional_attention = conditional_attention
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout_probability)
        self.attention = Attention(rescale=False)
        self.join_context = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, bias=False)

    def luong_attention(self, queries: torch.Tensor, keys: torch.Tensor,
                        values: torch.Tensor, value_lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Luong attention
        :param queries: [num_queries, batch_size, hidden_size]
             queries, e.g., emission of decoder RNN for current time step (range [-1,1])
        :param keys: [num_keys, batch_size, hidden_size] the keys to use for the Luong attention
        :param values: [num_memory, batch_size, hidden_size] the values to retrieve from with Luong attention
        :param value_lengths: [batch_size], the actual non-padded length of each sequence in values
        :return: a tuple with the context and the attention weights
        """
        # [batch_size, num_queries, hidden_size]
        decoder_query = queries.transpose(0, 1)

        # Attention over each time step in the query input.
        context, attention_weights = self.attention.forward_mask(queries=decoder_query,
                                                                 keys=keys.transpose(0, 1),
                                                                 values=values.transpose(0, 1),
                                                                 memory_length=value_lengths)
        # context : [batch_size, num_queries, hidden_size]
        # attention_weights : [batch_size, num_queries, num_memory]
        return context, attention_weights

    def forward_step(self, hidden: Tuple[torch.Tensor, torch.Tensor], target_tokens: torch.Tensor,
                     attention_keys=None,  attention_values=None,
                     attention_values_lengths=None, conditional_attention_values=None,
                     conditional_attention_values_lengths=None) -> Tuple[torch.Tensor,
                                                                         Tuple[torch.Tensor, torch.Tensor],
                                                                         torch.Tensor]:
        """
        Run batch decoder forward for a single time step.
         Each decoder step considers the values in `attention_values` through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param hidden: previous decoder state, which is pair of tensors [num_layers, batch_size, hidden_size]
        (pair for hidden and cell)
        :param target_tokens: [1, batch_size, hidden_size], the current time-step input to the RNN
        :param attention_keys: [num_keys, batch_size, hidden_size], the keys to do attention over
        :param attention_values: [num_memory, batch_size, hidden_size], the values to do attention over
        :param attention_values_lengths: [batch_size], the unpadded length of the sequences in attention_values
        :param conditional_attention_values: [num_memory, batch_size, hidden_size], values to do cond. attention over
        :param conditional_attention_values_lengths: [batch_size], unpadded length of sequences in c_attention_values
        :return: hidden : current decoder state, which is a pair of tensors [num_layers, batch_size, hidden_size]
                            (pair for hidden and cell)
                 attention_weights : attention weights, [batch_size, 1, max_input_length]
        """
        if self.use_attention:
            assert not attention_values is None, "Provide attention values for attention."
        if self.conditional_attention:
            assert not conditional_attention_values is None, "Provide conditional attention values for cond. attn."
        embedded_targets = self.dropout(self.embedding(target_tokens))  # [batch_size, hidden_size]

        # [1, batch_size, hidden_size] -> [1, batch_size, hidden_size]
        rnn_output, hidden = self.lstm(embedded_targets, hidden)
        # hidden: [num_layers, batch_size, hidden_size] (pair for hidden and cell)

        output, attention_weights = self.get_output(rnn_output, attention_keys, attention_values,
                                                    attention_values_lengths, conditional_attention_values,
                                                    conditional_attention_values_lengths)

        return output, hidden, attention_weights  # TODO: return conditional attention values?
        # output : [un-normalized probabilities] [batch_size, output_size]
        # hidden: tuple of size [num_layers, batch_size, hidden_size] (for hidden and cell)
        # attention_weights: [batch_size, max_input_length]

    def get_output(self, rnn_output, attention_keys, attention_values, attention_values_lengths,
                   conditional_attention_values=None, conditional_attention_values_lengths=None):
        if self.use_attention:
            # [1, batch_size, hidden_size]
            context, attention_weights = self.luong_attention(queries=rnn_output, keys=attention_keys,
                                                              values=attention_values,
                                                              value_lengths=attention_values_lengths)
            context = context.transpose(0, 1)

            if self.conditional_attention:
                queries = torch.cat([rnn_output, context], dim=-1)
                queries = self.tanh(self.join_context(queries))
                conditional_context, conditional_attention_weights = self.luong_attention(
                    queries=queries, keys=conditional_attention_values, values=conditional_attention_values,
                    value_lengths=conditional_attention_values_lengths)
                conditional_context = conditional_context.transpose(0, 1)
                context = torch.cat([context, conditional_context], dim=-1)
                context = self.tanh(self.join_context(context))

            joint_context = torch.cat([rnn_output, context], dim=-1)
            joint_context = torch.relu(self.join_context(joint_context))  # TODO: relu here?
        else:
            joint_context = rnn_output
            attention_weights = None
        output = self.hidden_to_output(joint_context)
        return output, attention_weights

    def forward(self, target_tokens: torch.Tensor, target_lengths: List[int],
                hidden: Tuple[torch.Tensor, torch.Tensor], attention_keys: torch.Tensor,
                attention_values: torch.Tensor, attention_values_lengths: List[int],
                conditional_attention_values=None,
                conditional_attention_values_lengths=None) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """
        Run batch attention decoder forward for a series of steps
         Each decoder step considers all sequences in `attention_values` through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param target_tokens: [batch_size, max_target_length]
        :param target_lengths: [batch_size], unpadded length of each input sequence in the batch
        :param hidden: tuple of tensors [num_layers, batch_size, hidden_size] (for hidden and cell)
        :param attention_keys: [num_keys, batch_size, hidden_size], the keys to do attention over
        :param attention_values: [num_memory, batch_size, hidden_size], the values to do attention over
        :param attention_values_lengths: [batch_size], the unpadded length of the sequences in attention_values
        :param conditional_attention_values: [num_memory, batch_size, hidden_size], values to do cond. attention over
        :param conditional_attention_values_lengths: [batch_size], unpadded length of sequences in c_attention_values
        :return: output : unnormalized log-score, [max_length, batch_size, output_size]
          hidden : current decoder state, tuple with each [num_layers, batch_size, hidden_size] (for hidden and cell)
        """
        # [batch_size, max_target_length] -> [batch_size, max_target_length, hidden_size]
        embedded_targets = self.dropout(self.embedding(target_tokens))
        # Sort the sequences by length in descending order.
        max_target_length = embedded_targets.size(1)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        target_lengths, perm_idx = torch.sort(target_lengths, descending=True)
        embedded_targets = embedded_targets[perm_idx]
        init_hidden = (hidden[0][:, perm_idx, :], hidden[1][:, perm_idx, :])

        # [batch_size, max_query_target_length, hidden_size] -> [sum(embedded_target_lengths), hidden_size]
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embedded_targets, target_lengths,
                                                               batch_first=True)

        # [sum(target_lengths), hidden_size] -> [sum(target_lengths), hidden_size]
        packed_output, (hidden, cell) = self.lstm(packed_input, init_hidden)
        # packed_output: [sum(target_lengths), hidden_size]
        # hidden and cell: [num_layers, batch_size, hidden_size]

        # [sum(target_lengths), hidden_size] -> [max_target_length, batch_size, hidden_size]
        rnn_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output,
                                                               total_length=max_target_length)

        # Reverse the sorting variables.
        _, unperm_idx = perm_idx.sort(0)
        embedded_targets = embedded_targets[unperm_idx]
        rnn_output = rnn_output[:, unperm_idx, :]
        seq_len = target_lengths[unperm_idx].tolist()

        output, attention_weights = self.get_output(rnn_output, attention_keys, attention_values,
                                                    attention_values_lengths, conditional_attention_values,
                                                    conditional_attention_values_lengths)

        return output, seq_len, attention_weights
        # output : [unnormalized log-score] [max_length, batch_size, output_size]
        # seq_len : length of each output sequence
        # attention_weights:

    def initialize_hidden(self, encoder_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Populate the hidden variables with a message from the encoder.
        All layers, and both the hidden and cell vectors, are filled with the same message.
        :param encoder_message:  [batch_size, hidden_size] tensor
        :return: tuple of Tensors representing the hidden and cell state of shape: [num_layers, batch_size, hidden_dim]
        """
        encoder_message = encoder_message.unsqueeze(0)  # [1, batch_size, hidden_size]
        encoder_message = encoder_message.expand(self.num_layers, -1,
                                                 -1).contiguous()  # [num_layers, batch_size, hidden_size]
        return encoder_message.clone(), encoder_message.clone()

    def extra_repr(self) -> str:
        return "AttentionDecoderRNN\n num_layers={}\n hidden_size={}\n dropout={}\n num_output_symbols={}\n".format(
            self.num_layers, self.hidden_size, self.dropout_probability, self.output_size
        )


class BOW(nn.Module):
    """
    Embed a sequence of symbols using an bag-of-words encoder.
    """
    def __init__(self, input_size: int, embedding_dim: int,  dropout_probability: float, padding_idx: int):
        """
        :param input_size: number of input symbols
        :param embedding_dim: number of hidden units in RNN encoder, and size of all embeddings
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(BOW, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.dropout_probability = dropout_probability
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, input_batch: torch.LongTensor, input_lengths: List[int]) -> Tuple[torch.Tensor, dict]:
        """
        :param input_batch: [batch_size, max_length]; batched padded input sequences
        :param input_lengths: [batch_size]; actual length of each padded input sequence
        :return: hidden states for last layer of last time step, the output of the last layer per time step and
        the sequence lengths per example in the batch.
        NB: The hidden states in the bidirectional case represent the final hidden state of each directional encoder,
        meaning the whole sequence in both directions, whereas the output per time step represents different parts of
        the sequences (0:t for the forward LSTM, t:T for the backward LSTM).
        """
        input_embeddings = self.embedding(input_batch)  # [batch_size, max_length, embedding_dim]
        input_embeddings = self.dropout(input_embeddings)  # [batch_size, max_length, embedding_dim]
        average_embedding = torch.mean(input_embeddings, dim=1)
        return average_embedding, {"encoder_outputs": input_embeddings, "sequence_lengths": input_lengths}


class BahdanauDecoder(nn.Module):
    """One-step batch decoder with Bahdanau et al. attention"""

    def __init__(self, hidden_size: int, output_size: int, num_layers: int, key_size: int, attention: LearnedAttention,
                 conditional_attention: LearnedAttention,
                 dropout_probability=0.1, padding_idx=0):
        """
        :param hidden_size: number of hidden units in RNN, and embedding size for output symbols
        :param output_size: number of output symbols
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(BahdanauDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.tanh = nn.Tanh()
        self.attention = attention
        self.conditional_attention = conditional_attention
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers=num_layers, dropout=dropout_probability)
        self.attention = attention
        if self.conditional_attention:
            self.join_context = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.output_to_hidden = nn.Linear(hidden_size * 3, hidden_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, bias=False)

    def forward_step(self, hidden: Tuple[torch.Tensor, torch.Tensor], target_tokens: torch.Tensor,
                     attention_keys=None,  attention_values=None,
                     attention_values_lengths=None, conditional_attention_values=None,
                     conditional_attention_values_lengths=None) -> Tuple[torch.Tensor,
                                                                         Tuple[torch.Tensor, torch.Tensor],
                                                                         torch.Tensor]:
        """
        Run batch decoder forward for a single time step.
         Each decoder step considers the values in `attention_values` through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param hidden: previous decoder state, which is pair of tensors [num_layers, batch_size, hidden_size]
        (pair for hidden and cell)
        :param target_tokens: [1, batch_size, hidden_size], the current time-step input to the RNN
        :param attention_keys: [num_keys, batch_size, hidden_size], the keys to do attention over
        :param attention_values: [num_memory, batch_size, hidden_size], the values to do attention over
        :param attention_values_lengths: [batch_size], the unpadded length of the sequences in attention_values
        :param conditional_attention_values: [num_memory, batch_size, hidden_size], values to do cond. attention over
        :param conditional_attention_values_lengths: [batch_size], unpadded length of sequences in c_attention_values
        :return: hidden : current decoder state, which is a pair of tensors [num_layers, batch_size, hidden_size]
                            (pair for hidden and cell)
                 attention_weights : attention weights, [batch_size, 1, max_input_length]
        """
        if self.attention:
            assert not attention_values is None, "Provide attention values for attention."
        if self.conditional_attention:
            assert not conditional_attention_values is None, "Provide conditional attention values for cond. attn."
        last_hidden, last_cell = hidden

        # Embed each input symbol
        embedded_input = self.dropout(self.embedding(target_tokens))  # [batch_size, hidden_size]

        # Bahdanau attention
        context, attention_weights_commands = self.attention(
            queries=last_hidden.transpose(0, 1), keys=attention_keys.transpose(0, 1),
            values=attention_values.transpose(0, 1), memory_lengths=attention_values_lengths)

        if self.conditional_attention:
            queries = torch.cat([last_hidden.transpose(0, 1), context], dim=-1)
            queries = self.tanh(self.join_context(queries))
            conditional_context, conditional_attention_weights = self.attention(
                queries=queries, keys=conditional_attention_values.transpose(0, 1),
                values=conditional_attention_values.transpose(0, 1),
                memory_lengths=conditional_attention_values_lengths)
            context = torch.cat([context, conditional_context], dim=-1)
            context = self.tanh(self.join_context(context))

        # Concatenate the context vector and RNN hidden state, and map to an output
        attention_weights_commands = attention_weights_commands.squeeze(1)  # [batch_size, max_input_length]
        concat_input = torch.cat([embedded_input,
                                  context.transpose(0, 1)], dim=2)  # [1, batch_size hidden_size*2]
        last_hidden = (last_hidden, last_cell)
        lstm_output, hidden = self.lstm(concat_input, last_hidden)
        # lstm_output: [1, batch_size, hidden_size]
        # hidden: tuple of each [num_layers, batch_size, hidden_size] (pair for hidden and cell)
        # output = self.hidden_to_output(lstm_output)  # [batch_size, output_size]
        # output = output.squeeze(dim=0)

        # Concatenate all outputs and project to output size.
        pre_output = torch.cat([embedded_input, lstm_output,
                                context.transpose(0, 1)], dim=2)
        pre_output = self.output_to_hidden(pre_output)  # [1, batch_size, hidden_size]
        output = self.hidden_to_output(pre_output)  # [batch_size, output_size]
        # output = output.squeeze(dim=0)   # [batch_size, output_size]

        return output, hidden, attention_weights_commands
        # output : [un-normalized probabilities] [batch_size, output_size]
        # hidden: tuple of size [num_layers, batch_size, hidden_size] (for hidden and cell)
        # attention_weights: [batch_size, max_input_length]

    def forward(self, target_tokens: torch.Tensor, target_lengths: List[int],
                hidden: Tuple[torch.Tensor, torch.Tensor], attention_keys: torch.Tensor,
                attention_values: torch.Tensor, attention_values_lengths: List[int],
                conditional_attention_values=None,
                conditional_attention_values_lengths=None) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """
        Run batch attention decoder forward for a series of steps
         Each decoder step considers all sequences in `attention_values` through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param target_tokens: [batch_size, max_target_length]
        :param target_lengths: [batch_size], unpadded length of each input sequence in the batch
        :param hidden: tuple of tensors [num_layers, batch_size, hidden_size] (for hidden and cell)
        :param attention_keys: [num_keys, batch_size, hidden_size], the keys to do attention over
        :param attention_values: [num_memory, batch_size, hidden_size], the values to do attention over
        :param attention_values_lengths: [batch_size], the unpadded length of the sequences in attention_values
        :param conditional_attention_values: [num_memory, batch_size, hidden_size], values to do cond. attention over
        :param conditional_attention_values_lengths: [batch_size], unpadded length of sequences in c_attention_values
        :return: output : unnormalized log-score, [max_length, batch_size, output_size]
          hidden : current decoder state, tuple with each [num_layers, batch_size, hidden_size] (for hidden and cell)
        """
        batch_size, max_time = target_tokens.size()

        # Sort the sequences by length in descending order
        input_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_tokens_sorted = target_tokens.index_select(dim=0, index=perm_idx)
        initial_h, initial_c = hidden
        hidden = (initial_h.index_select(dim=1, index=perm_idx),
                  initial_c.index_select(dim=1, index=perm_idx))
        encoded_commands = attention_values.index_select(dim=1, index=perm_idx)
        commands_lengths = torch.tensor(attention_values_lengths, device=device)
        commands_lengths = commands_lengths.index_select(dim=0, index=perm_idx)

        # # For efficiency
        # projected_keys_textual = self.attention.key_layer(
        #     encoded_commands)  # [max_input_length, batch_size, dec_hidden_dim]

        all_attention_weights = []
        lstm_output = []
        for time in range(max_time):
            input_token = input_tokens_sorted[:, time]
            (output, hidden, attention_weights_commands) = self.forward_step(target_tokens=input_token.unsqueeze(dim=0),
                                                                             hidden=hidden,
                                                                             attention_keys=encoded_commands,
                                                                             attention_values=encoded_commands,
                                                                             attention_values_lengths=commands_lengths,
                                                                             conditional_attention_values=conditional_attention_values,
                                                                             conditional_attention_values_lengths=conditional_attention_values_lengths)
            all_attention_weights.append(attention_weights_commands.unsqueeze(dim=0))
            lstm_output.append(output)
        lstm_output = torch.cat(lstm_output, dim=0)  # [max_time, batch_size, output_size]
        attention_weights = torch.cat(all_attention_weights, dim=0)  # [max_time, batch_size, command_length]

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = lstm_output.index_select(dim=1, index=unperm_idx)  # [max_time, batch_size, output_size]
        seq_len = input_lengths[unperm_idx].tolist()
        attention_weights = attention_weights.index_select(dim=1, index=unperm_idx)

        return lstm_output, seq_len, attention_weights.sum(dim=0)
        # output : [unnormalized log-score] [max_length, batch_size, output_size]
        # seq_len : length of each output sequence

    def initialize_hidden(self, encoder_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Populate the hidden variables with a message from the encoder.
        All layers, and both the hidden and cell vectors, are filled with the same message.
        :param encoder_message:  [batch_size, hidden_size] tensor
        :return: tuple of Tensors representing the hidden and cell state of shape: [num_layers, batch_size, hidden_dim]
        """
        encoder_message = encoder_message.unsqueeze(0)  # [1, batch_size, hidden_size]
        encoder_message = encoder_message.expand(self.num_layers, -1,
                                                 -1).contiguous()  # [num_layers, batch_size, hidden_size]
        return encoder_message.clone(), encoder_message.clone()

    def extra_repr(self) -> str:
        return "AttentionDecoderRNN\n num_layers={}\n hidden_size={}\n dropout={}\n num_output_symbols={}\n".format(
            self.num_layers, self.hidden_size, self.dropout_probability, self.output_size
        )


class BahdanauAttentionDecoderRNN(nn.Module):
    """One-step batch decoder with Luong et al. attention"""

    def __init__(self, hidden_size: int, output_size: int, num_layers: int, textual_attention: Attention,
                 visual_attention: Attention, dropout_probability=0.1, padding_idx=0,
                 conditional_attention=False):
        """
        :param hidden_size: number of hidden units in RNN, and embedding size for output symbols
        :param output_size: number of output symbols
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(BahdanauAttentionDecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.conditional_attention = conditional_attention
        if self.conditional_attention:
            self.queries_to_keys = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_probability)
        self.textual_attention = textual_attention
        if conditional_attention:
            self.visual_attention = visual_attention
            self.lstm = nn.LSTM(hidden_size * 3, hidden_size, num_layers=num_layers, dropout=dropout_probability)
            self.output_to_hidden = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        else:
            self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers=num_layers, dropout=dropout_probability)
            self.output_to_hidden = nn.Linear(hidden_size * 3, hidden_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, bias=False)

    def forward_step(self, target_tokens: torch.LongTensor, hidden: Tuple[torch.Tensor, torch.Tensor],
                     attention_keys: torch.Tensor, attention_values: torch.Tensor,
                     attention_values_lengths: List[int], conditional_attention_values=None,
                     conditional_attention_values_lengths=None) -> Tuple[torch.Tensor,
                                                                               Tuple[torch.Tensor, torch.Tensor],
                                                                               torch.Tensor]:
        """
        Run batch decoder forward for a single time step.
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param target_tokens: one time step inputs tokens of length batch_size
        :param hidden: previous decoder state, which is pair of tensors [num_layers, batch_size, hidden_size]
        (pair for hidden and cell)
        :param attention_values: all encoder outputs, [max_input_length, batch_size, hidden_size]
        :param attention_values_lengths: length of each padded input seqencoded_commandsuence that were passed to the encoder.
        :param conditional_attention_values: the situation encoder outputs, [image_dimension * image_dimension, batch_size,
         hidden_size]
        :return: output : un-normalized output probabilities, [batch_size, output_size]
          hidden : current decoder state, which is a pair of tensors [num_layers, batch_size, hidden_size]
           (pair for hidden and cell)
          attention_weights : attention weights, [batch_size, 1, max_input_length]
        """
        last_hidden = hidden
        input_tokens = target_tokens.squeeze(dim=0)  # TODO: check if input tokens should be [1, bsz] or [bsz]
        encoded_commands = attention_values
        commands_lengths = attention_values_lengths
        if self.conditional_attention:
            encoded_situations = conditional_attention_values.transpose(0, 1)
        last_hidden, last_cell = last_hidden

        # Embed each input symbol
        embedded_input = self.embedding(input_tokens)  # [batch_size, hidden_size]
        embedded_input = self.dropout(embedded_input)
        embedded_input = embedded_input.unsqueeze(0)  # [1, batch_size, hidden_size]

        # Bahdanau attention
        context_command, attention_weights_commands = self.textual_attention(
            queries=last_hidden.transpose(0, 1), keys=encoded_commands.transpose(0, 1),
            values=encoded_commands.transpose(0, 1), memory_lengths=commands_lengths)
        if self.conditional_attention:
            batch_size, image_num_memory, _ = encoded_situations.size()
        situation_lengths = conditional_attention_values_lengths

        if self.conditional_attention:
            queries = torch.cat([last_hidden.transpose(0, 1), context_command], dim=-1)
            queries = self.tanh(self.queries_to_keys(queries))
        else:
            queries = last_hidden.transpose(0, 1)

        # Concatenate the context vector and RNN hidden state, and map to an output
        attention_weights_commands = attention_weights_commands.squeeze(1)  # [batch_size, max_input_length]

        if self.conditional_attention:
            context_situation, attention_weights_situations = self.visual_attention(
                queries=queries, keys=encoded_situations,
                values=encoded_situations, memory_lengths=situation_lengths)
            # context : [batch_size, 1, hidden_size]
            # attention_weights : [batch_size, 1, max_input_length]
            attention_weights_situations = attention_weights_situations.squeeze(1)  # [batch_size, im_dim * im_dim]

            concat_input = torch.cat([embedded_input,
                                      context_command.transpose(0, 1),
                                      context_situation.transpose(0, 1)], dim=2)  # [1, batch_size hidden_size*3]
        else:
            concat_input = torch.cat([embedded_input,
                                      context_command.transpose(0, 1)], dim=2)  # [1, batch_size hidden_size*3]

        last_hidden = (last_hidden, last_cell)
        lstm_output, hidden = self.lstm(concat_input, last_hidden)
        # lstm_output: [1, batch_size, hidden_size]
        # hidden: tuple of each [num_layers, batch_size, hidden_size] (pair for hidden and cell)
        # output = self.hidden_to_output(lstm_output)  # [batch_size, output_size]
        # output = output.squeeze(dim=0)

        # Concatenate all outputs and project to output size.
        if self.conditional_attention:
            pre_output = torch.cat([embedded_input, lstm_output,
                                    context_command.transpose(0, 1), context_situation.transpose(0, 1)], dim=2)
        else:
            pre_output = torch.cat([embedded_input, lstm_output,
                                    context_command.transpose(0, 1)], dim=2)
        pre_output = self.output_to_hidden(pre_output)  # [1, batch_size, hidden_size]
        output = self.hidden_to_output(pre_output)  # [batch_size, output_size]
        # output = output.squeeze(dim=0)   # [batch_size, output_size]

        return output, hidden, attention_weights_commands
        # output : [un-normalized probabilities] [batch_size, output_size]
        # hidden: tuple of size [num_layers, batch_size, hidden_size] (for hidden and cell)
        # attention_weights: [batch_size, max_input_length]

    def forward(self, target_tokens: torch.LongTensor, target_lengths: List[int],
                hidden: Tuple[torch.Tensor, torch.Tensor], attention_keys: torch.Tensor,
                attention_values: torch.Tensor, attention_values_lengths: List[int],
                conditional_attention_values=None,
                conditional_attention_values_lengths=None) -> Tuple[torch.Tensor, List[int],
                                                                    torch.Tensor]:
        """
        Run batch attention decoder forward for a series of steps
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param input_tokens: [batch_size, max_length];  padded target sequences
        :param input_lengths: [batch_size] for sequence length of each padded target sequence
        :param init_hidden: tuple of tensors [num_layers, batch_size, hidden_size] (for hidden and cell)
        :param encoded_commands: [max_input_length, batch_size, embedding_dim]
        :param commands_lengths: [batch_size] sequence length of each encoder sequence (without padding)
        :param encoded_situations: [batch_size, image_width * image_width, image_features]; encoded image situations.
        :return: output : unnormalized log-score, [max_length, batch_size, output_size]
          hidden : current decoder state, tuple with each [num_layers, batch_size, hidden_size] (for hidden and cell)
        """
        input_tokens = target_tokens
        input_lengths = target_lengths
        init_hidden = hidden
        encoded_commands = attention_values
        commands_lengths = attention_values_lengths
        encoded_situations = conditional_attention_values

        batch_size, max_time = input_tokens.size()

        # Sort the sequences by length in descending order
        input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_tokens_sorted = input_tokens.index_select(dim=0, index=perm_idx)
        initial_h, initial_c = init_hidden
        hidden = (initial_h.index_select(dim=1, index=perm_idx),
                  initial_c.index_select(dim=1, index=perm_idx))
        encoded_commands = encoded_commands.index_select(dim=1, index=perm_idx)
        commands_lengths = torch.tensor(commands_lengths, device=device)
        commands_lengths = commands_lengths.index_select(dim=0, index=perm_idx)
        if self.conditional_attention:
            encoded_situations = encoded_situations.index_select(dim=1, index=perm_idx)

        # For efficiency
        # projected_keys_visual = self.visual_attention.key_layer(
        #     encoded_situations)  # [batch_size, situation_length, dec_hidden_dim]
        # projected_keys_textual = self.textual_attention.key_layer(
        #     encoded_commands)  # [max_input_length, batch_size, dec_hidden_dim]

        all_attention_weights = []
        lstm_output = []
        for time in range(max_time):
            input_token = input_tokens_sorted[:, time]
            (output, hidden, attention_weights_commands) = self.forward_step(
                target_tokens=input_token, hidden=hidden, attention_keys=encoded_commands,
                attention_values=encoded_commands, attention_values_lengths=commands_lengths,
                conditional_attention_values=encoded_situations,
                conditional_attention_values_lengths=conditional_attention_values_lengths)
            all_attention_weights.append(attention_weights_commands.unsqueeze(0))
            lstm_output.append(output)
        lstm_output = torch.cat(lstm_output, dim=0)  # [max_time, batch_size, output_size]
        attention_weights = torch.cat(all_attention_weights, dim=0)  # [max_time, batch_size, situation_dim**2]

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = lstm_output.index_select(dim=1, index=unperm_idx)  # [max_time, batch_size, output_size]
        seq_len = input_lengths[unperm_idx].tolist()
        attention_weights = attention_weights.index_select(dim=1, index=unperm_idx)

        return lstm_output, seq_len, attention_weights.sum(dim=0)
        # output : [unnormalized log-score] [max_length, batch_size, output_size]
        # seq_len : length of each output sequence

    def initialize_hidden(self, encoder_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Populate the hidden variables with a message from the encoder.
        All layers, and both the hidden and cell vectors, are filled with the same message.
        :param encoder_message:  [batch_size, hidden_size] tensor
        :return: tuple of Tensors representing the hidden and cell state of shape: [num_layers, batch_size, hidden_dim]
        """
        encoder_message = encoder_message.unsqueeze(0)  # [1, batch_size, hidden_size]
        encoder_message = encoder_message.expand(self.num_layers, -1,
                                                 -1).contiguous()  # [num_layers, batch_size, hidden_size]
        return encoder_message.clone(), encoder_message.clone()

    def extra_repr(self) -> str:
        return "AttentionDecoderRNN\n num_layers={}\n hidden_size={}\n dropout={}\n num_output_symbols={}\n".format(
            self.num_layers, self.hidden_size, self.dropout_probability, self.output_size
        )