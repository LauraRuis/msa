# Hyperparameter details per experiment

## Baseline

<details>
<summary>Click for hyperparameter details.</summary>
<br>

```
training_batch_size: 200
k: 5
test_batch_size: 1
max_training_examples: None
learning_rate: 0.001
lr_decay: 0.9
lr_decay_steps: 20000
adam_beta_1: 0.9
adam_beta_2: 0.999
print_every: 100
evaluate_every: 1000
max_training_iterations: 200000
weight_target_loss: 0.3
max_testing_examples: 3500
max_decoding_steps: 120
cnn_hidden_num_channels: 50
cnn_kernel_size: 7
cnn_dropout_p: 0.1
embedding_dimension: 25
num_encoder_layers: 1
encoder_hidden_size: 100
encoder_dropout_p: 0.3
encoder_bidirectional: True
num_decoder_layers: 1
attention_type: bahdanau
decoder_dropout_p: 0.3
decoder_hidden_size: 100
```

</details>

## Modularity & Augmentation

### Position Module

<details>
<summary>Click for hyperparameter details.</summary>
<br>

```
training_batch_size: 200
k: 5
upsample_isolated: 100
test_batch_size: 1
learning_rate: 0.001
lr_decay: 0.9
lr_decay_steps: 20000
adam_beta_1: 0.9
adam_beta_2: 0.999
weight_decay: 0.0
evaluate_every: 1000
max_training_iterations: 200000
max_testing_examples: None
max_decoding_steps: 120
cnn_hidden_num_channels: 50
cnn_kernel_size: 7
cnn_dropout_p: 0.1
grid_size: 6
embedding_dimension: 25
num_encoder_layers: 1
encoder_hidden_size: 100
encoder_dropout_p: 0.3
encoder_bidirectional: True
num_decoder_layers: 1
decoder_dropout_p: 0.3
decoder_hidden_size: 100
use_attention: no
use_conditional_attention: no
type_attention: luong
attention_values_key: input_tensor
conditional_attention_values_key: world_state_tensor
```

</details>

### Navigation Module

<details>
<summary>Click for hyperparameter details.</summary>
<br>

```
training_batch_size: 200
k: 5
upsample_isolated: 100
test_batch_size: 1
learning_rate: 0.001
lr_decay: 0.9
lr_decay_steps: 20000
adam_beta_1: 0.9
adam_beta_2: 0.999
weight_decay: 1e-05
evaluate_every: 1000
max_training_iterations: 200000
max_testing_examples: None
max_decoding_steps: 120
cnn_hidden_num_channels: 50
cnn_kernel_size: 7
cnn_dropout_p: 0.1
grid_size: 6
embedding_dimension: 25
num_encoder_layers: 1
encoder_hidden_size: 100
encoder_dropout_p: 0.3
encoder_bidirectional: True
num_decoder_layers: 1
decoder_dropout_p: 0.3
decoder_hidden_size: 100
simplified_architecture: no
use_attention: yes
use_conditional_attention: no
type_attention: luong
attention_values_key: input_tensor
conditional_attention_values_key: None
```

</details>

### Interaction Module

<details>
<summary>Click for hyperparameter details.</summary>
<br>

```

```

</details>

### Transformation Module

<details>
<summary>Click for hyperparameter details.</summary>
<br>

```

```

</details>

## Modularity

### Position Module

See Position Module under Modularity & Augmentation above

### Navigation Module

See Navigation Module under Modularity & Augmentation above

### Interaction Module

See Interaction Module under Modularity & Augmentation above

### Transformation Module

<details>
<summary>Click for hyperparameter details.</summary>
<br>

```

```

</details>

## Baseline & Augmentation

<details>
<summary>Click for hyperparameter details.</summary>
<br>

```
training_batch_size: 200
k: 5
test_batch_size: 1
max_training_examples: None
learning_rate: 0.001
lr_decay: 0.9
lr_decay_steps: 20000
adam_beta_1: 0.9
adam_beta_2: 0.999
evaluate_every: 1000
max_training_iterations: 200000
weight_target_loss: 0.3
max_testing_examples: 3500
max_decoding_steps: 120
cnn_hidden_num_channels: 50
cnn_kernel_size: 7
cnn_dropout_p: 0.1
embedding_dimension: 50
num_encoder_layers: 1
encoder_hidden_size: 400
encoder_dropout_p: 0.2
encoder_bidirectional: True
num_decoder_layers: 1
attention_type: bahdanau
decoder_dropout_p: 0.2
decoder_hidden_size: 400

```

</details>


## All other experiments (k=1,5,10,50 and adverb vocabulary=10,50,100,150)

See "Modularity & Augmentation" above.
