# Architecture details per experiment

## Baseline

<details>
<summary>Click for architecture details.</summary>
<br>
 
```
Total parameters: 440275
situation_encoder.conv_1.weight : [50, 16, 1, 1]
situation_encoder.conv_1.bias : [50]
situation_encoder.conv_2.weight : [50, 16, 5, 5]
situation_encoder.conv_2.bias : [50]
situation_encoder.conv_3.weight : [50, 16, 7, 7]
situation_encoder.conv_3.bias : [50]
visual_attention.key_layer.weight : [100, 150]
visual_attention.query_layer.weight : [100, 100]
visual_attention.energy_layer.weight : [1, 100]
encoder.embedding.weight : [21, 25]
encoder.lstm.weight_ih_l0 : [400, 25]
encoder.lstm.weight_hh_l0 : [400, 100]
encoder.lstm.bias_ih_l0 : [400]
encoder.lstm.bias_hh_l0 : [400]
encoder.lstm.weight_ih_l0_reverse : [400, 25]
encoder.lstm.weight_hh_l0_reverse : [400, 100]
encoder.lstm.bias_ih_l0_reverse : [400]
encoder.lstm.bias_hh_l0_reverse : [400]
enc_hidden_to_dec_hidden.weight : [100, 100]
enc_hidden_to_dec_hidden.bias : [100]
textual_attention.key_layer.weight : [100, 100]
textual_attention.query_layer.weight : [100, 100]
textual_attention.energy_layer.weight : [1, 100]
attention_decoder.queries_to_keys.weight : [100, 200]
attention_decoder.queries_to_keys.bias : [100]
attention_decoder.embedding.weight : [9, 100]
attention_decoder.lstm.weight_ih_l0 : [400, 300]
attention_decoder.lstm.weight_hh_l0 : [400, 100]
attention_decoder.lstm.bias_ih_l0 : [400]
attention_decoder.lstm.bias_hh_l0 : [400]
attention_decoder.output_to_hidden.weight : [100, 400]
attention_decoder.hidden_to_output.weight : [9, 100]
```

</details>

## Modularity & Augmentation

### Position Module

<details>
<summary>Click for architecture details.</summary>
<br>

```
Total parameters: 258079
encoder_module_dict.world_state_tensor.conv_1.weight : [50, 16, 1, 1]
encoder_module_dict.world_state_tensor.conv_1.bias : [50]
encoder_module_dict.world_state_tensor.conv_2.weight : [50, 16, 5, 5]
encoder_module_dict.world_state_tensor.conv_2.bias : [50]
encoder_module_dict.world_state_tensor.conv_3.weight : [50, 16, 7, 7]
encoder_module_dict.world_state_tensor.conv_3.bias : [50]
encoder_module_dict.input_tensor.embedding.weight : [21, 25]
encoder_module_dict.input_tensor.lstm.weight_ih_l0 : [400, 25]
encoder_module_dict.input_tensor.lstm.weight_hh_l0 : [400, 100]
encoder_module_dict.input_tensor.lstm.bias_ih_l0 : [400]
encoder_module_dict.input_tensor.lstm.bias_hh_l0 : [400]
encoder_module_dict.input_tensor.lstm.weight_ih_l0_reverse : [400, 25]
encoder_module_dict.input_tensor.lstm.weight_hh_l0_reverse : [400, 100]
encoder_module_dict.input_tensor.lstm.bias_ih_l0_reverse : [400]
encoder_module_dict.input_tensor.lstm.bias_hh_l0_reverse : [400]
decoder_module_dict.agent_position.attention.key_layer.weight : [100, 150]
decoder_module_dict.agent_position.attention.query_layer.weight : [100, 100]
decoder_module_dict.agent_position.attention.energy_layer.weight : [1, 100]
decoder_module_dict.target_position.attention.key_layer.weight : [100, 150]
decoder_module_dict.target_position.attention.query_layer.weight : [100, 100]
decoder_module_dict.target_position.attention.energy_layer.weight : [1, 100]
decoder_module_dict.agent_direction.attention.key_layer.weight : [100, 150]
decoder_module_dict.agent_direction.attention.query_layer.weight : [100, 100]
decoder_module_dict.agent_direction.attention.energy_layer.weight : [1, 100]
decoder_module_dict.agent_direction.projection.0.weight : [100, 200]
decoder_module_dict.agent_direction.projection.0.bias : [100]
decoder_module_dict.agent_direction.projection.2.weight : [4, 100]
decoder_module_dict.agent_direction.projection.2.bias : [4]
```
  
</details>

### Navigation Module

<details>
<summary>Click for architecture details.</summary>
<br>

```
Total parameters: 235525
encoder_module_dict.input_tensor.embedding.weight : [21, 25]
encoder_module_dict.input_tensor.lstm.weight_ih_l0 : [400, 25]
encoder_module_dict.input_tensor.lstm.weight_hh_l0 : [400, 100]
encoder_module_dict.input_tensor.lstm.bias_ih_l0 : [400]
encoder_module_dict.input_tensor.lstm.bias_hh_l0 : [400]
encoder_module_dict.input_tensor.lstm.weight_ih_l0_reverse : [400, 25]
encoder_module_dict.input_tensor.lstm.weight_hh_l0_reverse : [400, 100]
encoder_module_dict.input_tensor.lstm.bias_ih_l0_reverse : [400]
encoder_module_dict.input_tensor.lstm.bias_hh_l0_reverse : [400]
encoder_module_dict.mlp.0.weight : [100, 3]
encoder_module_dict.mlp.0.bias : [100]
encoder_module_dict.mlp.1.inner.weight : [100, 100]
encoder_module_dict.mlp.1.inner.bias : [100]
decoder_module_dict.planner_target_tensor.embedding.weight : [10, 100]
decoder_module_dict.planner_target_tensor.lstm.weight_ih_l0 : [400, 100]
decoder_module_dict.planner_target_tensor.lstm.weight_hh_l0 : [400, 100]
decoder_module_dict.planner_target_tensor.lstm.bias_ih_l0 : [400]
decoder_module_dict.planner_target_tensor.lstm.bias_hh_l0 : [400]
decoder_module_dict.planner_target_tensor.join_context.weight : [100, 200]
decoder_module_dict.planner_target_tensor.hidden_to_output.weight : [10, 100]
encoder_hiddens_to_initial_hidden.weight : [100, 200]
encoder_hiddens_to_initial_hidden.bias : [100]
```
  
</details>

### Interaction Module

<details>
<summary>Click for architecture details.</summary>
<br>

```
Total parameters: 309575
encoder_module_dict.world_state_tensor.conv_1.weight : [50, 16, 1, 1]
encoder_module_dict.world_state_tensor.conv_1.bias : [50]
encoder_module_dict.world_state_tensor.conv_2.weight : [50, 16, 5, 5]
encoder_module_dict.world_state_tensor.conv_2.bias : [50]
encoder_module_dict.world_state_tensor.conv_3.weight : [50, 16, 7, 7]
encoder_module_dict.world_state_tensor.conv_3.bias : [50]
encoder_module_dict.input_tensor.embedding.weight : [21, 25]
encoder_module_dict.input_tensor.lstm.weight_ih_l0 : [400, 25]
encoder_module_dict.input_tensor.lstm.weight_hh_l0 : [400, 100]
encoder_module_dict.input_tensor.lstm.bias_ih_l0 : [400]
encoder_module_dict.input_tensor.lstm.bias_hh_l0 : [400]
encoder_module_dict.input_tensor.lstm.weight_ih_l0_reverse : [400, 25]
encoder_module_dict.input_tensor.lstm.weight_hh_l0_reverse : [400, 100]
encoder_module_dict.input_tensor.lstm.bias_ih_l0_reverse : [400]
encoder_module_dict.input_tensor.lstm.bias_hh_l0_reverse : [400]
encoder_module_dict.mlp.0.weight : [100, 1]
encoder_module_dict.mlp.0.bias : [100]
encoder_module_dict.mlp.1.inner.weight : [100, 100]
encoder_module_dict.mlp.1.inner.bias : [100]
decoder_module_dict.transitive_target_tensor.embedding.weight : [5, 100]
decoder_module_dict.transitive_target_tensor.lstm.weight_ih_l0 : [400, 100]
decoder_module_dict.transitive_target_tensor.lstm.weight_hh_l0 : [400, 100]
decoder_module_dict.transitive_target_tensor.lstm.bias_ih_l0 : [400]
decoder_module_dict.transitive_target_tensor.lstm.bias_hh_l0 : [400]
decoder_module_dict.transitive_target_tensor.join_context.weight : [100, 200]
decoder_module_dict.transitive_target_tensor.hidden_to_output.weight : [5, 100]
world_state_to_decoder_hidden.weight : [100, 150]
world_state_to_decoder_hidden.bias : [100]
encoder_hiddens_to_initial_hidden.weight : [100, 200]
encoder_hiddens_to_initial_hidden.bias : [100]
```

</details>

### Transformation Module

<details>
<summary>Click for architecture details.</summary>
<br>

```
Total parameters: 4512350
encoder_module_dict.input_tensor.embedding.weight : [171, 50]
encoder_module_dict.input_tensor.lstm.weight_ih_l0 : [1600, 50]
encoder_module_dict.input_tensor.lstm.weight_hh_l0 : [1600, 400]
encoder_module_dict.input_tensor.lstm.bias_ih_l0 : [1600]
encoder_module_dict.input_tensor.lstm.bias_hh_l0 : [1600]
encoder_module_dict.input_tensor.lstm.weight_ih_l0_reverse : [1600, 50]
encoder_module_dict.input_tensor.lstm.weight_hh_l0_reverse : [1600, 400]
encoder_module_dict.input_tensor.lstm.bias_ih_l0_reverse : [1600]
encoder_module_dict.input_tensor.lstm.bias_hh_l0_reverse : [1600]
encoder_module_dict.adverb_input_tensor.embedding.weight : [12, 50]
encoder_module_dict.adverb_input_tensor.lstm.weight_ih_l0 : [1600, 50]
encoder_module_dict.adverb_input_tensor.lstm.weight_hh_l0 : [1600, 400]
encoder_module_dict.adverb_input_tensor.lstm.bias_ih_l0 : [1600]
encoder_module_dict.adverb_input_tensor.lstm.bias_hh_l0 : [1600]
encoder_module_dict.adverb_input_tensor.lstm.weight_ih_l0_reverse : [1600, 50]
encoder_module_dict.adverb_input_tensor.lstm.weight_hh_l0_reverse : [1600, 400]
encoder_module_dict.adverb_input_tensor.lstm.bias_ih_l0_reverse : [1600]
encoder_module_dict.adverb_input_tensor.lstm.bias_hh_l0_reverse : [1600]
decoder_module_dict.adverb_target_tensor.embedding.weight : [9, 400]
decoder_module_dict.adverb_target_tensor.lstm.weight_ih_l0 : [1600, 400]
decoder_module_dict.adverb_target_tensor.lstm.weight_hh_l0 : [1600, 400]
decoder_module_dict.adverb_target_tensor.lstm.bias_ih_l0 : [1600]
decoder_module_dict.adverb_target_tensor.lstm.bias_hh_l0 : [1600]
decoder_module_dict.adverb_target_tensor.join_context.weight : [400, 800]
decoder_module_dict.adverb_target_tensor.hidden_to_output.weight : [9, 400]
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
<summary>Click for architecture details.</summary>
<br>


```
Total parameters: 306625
encoder_module_dict.input_tensor.embedding.weight : [21, 25]
encoder_module_dict.input_tensor.lstm.weight_ih_l0 : [400, 25]
encoder_module_dict.input_tensor.lstm.weight_hh_l0 : [400, 100]
encoder_module_dict.input_tensor.lstm.bias_ih_l0 : [400]
encoder_module_dict.input_tensor.lstm.bias_hh_l0 : [400]
encoder_module_dict.input_tensor.lstm.weight_ih_l0_reverse : [400, 25]
encoder_module_dict.input_tensor.lstm.weight_hh_l0_reverse : [400, 100]
encoder_module_dict.input_tensor.lstm.bias_ih_l0_reverse : [400]
encoder_module_dict.input_tensor.lstm.bias_hh_l0_reverse : [400]
encoder_module_dict.adverb_input_tensor.embedding.weight : [12, 25]
encoder_module_dict.adverb_input_tensor.lstm.weight_ih_l0 : [400, 25]
encoder_module_dict.adverb_input_tensor.lstm.weight_hh_l0 : [400, 100]
encoder_module_dict.adverb_input_tensor.lstm.bias_ih_l0 : [400]
encoder_module_dict.adverb_input_tensor.lstm.bias_hh_l0 : [400]
encoder_module_dict.adverb_input_tensor.lstm.weight_ih_l0_reverse : [400, 25]
encoder_module_dict.adverb_input_tensor.lstm.weight_hh_l0_reverse : [400, 100]
encoder_module_dict.adverb_input_tensor.lstm.bias_ih_l0_reverse : [400]
encoder_module_dict.adverb_input_tensor.lstm.bias_hh_l0_reverse : [400]
decoder_module_dict.adverb_target_tensor.embedding.weight : [9, 100]
decoder_module_dict.adverb_target_tensor.lstm.weight_ih_l0 : [400, 100]
decoder_module_dict.adverb_target_tensor.lstm.weight_hh_l0 : [400, 100]
decoder_module_dict.adverb_target_tensor.lstm.bias_ih_l0 : [400]
decoder_module_dict.adverb_target_tensor.lstm.bias_hh_l0 : [400]
decoder_module_dict.adverb_target_tensor.join_context.weight : [100, 200]
decoder_module_dict.adverb_target_tensor.hidden_to_output.weight : [9, 100]
```

</details>

## Baseline & Augmentation

<details>
<summary>Click for architecture details.</summary>
<br>

```
Total parameters: 5747100
situation_encoder.conv_1.weight : [50, 16, 1, 1]
situation_encoder.conv_1.bias : [50]
situation_encoder.conv_2.weight : [50, 16, 5, 5]
situation_encoder.conv_2.bias : [50]
situation_encoder.conv_3.weight : [50, 16, 7, 7]
situation_encoder.conv_3.bias : [50]
visual_attention.key_layer.weight : [400, 150]
visual_attention.query_layer.weight : [400, 400]
visual_attention.energy_layer.weight : [1, 400]
encoder.embedding.weight : [171, 50]
encoder.lstm.weight_ih_l0 : [1600, 50]
encoder.lstm.weight_hh_l0 : [1600, 400]
encoder.lstm.bias_ih_l0 : [1600]
encoder.lstm.bias_hh_l0 : [1600]
encoder.lstm.weight_ih_l0_reverse : [1600, 50]
encoder.lstm.weight_hh_l0_reverse : [1600, 400]
encoder.lstm.bias_ih_l0_reverse : [1600]
encoder.lstm.bias_hh_l0_reverse : [1600]
enc_hidden_to_dec_hidden.weight : [400, 400]
enc_hidden_to_dec_hidden.bias : [400]
textual_attention.key_layer.weight : [400, 400]
textual_attention.query_layer.weight : [400, 400]
textual_attention.energy_layer.weight : [1, 400]
attention_decoder.queries_to_keys.weight : [400, 800]
attention_decoder.queries_to_keys.bias : [400]
attention_decoder.embedding.weight : [9, 400]
attention_decoder.lstm.weight_ih_l0 : [1600, 1200]
attention_decoder.lstm.weight_hh_l0 : [1600, 400]
attention_decoder.lstm.bias_ih_l0 : [1600]
attention_decoder.lstm.bias_hh_l0 : [1600]
attention_decoder.output_to_hidden.weight : [400, 1600]
attention_decoder.hidden_to_output.weight : [9, 400]
```

</details>

## All other experiments (k=1,5,10,50 and adverb vocabulary=10,50,100,150)
See "Modularity & Augmentation above"
