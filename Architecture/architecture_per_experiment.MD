# Architecture details per experiment

## Baseline

## Modularity & Augmentation

### Position Module

### Navigation Module

### Interaction Module

### Adverb Module

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

## Modularity

### Position Module

### Navigation Module

### Interaction Module

### Adverb Module

## Baseline & Augmentation

## All other experiments (k=1,5,10,50 and adverb vocabulary=10,50,100,150)
See "Modularity & Augmentation above"
