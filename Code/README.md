# Demo the code
The experiments can only be run with the real data (and to see how see the file Scripts/README.md), but find below how to demo the code for both the modular model and the baseline.

## Baseline

### Train

Go to the folder `multimodal_seq2seq_gSCAN`, install the dependencies with `setup.py`, and run the below command to demo the baseline training.
```
python3.8 -m seq2seq --k=0 --mode=train --max_decoding_steps=120 --max_testing_examples=5 \
  --data_directory=data/demo_dataset --attention_type=bahdanau --no_auxiliary_task \
  --generate_vocabularies --conditional_attention --training_batch_size=2 \
  --model_size=small --max_training_iterations=100 --print_every=1 --evaluate_every=10  --output_directory=demo
```

### Test

```
python3.8 -m seq2seq --k=0 --mode=test --resume_from_file=demo/model_best.pth.tar --max_decoding_steps=120 --max_testing_examples=5 \
  --data_directory=data/demo_dataset --attention_type=bahdanau --no_auxiliary_task \
  --generate_vocabularies --conditional_attention --training_batch_size=2 \
  --model_size=small --max_training_iterations=100 --print_every=1 --evaluate_every=10 --output_directory=demo
```

This should give the output (if you trained a model with the code above):

```
2022-01-29 18:28 Loading checkpoint from file at 'demo/model_best.pth.tar'
2022-01-29 18:28 Loaded checkpoint 'demo/model_best.pth.tar' (iter 11)
2022-01-29 18:28 Predicted for 4 examples.
2022-01-29 18:28 Done predicting in 0.02988719940185547 seconds.
2022-01-29 18:28 Wrote predictions for 4 examples.
2022-01-29 18:28 Average exact match: 1.0
2022-01-29 18:28 Saved predictions to demo/test_predict.json
```

## Modular

### Train

Go to the folder `Modular_gSCAN`, install the dependencies with `setup.py`, and run the below command to demo the position module training.

```
python3.8 -m Modularity --mode=train --data_directory=data/demo_dataset \
      --generate_vocabularies --max_decoding_steps=120 --module=position --max_training_iterations=100 \
      --simplified_architecture=False --use_attention=no --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --training_batch_size=2 --weight_decay=0. --k=5 --model_size=small --simplified_architecture=no \
      --output_directory=position_demo --k=0 --print_every=1 --evaluate_every=10
```

### Test

```
python3.8 -m Modularity --mode=test --data_directory=data/demo_dataset \
      --generate_vocabularies --max_decoding_steps=120 --module=position --max_training_iterations=100 \
      --simplified_architecture=False --use_attention=no --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --training_batch_size=2 --weight_decay=0. --k=5 --model_size=small --simplified_architecture=no \
      --output_directory=position_demo --k=0 --print_every=1 --evaluate_every=10 --resume_from_file=position_demo/model_best.pth.tar

```

This should give (if you trained a model with the code above):
```
2022-01-29 18:26 Loading checkpoint from file at 'position_demo/model_best.pth.tar'
2022-01-29 18:26 Loaded checkpoint 'position_demo/model_best.pth.tar' (iter 11)
2022-01-29 18:26 Converting dataset to tensors...
2022-01-29 18:26 Predicted for 4 examples.
2022-01-29 18:26 Done predicting in 0.6675999164581299 seconds.
2022-01-29 18:26 Wrote predictions for 4 examples.
2022-01-29 18:26 Average agent_position: 100.00
2022-01-29 18:26 Average target_position: 100.00
2022-01-29 18:26 Average agent_direction: 100.00
```

