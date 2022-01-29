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

## Modular
