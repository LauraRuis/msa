########################################################################################
#                                                                                      #
#    The command to run for each experiment in the paper.                              #
#                                                                                      #
#                                                                                      #
########################################################################################

# Baseline
python3.8 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=3500 \
  --data_directory=data/modular_0_extra_bigger_full --attention_type=bahdanau --no_auxiliary_task \
  --generate_vocabularies --conditional_attention --training_batch_size=200 \
  --model_size=small --max_training_iterations=200000

# Baseline & Augmentation
python3.8 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=3500 \
  --data_directory=data/modular_150_extra_full --attention_type=bahdanau --no_auxiliary_task \
  --generate_vocabularies --conditional_attention --training_batch_size=200 \
  --model_size=big --max_training_iterations=200000