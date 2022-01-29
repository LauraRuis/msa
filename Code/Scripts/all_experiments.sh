########################################################################################
#                                                                                      #
#    The command to run for each experiment in the paper.                              #
#                                                                                      #
#                                                                                      #
########################################################################################

# Modularity & Augmentation - Position Module
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_0_extra_bigger_full \
      --generate_vocabularies --max_decoding_steps=120 --module=position --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=no --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --training_batch_size=200 --weight_decay=0. --k=5 --model_size=small

# Modularity & Augmentation - Navigation Module
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_0_extra_bigger_full \
      --generate_vocabularies --max_decoding_steps=120 --module=planner --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=no --k=5 --model_size=small

# Modularity & Augmentation - Interaction Module
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_0_extra_bigger_full \
      --generate_vocabularies --max_decoding_steps=120 --module=interaction --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=yes --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=input_tensor --conditional_attention_values_key=world_state_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=no --k=5 --model_size=small

# Modularity - Transformation Module
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_150_extra_full \
      --generate_vocabularies --max_decoding_steps=120 --module=adverb_transform --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=adverb_input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=yes --k=5 --model_size=big

# Modularity - Position Module
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_0_extra_bigger_full \
      --generate_vocabularies --max_decoding_steps=120 --module=position --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=no --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --training_batch_size=200 --weight_decay=0. --k=5 --model_size=small

# Modularity - Navigation Module
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_0_extra_bigger_full \
      --generate_vocabularies --max_decoding_steps=120 --module=planner --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=no --k=5 --model_size=small

# Modularity - Interaction Module
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_0_extra_bigger_full \
      --generate_vocabularies --max_decoding_steps=120 --module=interaction --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=yes --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=input_tensor --conditional_attention_values_key=world_state_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=no --k=5 --model_size=small

# Modularity - Transformation Module
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_0_extra_bigger_full \
      --generate_vocabularies --max_decoding_steps=120 --module=adverb_transform --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=adverb_input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=yes --k=5 --model_size=small

# Modularity & Augmentation - Transformation Module - k=1
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_150_extra_full \
      --generate_vocabularies --max_decoding_steps=120 --module=adverb_transform --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=adverb_input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=yes --k=1 --model_size=big

# Modularity & Augmentation - Transformation Module - k=10
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_150_extra_full \
      --generate_vocabularies --max_decoding_steps=120 --module=adverb_transform --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=adverb_input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=yes --k=10 --model_size=big

# Modularity & Augmentation - Transformation Module - k=50
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_150_extra_full \
      --generate_vocabularies --max_decoding_steps=120 --module=adverb_transform --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=adverb_input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=yes --k=50 --model_size=big

# Modularity & Augmentation - Transformation Module - adverb vocabulary size 10
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_10 \
      --generate_vocabularies --max_decoding_steps=120 --module=adverb_transform --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=adverb_input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=yes --k=5 --model_size=big

# Modularity & Augmentation - Transformation Module - adverb vocabulary size 50
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_50 \
      --generate_vocabularies --max_decoding_steps=120 --module=adverb_transform --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=adverb_input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=yes --k=5 --model_size=big

# Modularity & Augmentation - Transformation Module - adverb vocabulary size 100
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_100 \
      --generate_vocabularies --max_decoding_steps=120 --module=adverb_transform --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=adverb_input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=yes --k=5 --model_size=big

# Modularity & Augmentation - Transformation Module - no cautiously type
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_150_extra_full \
      --generate_vocabularies --max_decoding_steps=120 --module=adverb_transform --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=adverb_input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=yes --k=5 --model_size=big \
      --isolate_adverb_types=nonmovement_direction,movement,movement_rewrite

# Modularity & Augmentation - Transformation Module - only cautiously type
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_150_extra_full \
      --generate_vocabularies --max_decoding_steps=120 --module=adverb_transform --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=adverb_input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=yes --k=5 --model_size=big \
      --isolate_adverb_types=nonmovement_first_person

# Modularity & Augmentation - Transformation Module - one cautiously type
srun python3.8 -m Modularity --mode=train --data_directory=data/modular_150_extra_full \
      --generate_vocabularies --max_decoding_steps=120 --module=adverb_transform --max_training_iterations=200000 \
      --simplified_architecture=False --use_attention=yes --use_conditional_attention=no --upsample_isolated=100 \
      --type_attention=luong --attention_values_key=adverb_input_tensor \
      --training_batch_size=200 --weight_decay=1e-5 --collapse_alo=yes --k=5 --model_size=big \
      --isolate_adverb_types=None --only_keep_adverbs=adverb_13


