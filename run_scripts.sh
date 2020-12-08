#! /bin/bash

# create environment
#$conda env create --file=pretrain_tf.yml

# activate environment
#$conda activate pretrain_tf

# get bert base uncased
curl --output uncased_L-12_H-767_A-12.zip https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

# unzip bert base
unzip uncased_L-12_H-767_A-12.zip

# convert domain specific .txt into train/val for additional MLM pretraining
python3 codes/further-pre-training/create_pretraining_data.py --input_file=pt_medium.txt \
                       --output_file=tmp/pt_medium.tfrecord \
                       --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
                       --do_lower_case=True --max_seq_length=128 \
                       --max_predictions_per_seq=20 --masked_lm_prob=0.15 \
                       --random_seed=12345 --dupe_factor=5

# perform additional domain specific MLM pretraining
python3 codes/further-pre-training/run_pretraining.py --input_file=tmp/pt_medium.tfrecord  \
                         --output_dir=uncased_L-12_H-768_A-12_pt_medium_pretrain --do_train=True   \
                         --do_eval=False --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json   \
                         --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt   \
                         --train_batch_size=16  --max_seq_length=128  --max_predictions_per_seq=20  \
                         --num_train_steps=20  --num_warmup_steps=5  --save_checkpoints_steps=10  \
                         --learning_rate=5e-5

# convert tf model (with additionally pretrained weights) to pytorch
python3 codes/fine-tuning/convert_tf_checkpoint_to_pytorch.py \
                    --tf_checkpoint_path=uncased_L-12_H-768_A-12_pt_medium_pretrain/model.ckpt-20   \
                    --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json   \
                    --pytorch_dump_path ./models/pytorch_model.bin

# upload the final pytorch model to gcp