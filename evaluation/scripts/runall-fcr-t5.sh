#!/bin/sh

if [ "$DEBUG" = "1" ]; then
	DEBUG_FLAGS="--num_epochs 2 --max_samples 100"
else
	DEBUG_FLAGS=""
fi

echo "==== RUNNING T5-Large"
python evaluation/classifier_t5.py \
	--output_path output/scripttest \
	--output_name fcr-t5-large \
	--config ./config/fcr-t5-large.json \
	$DEBUG_FLAGS
printf "\n\n"

echo "==== RUNNING T5-Large Seq"
python evaluation/classifier_t5_seq.py \
	--output_path output/scripttest \
	--output_name fcr-t5-seq-large \
	--config ./config/fcr-t5-large-seq.json \
	$DEBUG_FLAGS
printf "\n\n"

echo "==== RUNNING FLAN-T5-Large"
python evaluation/classifier_t5.py \
	--output_path output/scripttest \
	--output_name fcr-flan-t5-large \
	--config ./config/fcr-flan-t5-large.json \
	$DEBUG_FLAGS
printf "\n\n"

echo "==== RUNNING FLAN-T5-Large Seq"
python evaluation/classifier_t5_seq.py \
	--output_path output/scripttest \
	--output_name fcr-flan-t5-seq-large \
	--config ./config/fcr-flan-t5-large-seq.json \
	$DEBUG_FLAGS
