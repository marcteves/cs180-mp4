#! /bin/bash

source config.sh

> "$TEST_LIST"
> "$TRAIN_LIST"
> "$TEST_TAGS"
> "$TRAIN_TAGS"

tag=0
# exclude this IMAGES_DIR from output
for i in $(find "$IMAGES_DIR" ! -path "$IMAGES_DIR" -type d | sort -V)
do
	find "$i" -type f | sort -V | head -n 6 >> "$TRAIN_LIST"
	find "$i" -type f | sort -V  >> "$TEST_LIST"

	# apply tags
	printf "$tag%.0s\n" {1..6} >> "$TRAIN_TAGS"
	printf "$tag%.0s\n" {1..4} >> "$TEST_TAGS"
	((tag++))
done

diff --new-line-format="%L" --unchanged-line-format="" \
	"$TRAIN_LIST" "$TEST_LIST" > "$TEST_LIST".$$
mv "$TEST_LIST".$$ "$TEST_LIST"
