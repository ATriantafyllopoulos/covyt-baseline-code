features=(
    "eGeMAPSv02"
    "wav2vec2-large-xlsr-53-features"
    "ComParE"
)
partitions=(
    "language.disjoint"
    "speaker.disjoint"
    "file.disjoint"
    "speaker.inclusive"
)
dataset=$1
results=$1
for feature in ${features[@]}; do
for partition in ${partitions[@]}; do
    python cross_validation.py \
        --dataset "$dataset/dataset" \
        --features "$dataset/features/${feature}.csv" \
        --partitioning $partition \
        --results "$results/${partition}/${feature}/"
    done
done

python evaluate.py \
    $results \
    --dataset "$dataset/dataset" \