#!/bin/bash
if [ ! -d "prototypes_init" ]; then
  mkdir "prototypes_init"
  echo "prototypes_init folder created."
else
  echo "prototypes_init folder already exists."
fi
data_list=(
"ArTaxOr"
"clipart1k"
"DIOR"
"UODD"
"NEUDET"
"FISH"
)
shot_list=(
1
5
10
)
model_list=(
"l"
)
for model in "${model_list[@]}"; do
  for dataset in "${data_list[@]}"; do
    for shot in ${shot_list[@]}; do
      python3 ./tools/extract_instance_prototypes.py   --dataset ${dataset}_${shot}shot --out_dir prototypes_init --model vit${model}14 --epochs 1 --use_bbox yes --without_mask True
      echo "extract_instance_prototypes with vit${model} for ${shot}shot ${dataset} done, save at prototypes_init dir."
      python3 ./tools/run_sinkhorn_cluster.py  --inp  prototypes_init/${dataset}_${shot}shot.vit${model}14.bbox.pkl  --epochs 30 --momentum 0.002    --num_prototypes ${shot}
      echo "run_sinkhorn_cluster with vit${model} for ${shot}shot ${dataset} done, save at prototypes_init dir."
    done
  done
done
