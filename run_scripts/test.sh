cd ../
model=(meta-llama/Meta-Llama-3.1-8B-Instruct google/gemma-2-9b-it mistralai/Mistral-Nemo-Instruct-2407)
dataset=(ACTER ACL-RD BCGM)
num_shots=(10)
retrieval_method=(default default_w_ins bm25 random fastkassim)   
seed=(42)

for i in "${model[@]}"; do
    for j in "${dataset[@]}"; do
        for l in "${num_shots[@]}"; do
            for m in "${retrieval_method[@]}"; do
                for t in "${seed[@]}"; do
                    CUDA_VISIBLE_DEVICES=4,5,6 python main.py \
                    --config_path configs/test.json \
                    --model_name $i \
                    --dataset_name $j \
                    --num_shots $l \
                    --retrieval_method $m \
                    --seed $t
                done
            done
        done
    done
done
