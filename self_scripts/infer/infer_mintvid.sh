export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

find_free_port() {
    local port=29500
    while true; do
        if ! ss -tuln | grep -q ":$port " && ! lsof -i :$port > /dev/null 2>&1; then
            echo $port
            return
        fi
        ((port++))
    done
}

export FPS=3
export DURATION=5.0

## general
for dataset in mintvid_jimeng mintvid_kling mintvid_seedance mintvid_sora2 mintvid_wan25 mintvid_hailuo
do
    python infer_offline_batch_system.py \
    --data_name ${dataset} \
    --model_path /path/to/VideoVeritas
done

## face
for dataset in mintvid_fantasy mintvid_omniavatar mintvid_phantom
do
    python infer_offline_batch_system.py \
    --data_name ${dataset} \
    --model_path /path/to/VideoVeritas
done

## fact
for dataset in mintvid_fact
do
    python infer_offline_batch_system.py \
    --data_name ${dataset} \
    --model_path /path/to/VideoVeritas
done
