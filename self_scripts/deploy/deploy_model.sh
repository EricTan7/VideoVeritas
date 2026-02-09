model_path=/path/to/VideoVeritas

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ${model_path} \
--port 8000 \
--host 0.0.0.0 \
--dtype bfloat16 \
--tensor-parallel-size 4 \
--pipeline-parallel-size 1 \
--limit-mm-per-prompt '{"image": 1, "video": 1}' \
--media-io-kwargs '{"video": {"num_frames": -1}}' \
--trust_remote_code
