env -i HOME="$HOME" \
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin" \
LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu" \
HF_HOME=/mnt/data3/VoiceModels/huggingface \
HUGGINGFACE_HUB_CACHE=/mnt/data3/VoiceModels/huggingface \
TRANSFORMERS_CACHE=/mnt/data3/VoiceModels/huggingface \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/mintdude/venvs/higgs-py310/bin/python -m uvicorn tools.higgs_local_server:app \
  --host 0.0.0.0 --port 8010

