from modelscope.hub.snapshot_download import snapshot_download

# Download the model into the local `models/bge-m3` cache directory.
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='D:/ai_models/modelscope_cache/models')
print(f"Model downloaded to: {model_dir}")
