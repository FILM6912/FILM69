from huggingface_hub import get_safetensors_metadata

def requires_memory(model_id="deepseek-ai/DeepSeek-R1"):
    dtype_bytes = {"F32": 4, "F16": 2, "BF16": 2, "F8": 1}
    metadata = get_safetensors_metadata(model_id)
    memory = (sum(count* dtype_bytes [key.split("_")[0]] for key, count in metadata.parameter_count.items())/ (1024**3)
    *1.18)
    return "model_id\t: {model_id} \nrequires memory\t: {memory}GB"