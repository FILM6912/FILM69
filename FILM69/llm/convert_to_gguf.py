import platform,os
import shutil

def __check_file__(path):
        """Checks files in a directory and returns their paths and sizes."""
        files_path = []
        files_size = []
        for root, dirs, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_size = os.path.getsize(file_path)
                file_size_gb = file_size / (1024**3)
                files_path.append(file_path)
                files_size.append(file_size_gb)
        return files_path, files_size
    
def _convert_to_gb( size_str):
        """Converts a size string (e.g., '49G') to GB."""
        unit_multipliers = {"M": 1 / 1024, "G": 1}
        num = float(size_str[:-1])
        unit = size_str[-1]
        return num * unit_multipliers.get(unit, 1)
    
def convert_to_gguf(
        path_model:str,
        path_output:str=None,
        quantization_method:list[str]= ["q4_0","q8_0"],
        max_size_gguf="49GB",
        build_gpu=False,
        save_original_gguf=False,
        install_req=False):
    

    p="./"
    max_size_gguf=max_size_gguf[:-1]
    
    if platform.system()=="Linux":re=os.popen("ls").read().split("\n")
    else:re=os.popen("dir /b").read().split("\n")

    folder_path = f"{path_model}/GGUF"
    os.makedirs(folder_path, exist_ok=True)

    if not "llama.cpp" in re:
        os.system("git clone https://github.com/ggerganov/llama.cpp.git")
        
        with open(os.devnull, 'w') as devnull:
            print("Installing llama.cpp...")
            
            req='&& pip install -r requirements.txt' if install_req else ""
            build_gpu_command = "-DGGML_CUDA=ON"
            command = f"""
                cd llama.cpp && \
                cmake -B build {build_gpu_command if build_gpu else ''} && \
                cmake --build build --config Release -j 8 \
                {req} > /dev/null 2>&1"
                """
            os.system(command)
            
            print("Done!")    
    
    command = f"python {p}llama.cpp/convert_hf_to_gguf.py {path_model} --outfile {path_output if path_output !=None else folder_path+'/'+path_model}.F16.gguf"
    os.system(command)
    
    for i in quantization_method:
        if i.upper() != "F16":
            command = f'{p}llama.cpp/build/bin/llama-quantize\
                {path_output if path_output !=None else folder_path+'/'+path_model}.F16.gguf\
                {path_output if path_output !=None else folder_path+'/'+path_model}.{i.upper()}.gguf {i.upper()}'
                
            os.system(command)
            
    files_path, files_size = __check_file__(folder_path)

    if max(files_size) > _convert_to_gb(max_size_gguf):
        for i in files_path:
            new_path = os.path.join(folder_path, i.split(".")[-2])
            os.makedirs(new_path, exist_ok=True)
            shutil.move(i, os.path.join(new_path, i.split("/")[-1]))


    files_path, files_size = __check_file__(folder_path)
    for i in range(len(files_path)):
        if files_size[i] > _convert_to_gb(max_size_gguf):
            command = f"""{p}llama.cpp/build/bin/llama-gguf-split --split \
                --split-max-size {max_size_gguf}\
                {files_path[i]} {files_path[i][:-5]}
            """
            os.system(command)
            if not save_original_gguf:
                os.remove(files_path[i])
            
if __name__=="__main__":
    convert_to_gguf("path_model","file_name",["q8_0"])