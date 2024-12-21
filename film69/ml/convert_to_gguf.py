import platform,os

def convert_to_gguf(dir_model:str,output_name:str=None,quantization_method:list[str]= ["q3_k_l","q4_k_m","q5_k_m","q8_0","f16"]):
    if platform.system()=="Linux":re=os.popen("ls").read().split("\n")
    else:re=os.popen("dir /b").read().split("\n")

    if not "llama.cpp" in re:
        os.system("git clone https://github.com/ggerganov/llama.cpp")
        with open(os.devnull, 'w') as devnull:os.system("cd ./llama.cpp && pip install -r requirements.txt > /dev/null 2>&1")
    
    for i in quantization_method:
        command = f"python llama.cpp/convert_hf_to_gguf.py {dir_model} --outfile {output_name if output_name !=None else dir_model}.{i.upper()}.gguf --outtype {i}"
        os.system(command)

if __name__=="__main__":
    convert_to_gguf("dir_model","file_name",["q8_0"])