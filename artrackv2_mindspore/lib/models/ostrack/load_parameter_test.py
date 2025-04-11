import torch
import sys
sys.path.append("/home/djh/python-code/Artrackv2/2stage")
from mindspore import Tensor
from mindspore import save_checkpoint
def get_keymap_txt(pth_file):
    # 如果是tar压缩文件。需要执行下面这段代码
    checkpoint = torch.load(pth_file,map_location="cpu")
    state_dict = checkpoint['net']
    # end

    # # 否则就是执行下面这段代码
    # map_path = pth_file.split('.')[0] + '_key_map.txt'
    # map_file = open(map_path, 'w')
    # state_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    # if 'model_state' in state_dict:
    #     state_dict = state_dict['model_state']
    # elif 'module' in state_dict:
    #     state_dict = state_dict['module']
    # elif 'model' in state_dict:
    #     state_dict = state_dict['model']
    # # end
    list = []
    dict = {}
    for name,value in state_dict.items():
        print(name)
        if name == "cross_2_decoder.decoder_blocks.0.norm1.weight":
            print(state_dict[name])
        new_name = name.replace("norm1.weight","norm1.gamma")
        new_name = new_name.replace("norm1.bias","norm1.beta")
        new_name = new_name.replace("norm2.weight","norm2.gamma")
        new_name = new_name.replace("norm2.bias","norm2.beta")
        new_name = new_name.replace("decoder_norm.weight","decoder_norm.gamma")
        new_name = new_name.replace("decoder_norm.bias","decoder_norm.beta")
        new_name = new_name.replace("word_embeddings.weight", "word_embeddings.embedding_table")
        new_name = new_name.replace("position_embeddings.weight", "position_embeddings.embedding_table")
        new_name = new_name.replace("prev_position_embeddings.weight", "prev_position_embeddings.embedding_table")
        new_name = new_name.replace("norm.weight","norm.gamma")
        new_name = new_name.replace("norm.bias","norm.beta")
        list.append(new_name)
    ms_params_list=[]
    for name,value in dict.items():
        param_dict={}
        param_dict['name'] = name
        param_dict['data'] = Tensor(value.numpy())
        if name=="backbone.pos_embed_z0":
            print(param_dict['data'])
        ms_params_list.append(param_dict)
    # 要生成转换文件的话，就执行下面这句注释代码
    # save_checkpoint(ms_params_list, "/home/djh/python-code/Artrackv2/checkpoint1.ckpt")
pth_file = "/mnt/d/Download/OSTrack_ep0030.pth.tar"
get_keymap_txt(pth_file)
