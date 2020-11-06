import json
import os
# patient101_ED.nii.gz


def write_json(obj, fpath):
    with open(fpath,'w') as f:
        json.dump(obj,f,indent=4)


data_list = '/home/laisong/github/Unet_Cardiac/Torch_Unet/libs/datasets/acdcjson/ACDCDataList_forupload.json'
with open(data_list, 'r') as f:
    data_path = json.load(f)
print(len(data_path))
TPE = ['ED','ES']
ID = range(101,151)
total_mode = [[] for x in range(0,100)]
for i in range(len(data_path)):
    path = data_path[i]
    mode_index = 0
    for index_id,id in enumerate(ID):
        for index_tye,tye in enumerate(TPE):
            mode = str(id) + '_' + tye
            if mode in path:
                total_mode[mode_index].append(path)
                continue
            mode_index += 1
    mode_index = 0
    for index_id,id in enumerate(ID):
        for index_tye,tye in enumerate(TPE):
            json_path = os.path.join('/home/laisong/github/Unet_Cardiac/'
                                     'Torch_Unet/libs/datasets/acdcjson/split_json',str(id)+tye+'.json')
            write_json(total_mode[mode_index],json_path)
            mode_index += 1
