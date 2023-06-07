import json5
import torch

def load_cam_pos_args(json_file):
    with open(json_file,"r") as f:
        js=json5.load(f)
        # print(js)
        # for k in js:
        #     print(k)
        #     print(type(k))    
        
        l=js["camera-pose"]
        t=torch.zeros((4,4),dtype=torch.float32)
        i=0
        for ll in l:
            # print(ll)
            t[i]=torch.Tensor(ll)
            i+=1
        return t
        
if __name__=='__main__':
    file=r"E:\2019\dataset_1\keyframe_1\data\frame_data000000.json"
    pos=load_cam_pos_args(file)
    print(pos)