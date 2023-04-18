import PIL
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import ImageFont, ImageDraw, Image
import torch
import cv2
import numpy as np
import torch


def plot_image(image, lable, title):
    
    fig=plt.figure()
    for i in range(6):
        fig.add_subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(image[i][0]*0.3081+0.1307, cmap='gray',interpolation='none')
        plt.title("{}:{}".format(title,lable[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_curve(data):
    fig=plt.figure()
    fig.plot(range(len(data)),data,color='blue')
    plt.legend('step')
    plt.xlabel('step')
    plt.ylabel('value') 
    plt.show()

def one_hot(lable,depth=10):
    out=torch.zeros(lable.size(0),depth)
    idex=torch.LongTensor(lable).view(-1,1)
    out.scatter_(dim=1,index=idex,value=1)
    return out


def mat_to_tensor(img:cv2.Mat)->torch.tensor:
    image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_np = np.transpose(image_np, (2, 0, 1))

    return torch.from_numpy(image_np).float()
    


def Image_to_tensor(img:PIL.Image)->torch.Tensor:
    transform = transforms.ToTensor()
    return transform(img)

def mat_to_numpy(mat:cv2.Mat)->np.ndarray:
    return np.asarray(mat)
    


"""
@param img_mat 输入一个cv2.mat   
@return  返回数组，储存灰度值的像素个数的

"""
def count_pixel(img_mat:cv2.Mat)->cv2.Mat:
    
    img_mat_grey=cvtcolor2gray(img_mat)
    img_tensor:torch.Tensor = mat_to_tensor(img_mat_grey)
    gray_value_list:list = [0 for i in range(256)]
    _,h,w=img_tensor.shape
    for i in range(h):
        for j in range(w):
            val_tensor:torch.Tensor = img_tensor[0,i,j]*255
            val_number = val_tensor.item()
            val_int = int(val_number)
            gray_value_list[val_int] += 1
    return gray_value_list
    

"""
@param 传入一个数组,长度是256
@return 返回灰度直方图
"""
def to_gray_histogram (arr:list,img_size=(600,800),line_color=(255,0,0),thickness:int=3)->cv2.Mat:
    start_point:tuple=(15,img_size[0]-30)
    max=np.max(arr)
    ratio=max/(start_point[1]*0.9)
    narr=np.zeros(img_size,np.uint8)
    mat=cv2.Mat(narr)
    
    for i in range(0,256):
        current_start_point=(start_point[0]+i*thickness,start_point[1])
        end_point=(current_start_point[0],current_start_point[1]-int(arr[i]/ratio))
        mat=cv2.line(mat,current_start_point,end_point,line_color,thickness)
        if i%10==0:
            font_start_point=(current_start_point[0],current_start_point[1]+20)
            mat = cv2.putText(mat,str(i),font_start_point,cv2.FONT_HERSHEY_SIMPLEX,0.4,line_color)
    return mat


def cvtcolor2gray(img_mat:cv2.Mat)->cv2.Mat:
    
    if len(img_mat.shape)==3:
        return cv2.cvtColor(img_mat,cv2.COLOR_BGR2GRAY)
    return img_mat




"""

@link= https://stackoverflow.com/questions/50854235/how-to-draw-chinese-text-on-the-image-using-cv2-puttextcorrectly-pythonopen
"""
def mat_merge_with_labels(mats,lables,allignment_type):
   
    height,width,channels=mats[0].shape
    count=len(mats)
    mat_lists=[]
    for i in range(len(mats)):
        # n=np.zeros((50,width,channels),np.uint8)
        label_header=create_mat((40,width),channels=channels)
        # mat_with_label=cv2.putText(label_header,lables[i],(10,20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255))
        img_n=mat_to_numpy(mats[i])
        img=np.concatenate([label_header,img_n])
        
        fontpath = "simsun.ttc" # <== 这里是宋体路径 
        font = ImageFont.truetype(fontpath, 32)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((0, 0),  lables[i], font = font, fill = ( 0,0,0,0))
        img = np.array(img_pil)

        # cv2.imshow(str(i),img)
        
        mat_lists.append(img)
    
    #储存返回的大图
    ret=[]
    
    allign_height,allign_width=allignment_type
    # count=len(mats)
    for i in  range(allign_height):        
        
        if (final_count:=count-i*allign_width) >=allign_width:          
            mat_concat=cv2.hconcat( mat_lists[i*allign_width:i*allign_width+allign_width])
            ret.append(mat_concat)
        else:
            # for k in range(leaveout:=v_len-final_count):
            leaveout=allign_width-final_count
            _height = mat_lists[0].shape[0]
            white_mat=create_mat((_height,width*leaveout),channels=channels)
                         
            tmp=cv2.hconcat( mat_lists[i*allign_width:i*allign_width+final_count])
            final_line_mat=cv2.hconcat([tmp,white_mat])
            ret.append(final_line_mat)
    return cv2.vconcat(ret)


"""
@return 返回一个大小为size的cv::Mat 

def create_mat(size:tuple)->cv2.Mat:
    matArr_with_zeros=np.zeros(size,np.uint8)
    return cv2.Mat(matArr_with_zeros)

"""

def create_mat(size,channels):
    if channels==1:
        n =np.zeros(size,dtype='uint8')
        return cv2.Mat(n)
    elif channels==3:
        l=[]
        for i in range(3):
            l.append(np.ones(size,dtype='uint8')*255)          
        return cv2.merge(l)
    else:
        raise Exception("channels must be 1 or 3")



# def get_histogram_from_str(img_path:str)->cv2.Mat:
#     graypixel_list=utils.count_pixel(cv2.imread(img_path))
#     return utils.to_histogram(graypixel_list)
# def get_histogram_from_mat(img_mat:cv2.Mat)->cv2.Mat:
#     graypixel_list=utils.count_pixel(img_mat)
#     return utils.to_histogram(graypixel_list)


# def match_img(img1,img2):
#     Image.merge(img1,img2)
""""
@return 返回一个掩膜后的cv.Mat
@param 
    img:torch.Tensor
"""
def do_mask(img:torch.Tensor,mask:torch.Tensor)->cv2.Mat:
    ...