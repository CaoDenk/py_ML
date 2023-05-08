import configparser
import os
def load_config():
    config=configparser.ConfigParser()
    dir=os.path.dirname(__file__)
    grand_dir=os.path.dirname(dir)
    config_path=grand_dir+"/config/config.ini"
    # # with open(config_path,"r") as f:
        
    # #     str=f.readline()
    # #     print(str)
    # #     config.read_file(f)
    # #     print(config.get("Img_seg","file"))
    # config = configparser.ConfigParser(allow_no_value=True)
    # config.read('config.ini')
    # file=config.get("Img_seg","file")
    # print(file)
    config = configparser.ConfigParser()

    # 读取配置文件
    config.read(config_path)

    # 获取配置值

    file = config.get('img_seg', 'file')

    is_depth=config.get('img_seg','is_depth')
    # 输出配置值
    
    print(f'file: {file}')   
    print(f'is_depth:{is_depth}')


if __name__=="__main__": 
    load_config()
# print(__file__)

# p=os.path.dirname(__file__)
# p=os.path.dirname(p)
# print(p+"/config/config.conf")


# 创建ConfigParser对象
