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
    db_host = config.get('database', 'host')
    db_port = config.getint('database', 'port')
    db_name = config.get('database', 'name')
    db_user = config.get('database', 'username')
    db_pass = config.get('database', 'password')
    file = config.get('database', 'file')

    # 输出配置值
    print(f'Database host: {db_host}')
    print(f'Database port: {db_port}')
    print(f'Database name: {db_name}')
    print(f'Database user: {db_user}')
    print(f'Database password: {db_pass}')   
    print(f'Database password: {file}')   
    
load_config()
# print(__file__)

# p=os.path.dirname(__file__)
# p=os.path.dirname(p)
# print(p+"/config/config.conf")


# 创建ConfigParser对象
