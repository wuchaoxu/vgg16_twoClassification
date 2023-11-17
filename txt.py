import os
from os import getcwd

classes=['cat','dog']
sets=['train']

if __name__=='__main__':
    wd=getcwd()
    #os.getcwd()是Python中的一个函数，它返回当前工作目录的字符串表示形式
    #工作目录是指当前正在运行的程序所在的目录。如果您需要在程序中访问文件或文件夹，但不想使用绝对路径，
    # 那么可以使用os.getcwd()来获取当前工作目录并构建相对路径。

    list_file=open('cls_'+ sets[0] +'.txt','w')

    types_name=os.listdir('train')#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    for type_name in types_name:
        if type_name not in classes:
            continue
        cls_id=classes.index(type_name)#输出0-1
        photos_path=os.path.join(sets[0],type_name)
        #os.path.join()是Python中的一个函数，它可以将多个路径组合成一个路径字符串
        photos_name=os.listdir(photos_path)
        for photo_name in photos_name:
            _,postfix=os.path.splitext(photo_name)
            #该函数用于分离文件名与拓展名,如果文件路径中包含扩展名，则返回的元组的第二个元素是扩展名，否则为空字符串。
            if postfix not in['.jpg','.png','.jpeg']:
                continue
            list_file.write(str(cls_id)+';'+'%s/%s'%(wd, os.path.join(photos_path,photo_name)))
            list_file.write('\n')
    list_file.close()


