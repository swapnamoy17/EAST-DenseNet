# convert ICDAR13 train set to test.
import glob
import sys
import os

def get_images(data_path):   #works. files contains path of every image
    files = []
    idx = 0
    for ext in ['jpg', 'png', 'jpeg']:                           #glob finds pathnames consiting of any names ending with 'ext'
        #files.extend(glob.glob(
        #    os.path.join(data_path,'*.{}'.format(ext))))               #Users/hp/Desktop/East-master/ICDAR2013+2015/train_data/*.jpg
        files.extend(glob.glob(data_path+'/'+'*.{}'.format(ext)))
        idx += 1
    return files


def get_text_file(image_file):
    #print(image_file)
    txt_file = image_file.replace(os.path.basename(image_file).split('.')[1], 'txt')   #replaces .jpg with .txt
    #print(txt_file)
    txt_file_name = txt_file.split("/")[-1]
    #print(txt_file_name)
    txt_file = txt_file.replace(txt_file_name, 'gt_' + txt_file_name)
    #print(txt_file)
    return txt_file

print('2')
train_data_path = '/content/drive/My Drive/EAST/ICDAR13/train/'
files = get_images(train_data_path)
print('Number of images: {}'.format(len(files)))
text_files = []
for file in files:
    text_file = get_text_file(file)
    text_files.append(text_file)
print(text_files)
print('Number of text_files: {}'.format(len(text_files)))
for text_file in text_files:
    new_file = '/content/drive/My Drive/EAST/ICDAR13/train2/'+text_file.split('/')[-1]
    print(new_file)
    f1 = open(new_file,'w')
    with open(text_file,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ',maxsplit=4)
            f1.write('{}, {}, {}, {}, {}'.format(line[0],line[1],line[2],line[3],line[4]))
    f.close()
    f1.close()
            
    