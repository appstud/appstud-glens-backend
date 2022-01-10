import cv2
import csv
import pdb
import shutil

def copyImages(path):
    labels=['Bald','Gray_Hair','Brown_Hair','Blond_Hair','Black_Hair','Wearing_Hat']
    source='/home/appstud/Downloads/celeba-dataset/img_align_celeba/img_align_celeba/'
    dst_dir='./celeba/'

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            print(row)
            if line_count == 0:
                data=[(i,x)  for i,x in enumerate(row) for attribut in labels if x==attribut] 
                line_count += 1
            else:
                for (i,attribut) in data:
                    if(int(row[i])==1):
                         shutil.copy(source+row[0], dst_dir+attribut)
                        
                #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                line_count += 1
     #print(f'Processed {line_count} lines.')


def prepareBeardDataset(path):
    labels=['No_Beard']
    source='/home/appstud/Downloads/celeba-dataset/img_align_celeba/img_align_celeba/'
    dst_dir='./beardDataset/'

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            print(row)
            if line_count == 0:
                data=[(i,x)  for i,x in enumerate(row) for attribut in labels if x==attribut] 
                pdb.set_trace()
                line_count += 1
            else:
                for (i,attribut) in data:
                    if(int(row[21])==1):
                        print("aa")
                        if(int(row[25])==1):
                             shutil.copy(source+row[0], dst_dir+"nobeard")
                        else:
                             shutil.copy(source+row[0], dst_dir+"beard")
                            
                #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                line_count += 1
     #print(f'Processed {line_count} lines.')


if(__name__=="__main__"):
    #copyImages('/home/appstud/Downloads/celeba-dataset/list_attr_celeba.csv')
    prepareBeardDataset('/home/appstud/Downloads/celeba-dataset/list_attr_celeba.csv')
