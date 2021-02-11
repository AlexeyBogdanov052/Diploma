from imutils import paths
import random
import json

def CreateFile(filename, list):
    f = open(filename, 'w')
    for l in list:
        f.write(l["name"] + ";" + str(l["label"]) + '\n')
    f.close()

img_list1 = list(paths.list_images('C:/Diploma/imgDiploma/manipulated_sequences_cropped/FaceSwap'))
img_list2 = list(paths.list_images('C:/Diploma/imgDiploma/orginal_sequences_cropped/original'))

train_list_crop = []
train_list_original = []

f = open("train.json", 'r')
text = json.load(f) #загнали все из файла в переменную
for txt in text:
    train_list_crop.append(txt[0] + "_" + txt[1])
    train_list_original.append(txt[0])

test_list_crop = []
test_list_original = []

f = open("test.json", 'r')
text = json.load(f) #загнали все из файла в переменную
for txt in text:
    test_list_crop.append(txt[0] + "_" + txt[1])
    test_list_original.append(txt[0])

val_list_crop = []
val_list_original = []

f = open("val.json", 'r')
text = json.load(f) #загнали все из файла в переменную
for txt in text:
    val_list_crop.append(txt[0] + "_" + txt[1])
    val_list_original.append(txt[0])

test_list = []
train_list = []
val_list = []

for imagePath in img_list1:
    or_dict = dict()
    or_dict.update({'name': imagePath, 'label': 1})
    if imagePath[61:-9] in test_list_crop:
        test_list.append(or_dict)
    elif imagePath[61:-9] in train_list_crop:
        train_list.append(or_dict)
    elif imagePath[61:-9] in val_list_crop:
        val_list.append(or_dict)

for imagePath in img_list2:
    or_dict = dict()
    or_dict.update({'name': imagePath, 'label': 0})
    if imagePath[57:-9] in test_list_original:
        test_list.append(or_dict)
    elif imagePath[57:-9] in train_list_original:
        train_list.append(or_dict)
    elif imagePath[57:-9] in val_list_original:
        val_list.append(or_dict)

random.shuffle(val_list)
random.shuffle(test_list)
random.shuffle(train_list)

CreateFile("Validation.csv", val_list)
CreateFile("Test.csv", test_list)
CreateFile("Train.csv", train_list)
