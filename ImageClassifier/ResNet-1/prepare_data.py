import os
import shutil


def main():
    # Step 1:创建训练集路径和测试集路径
    new_train_data_path = os.path.join(os.getcwd(), 'data/newtrain')
    new_test_data_path = os.path.join(os.getcwd(), 'data/newtest')

    if os.path.exists(new_train_data_path) is False:
        os.makedirs(new_train_data_path)

    if os.path.exists(new_test_data_path) is False:
        os.makedirs(new_test_data_path)

    # Step 2:将Cat和Dog类别中id>=10000的图片存到测试集路径中,其他图片存到训练集路径中
    origin_dataset_path = os.path.join(os.getcwd(), 'data/train')
    img_list = os.listdir(origin_dataset_path)
    for img_name in img_list:
        img_name_split = img_name.split('.')
        src_img = os.path.join(origin_dataset_path, img_name)

        if int(img_name_split[1]) >= 10000:
            shutil.copy(src_img, new_test_data_path)
        else:
            shutil.copy(src_img, new_train_data_path)


if __name__ == '__main__':
    main()