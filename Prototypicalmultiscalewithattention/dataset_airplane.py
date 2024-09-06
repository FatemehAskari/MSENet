# import os
# import shutil

# # مسیر پوشه عکس‌ها
# image_folder = 'data/airplane/materials/images'

# # خواندن نام عکس‌ها در پوشه
# image_names = os.listdir(image_folder)

# for image_name in image_names:
    # جدا کردن نام فایل و پسوند
#     name, extension = os.path.splitext(image_name)
    
    # جدا کردن لیبل از نام فایل
#     label = name.split('_')[0]
    
    # مسیر پوشه جدید برای عکس
#     label_folder = os.path.join('data/airplane/materials/Imagenewclass', label)
#     os.makedirs(label_folder, exist_ok=True)
    
    # مسیر جدید عکس
#     new_image_path = os.path.join(label_folder, image_name)
    
    # انتقال عکس به پوشه جدید
#     old_image_path = os.path.join(image_folder, image_name)
#     shutil.move(old_image_path, new_image_path)



# import os

# # مسیر دایرکتوری
# directory = 'data/airplane/materials/Imagenewclass'

# # شمارنده فولدرها
# folder_count = 0

# # خواندن نام‌های فایل و پوشه‌ها
# items = os.listdir(directory)

# # شمارش تعداد فولدرها
# for item in items:
#     item_path = os.path.join(directory, item)
#     if os.path.isdir(item_path):
#         folder_count += 1

# print("تعداد فولدرها:", folder_count)



import os
import random
import csv

# مسیر پوشه عکس‌ها
image_folder = 'data/airplane/materials/Imagenewclass'

# تعداد فولدرها مربوط به هر دسته
train_folder_count = 30
validation_folder_count = 6
test_folder_count = 8

# لیست تمام فولدرها
all_folders = os.listdir(image_folder)

# اختصاص دسته به فولدرها
random.shuffle(all_folders)
train_folders = all_folders[:train_folder_count]
validation_folders = all_folders[train_folder_count:train_folder_count+validation_folder_count]
test_folders = all_folders[train_folder_count+validation_folder_count:]

# تابع برای خواندن لیست عکس‌ها در یک فولدر و ایجاد ردیف‌های CSV
def process_folder(folder, category, csv_writer):
    folder_path = os.path.join(image_folder, folder)
    image_names = os.listdir(folder_path)
    for image_name in image_names:
        image_path = os.path.basename(image_name)
        label = folder.split('_')[0]
        csv_writer.writerow([image_path, label])

# ایجاد فایل CSV برای دسته Train
train_csv_file_path = 'data/airplane/materials/train.csv'
with open(train_csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image_name', 'class_name'])
    
    # پردازش فولدرهای Train
    for folder in train_folders:
        process_folder(folder, 'train', csv_writer)

print("فایل train.csv با موفقیت ایجاد شد.")

# ایجاد فایل CSV برای دسته Validation
validation_csv_file_path = 'data/airplane/materials/val.csv'
with open(validation_csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image_name', 'class_name'])
    
    # پردازش فولدرهای Validation
    for folder in validation_folders:
        process_folder(folder, 'validation', csv_writer)

print("فایل validation.csv با موفقیت ایجاد شد.")

# ایجاد فایل CSV برای دسته Test
test_csv_file_path = 'data/airplane/materials/test.csv'
with open(test_csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image_name', 'class_name'])
    
    # پردازش فولدرهای Test
    for folder in test_folders:
        process_folder(folder, 'test', csv_writer)

print("فایل test.csv با موفقیت ایجاد شد.")