

# YOUR_PATH
odir_image_path = "/data/OIA-ODIR/all_Cropped_Images/"
odir_train_anno_path = "/fundus_multi_label/data/ODIR/train.json"
odir_val_anno_path = "/fundus_multi_label/data/ODIR/val.json"
odir_test_anno_path = "/fundus_multi_label/data/ODIR/test.json"

rfmid_training_image_path = "/data/RFMiD/Training/"
rfmid_validation_image_path = "/data/RFMiD/Validation/"
rfmid_test_image_path = "/data/RFMiD/Test/"
rfmid_train_anno_path = "/fundus_multi_label/data/RFMiD/Training_annotations.json"
rfmid_val_anno_path = "/fundus_multi_label/data/RFMiD/Validation_annotations.json"
rfmid_test_anno_path = "/fundus_multi_label/data/RFMiD/Testing_annotations.json"

kaggle_image_path = "/data/kaggle_DR/Cropped_images/"
kaggle_train_anno_path = "/fundus_multi_label/data/KaggleDR+/train.json"
kaggle_val_anno_path = "/fundus_multi_label/data/KaggleDR+/val.json"
kaggle_test_anno_path = "/fundus_multi_label/data/KaggleDR+/test.json"

ODIR_CLASS = {
    '0': [0],
    '1': [1,2,3,4],
    '2': [5],
    '3': [6],
    '4': [7,8],
    '5': [9],
    '6': [10],
    '7': [11]
}

RFMiD_CLASS = {}

Kaggle_CLASS = {}

