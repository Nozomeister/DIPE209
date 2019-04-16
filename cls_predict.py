import numpy as np
import glob
import cv2
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
    
def numbers_to_strings(argument):
    switcher = {
        0: "Melanoma",
        1: "Melanocytic nevus",
        2: "Basal cell carcinoma",
        3: "Actinic keratosis / Bowen's disease (intraepithelial carcinoma)",
        4: "Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)",
        5: "Dermatofibroma",
        6: "Vascular lesion",
    }
    return switcher.get(argument, "nothing")

if __name__ == '__main__':

    from keras import Model
    from models import backbone
    from paths import submission_dir, mkdir_if_not_exist
    from datasets.ISIC2018 import load_validation_data, load_test_data
    from misc_utils.prediction_utils import cyclic_stacking

    def task3_tta_predict(model, img_arr):
        img_arr_tta = cyclic_stacking(img_arr)
        pred_logits = np.zeros(shape=(img_arr.shape[0], 7))

        for _img_crops in img_arr_tta:
            pred_logits += model.predict(_img_crops)

        pred_logits = pred_logits/len(img_arr_tta)

        return pred_logits

    backbone_name = 'inception_v3'
    version = '0'
    use_tta = False

    pred_set = 'test'  # or test
    load_func = load_validation_data if pred_set == 'validation' else load_test_data
    images, image_names = load_func(task_idx=3, output_size=224)

    # max_num_images = 10
    max_num_images = images.shape[0]
    images = images[:max_num_images]
    image_names = image_names[:max_num_images]

    num_folds = 1

    print('Starting prediction for set %s with TTA set to %r with num_folds %d' % (pred_set, use_tta, num_folds))


    y_pred = np.zeros(shape=(max_num_images, 7))

    for k_fold in range(num_folds):
        print('Processing fold ', k_fold)
        run_name = 'task3_' + backbone_name + '_k' + str(k_fold) + '_v' + version
        model = backbone(backbone_name).classification_model(load_from=run_name)
        predictions_model = Model(inputs=model.input, outputs=model.get_layer('predictions').output)
        if use_tta:
            y_pred += task3_tta_predict(model=predictions_model, img_arr=images)
        else:
            y_pred += predictions_model.predict(images)

    y_pred = y_pred / num_folds
    y_prob = softmax(y_pred)

    print('Done predicting -- creating submission')

    #submission_file = submission_dir + '/task3_' + pred_set + '_submission.csv'
    submission_file = submission_dir + '/task3_' + pred_set + '_submission.json' #writing to txt file
    f = open(submission_file, 'w')
    #f.write('image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n') //for csv only
    for i_image, i_name in enumerate(image_names):
        i_line = i_name
        max_prob = 0
        cls_prob = 1
        for i_cls in range(7):
            prob = y_prob[i_image, i_cls]
            if prob > max_prob:
               max_prob = prob
               cls_prob = i_cls
        
        i_line += ',' + numbers_to_strings(cls_prob)+'\n'
        


        f.write(i_line)  # Give your csv text here.
    #for i_image, i_name in enumerate(image_names):
    #    i_line = i_name
    #    for i_cls in range(7):
    #        prob = y_prob[i_image, i_cls]
    #        if prob < 0.001:
    #            prob = 0.
    #        i_line += ',' + str(prob)

    #    i_line += '\n'
    #   f.write(i_line)  # Give your csv text here.

    f.close()
    