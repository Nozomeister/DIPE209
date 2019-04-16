if __name__ == '__main__':
    import numpy as np
    import glob
    import cv2
    import skimage
    from keras import Model
    from models import backbone
    from paths import submission_dir, mkdir_if_not_exist
    from datasets.ISIC2018 import load_validation_data, load_test_data
    from misc_utils.prediction_utils import cyclic_stacking
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def numbers_to_strings(argument):
        switcher = {
            0: "Melanoma",
            1: "Melanocytic nevus",
            2: "Basal cell carcinoma",
            3: "Bowen's disease",
            4: "Benign keratosis",
            5: "Dermatofibroma",
            6: "Vascular lesion",
        }
        return switcher.get(argument, "nothing")
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
    
    for i_image, i_name in enumerate(image_names):
        i_line = i_name
        print(i_name)
        max_prob = 0
        cls_prob = 1
        for i_cls in range(7):
            prob = y_prob[i_image, i_cls]
            if prob > max_prob:
               max_prob = prob
               cls_prob = i_cls
        original_image = cv2.imread(submission_dir + '/Input/' + i_name + '.jpg') #load original
        #true_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) #convert original image
        mask_image = cv2.imread(submission_dir + '/Output/' + i_name + '.png') #load mask
        grey_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY) #convert mask to grey channel
        (thresh, black_image) = cv2.threshold(grey_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       	color_image = np.zeros(original_image.shape, original_image.dtype)  #set all value to 0
        if (cls_prob==0):
            color_image[:,:] = (0, 252, 124) #melanoma
            cv2.putText(original_image, numbers_to_strings(cls_prob), (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 252, 124), 1,  lineType=cv2.LINE_AA)      
        elif (cls_prob==1):
            color_image[:,:] = (139, 139, 0) #nevus
            cv2.putText(original_image, numbers_to_strings(cls_prob), (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (139, 139, 0), 1,  lineType=cv2.LINE_AA)
        elif (cls_prob==2):
            color_image[:,:] = (0, 0, 255) #BCC
            cv2.putText(original_image, numbers_to_strings(cls_prob), (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 1,  lineType=cv2.LINE_AA)
        elif (cls_prob==3):
            color_image[:,:] = (0, 69, 255) #Bowen's Disaease/AIKEC
            cv2.putText(original_image, numbers_to_strings(cls_prob), (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 69, 255), 1,  lineType=cv2.LINE_AA)
        elif (cls_prob==4):
            color_image[:,:] = (204, 50, 153) #Benign Keratoses
            cv2.putText(original_image, numbers_to_strings(cls_prob), (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (204, 50, 153), 1,  lineType=cv2.LINE_AA)
        elif (cls_prob==5):
            color_image[:,:] = (79, 79, 47) #Dermatofibroma
            cv2.putText(original_image, numbers_to_strings(cls_prob), (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (79, 79, 47), 1,  lineType=cv2.LINE_AA)
        elif (cls_prob==6):
            color_image[:,:] = (255, 144, 30) #Vascular
            cv2.putText(original_image, numbers_to_strings(cls_prob), (10, 35), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 144, 30), 1,  lineType=cv2.LINE_AA)
        else: color_image[:,:] = (140, 230, 240) #no disease
        color_mask = cv2.bitwise_and(color_image, color_image, mask=black_image)
        output_image = cv2.addWeighted(color_mask, 0.35, original_image, 1, 0 ,original_image)
        #cv2.putText(original_image, numbers_to_strings(cls_prob), (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1,  lineType=cv2.LINE_AA)
        cv2.imwrite(submission_dir + '/Output/' + i_name + '.png', original_image)
        print(cls_prob)
    

    