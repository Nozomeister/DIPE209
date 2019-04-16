
if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    from skimage.transform import resize as sk_resize
    from keras import Model
    from paths import submission_dir, mkdir_if_not_exist
    from models import backbone
    from runs.seg_eval import task1_post_process
    from datasets.ISIC2018 import load_validation_data, load_test_data
    from misc_utils.prediction_utils import inv_sigmoid, sigmoid, cyclic_pooling, cyclic_stacking
    from misc_utils.visualization_utils import view_by_batch
    
    def softmax(x):
    	  e_x = np.exp(x - np.max(x))
    	  return e_x / np.sum(e_x, axis=1, keepdims=True)
    def numbers_to_strings(argument):
 	      switcher ={
         	  0: "Melanoma",
        	  1: "Melanocytic nevus",
        	  2: "Basal cell carcinoma",
        	  3: "Actinic keratosis / Bowen's disease (intraepithelial carcinoma)",
        	  4: "Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)",
        	  5: "Dermatofibroma",
        	  6: "Vascular lesion",
    	  }
        return switcher.get(argument, "nothing")
         
    def task1_tta_predict(model, img_arr): #Task 1
        img_arr_tta = cyclic_stacking(img_arr)
        mask_arr_tta = []
        for _img_crops in img_arr_tta:
            _mask_crops = model.predict(_img_crops)
            mask_arr_tta.append(_mask_crops)

        mask_crops_pred = cyclic_pooling(*mask_arr_tta)

        return mask_crops_pred
        
    def task3_tta_predict(model, img_arr):  #Task 3 
        img_arr_tta = cyclic_stacking(img_arr)
        pred_logits = np.zeros(shape=(img_arr.shape[0], 7))

        for _img_crops in img_arr_tta:
            pred_logits += model.predict(_img_crops)

        pred_logits = pred_logits/len(img_arr_tta)

        return pred_logit

    task1_backbone_name = 'vgg19' #adding task
    task1_version = '0'
    task_idx = 1
    use_tta = False

    pred_set = 'test'  # or test
    load_func = load_validation_data if pred_set == 'validation' else load_test_data
    images, image_names, image_sizes = load_func(task_idx=1, output_size=224)

    # max_num_images = 10
    max_num_images = images.shape[0]
    images = images[:max_num_images]
    image_names = image_names[:max_num_images]
    image_sizes = image_sizes[:max_num_images]

    y_pred = np.zeros(shape=(max_num_images, 224, 224))

    num_folds = 1

    print('Starting prediction for set %s with TTA set to %r with num_folds %d' % (pred_set, use_tta, num_folds))

    for k_fold in range(num_folds):
        print('Processing fold ', k_fold)
        model_name = 'task%d_%s' % (task_idx, task1_backbone_name) #adding task 1
        run_name = 'task%d_%s_k%d_v%s' % (task_idx, task1_backbone_name, k_fold, task1_version)
        model = backbone(backbone_name).segmentation_model(load_from=run_name)
        if use_tta:
            y_pred += inv_sigmoid(task1_tta_predict(model=model, img_arr=images))[:, :, :, 0]
        else:
            y_pred += inv_sigmoid(model.predict(images))[:, :, :, 0]

    print('Done predicting task 1-- now doing post-processing')

    y_pred = y_pred / num_folds
    y_pred = sigmoid(y_pred)

    y_pred = task1_post_process(y_prediction=y_pred, threshold=0.5, gauss_sigma=2.)
    output_dir = submission_dir + '/Output' #change name to desired folder
    mkdir_if_not_exist([output_dir])

    for i_image, i_name in enumerate(image_names):

        current_pred = y_pred[i_image]
        current_pred = current_pred * 255
        
        resized_pred = sk_resize(current_pred,
                                 output_shape=image_sizes[i_image],
                                 preserve_range=True,
                                 mode='reflect',
                                 anti_aliasing=True)

        resized_pred[resized_pred > 128] = 255
        resized_pred[resized_pred <= 128] = 0

        im = Image.fromarray(resized_pred.astype(np.uint8))
        im.save(output_dir + '/' + i_name + '.png')
    
    task3_backbone_name = 'inception_v3'
    version = '0'
    use_tta = False

    #pred_set = 'validation'  # or test
    #load_func = load_validation_data if pred_set == 'validation' else load_test_data
    images, image_names = load_func(task_idx=3, output_size=224)

    # max_num_images = 10
    #max_num_images = images.shape[0]
    #mages = images[:max_num_images]
    #image_names = image_names[:max_num_images]

    #num_folds = 1

    print('Starting prediction for set %s with TTA set to %r with num_folds %d' % (pred_set, use_tta, num_folds))


    task3_y_pred = np.zeros(shape=(max_num_images, 7))

    for k_fold in range(num_folds):
        print('Processing fold ', k_fold)
        run_name = 'task3_' + task3_backbone_name + '_k' + str(k_fold) + '_v' + version
        model = backbone(task3_backbone_name).classification_model(load_from=run_name)
        predictions_model = Model(inputs=model.input, outputs=model.get_layer('predictions').output)
        if use_tta:
           task3_y_pred += task3_tta_predict(model=predictions_model, img_arr=images)
        else:
            task3_y_pred += predictions_model.predict(images)

    task3_y_pred = task3_y_pred / num_folds
    y_prob = softmax(task3_y_pred)

    print('Done predicting task 3 -- creating submission')

    #submission_file = submission_dir + '/task3_' + pred_set + '_submission.csv'
    submission_file = submission_dir + '/task3_' + pred_set + '_submission.txt' #writing to txt file
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

