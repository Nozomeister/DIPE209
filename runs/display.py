

if __name__ == '__main__':
  import numpy as np
  import matplotlib.pyplot as plt
  from PIL import Image
  from skimage.transform import resize as sk_resize
  import cv2
  from paths import submission_dir, mkdir_if_not_exist
  from models import backbone
  from runs.seg_eval import task1_post_process
  from datasets.ISIC2018 import load_validation_data, load_test_data
  from misc_utils.prediction_utils import inv_sigmoid, sigmoid, cyclic_pooling, cyclic_stacking
  from misc_utils.visualization_utils import view_by_batch
  import glob
  
  for filename in glob.glob(submission_dir+'/Input/*.jpg'): 
    img = cv2.imread(filename, 1) 
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert from BGR to RGB
    print(img)
    output = filename.replace('.jpg', '.png')
    pOutput = output.replace('Input', 'Output')
    print(pOutput)
    mask = cv2.imread(pOutput) #read mask from file.
    print(mask)
    plt.imshow(img, cmap="gray")    # Also set the cmap to gray
    plt.imshow(mask, alpha=0.4)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(img,img, mask = mask_inv) #blackout area in img
    mask_fg = cv2.bitwise_and(mask, mask, mask = mask)
    dst = cv2.add(img_bg, mask_fg)
    img_output = dst
    cv2.imshow(img_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # get first masked value (foreground)
    #fg = cv2.bitwise_or(img, img, mask=mask)
    # get second masked value (background) mask must be inverted
    #mask = cv2.bitwise_not(mask)
    #background = np.full(img.shape, 255, dtype=np.uint8)
    #bk = cv2.bitwise_or(background, background, mask=mask)
    # combine foreground+background
    #final = cv2.bitwise_or(fg, bk)
    #print(final)
    