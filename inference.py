import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms

from utils.flow_utils import flow_to_image, resize_flow
from utils.torch_utils import restore_model
from models.pwclite import PWCLite
import flowpy
import matplotlib.pyplot as plt
import cv2
from utils.warp_utils import flow_warp
from PIL import Image
from math import log10, sqrt 
import PIL
from skimage import measure


class TestHelper():
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/KITTI15/pwclite_ar.tar')
    parser.add_argument('-s', '--test_shape', default=[384, 640], type=int, nargs=2)
    parser.add_argument('-i', '--img_list', nargs='+',
                        default=['examples/img1.png', 'examples/img2.png'])
    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': len(args.img_list),
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }

    
    t3=args.img_list[0].split('_')
    t0=t3[2].split('.')
    t=args.img_list[1].split('_')
    t1=t[2].split('.')
       
    ts = TestHelper(cfg)

    imgs = [imageio.imread(img).astype(np.float32) for img in args.img_list]
    h, w = imgs[0].shape[:2]

    flow_12 = ts.run(imgs)['flows_fw'][0]

    flow_12 = resize_flow(flow_12, (h, w))
    np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])

    vis_flow = flowpy.flow_to_rgb(np_flow_12)

    cv2.imwrite(t0[0]+ " " +t1[0]+".png", vis_flow)

 
    #=flow_warp(im2, flow12, pad='border', mode='bilinear'):
    first_image = np.asarray(Image.open(args.img_list[0]))
    cv2.imwrite(t0[0]+".png",first_image)
    second_image = np.asarray(Image.open(args.img_list[1]))

    np_flow_12[np.isnan(np_flow_12)] = 0
    warped_first_image = flowpy.backward_warp(second_image, np_flow_12)
    
    cv2.imwrite("warped_oc"+t1[0]+".png", warped_first_image)
    def PSNR(original, compressed): 
      mse = np.mean((original - compressed) ** 2) 
      if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                    # Therefore PSNR have no importance. 
          return 100
      max_pixel = 255.0
      psnr = 20 * log10(max_pixel / sqrt(mse)) 
      return psnr 
  # 5. Compute the Structural Similarity Index (SSIM) between the two
  #    images, ensuring that the difference image is returned


    print(PSNR(warped_first_image,first_image))
#print( tf.image.ssim(second_image, warped_first_image, max_val=255, filter_size=11,
                       #   filter_sigma=1.5, k1=0.01, k2=0.03))


    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error

    import argparse
    import imutils
    import cv2
# construct the argument parse and parse the arguments
    grayA = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(warped_first_image, cv2.COLOR_BGR2GRAY)

    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    thresh = cv2.threshold(diff, 0, 255,
	  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	  cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
      (x, y, w, h) = cv2.boundingRect(c)
      cv2.rectangle(first_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
      cv2.rectangle(warped_first_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images
    from google.colab.patches import cv2_imshow
    cv2_imshow( first_image)
    cv2_imshow(warped_first_image)
    cv2_imshow(diff)
    cv2.imwrite("diff_"+t1[0]+".png", diff)

    cv2_imshow( thresh)
    cv2.imwrite("thresh_"+t1[0]+".png", thresh)

    cv2.waitKey(0)


