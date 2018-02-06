import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import glob


if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

'''
Example
python find_phone.py --images_path=test_images
'''

flags=tf.app.flags
flags.DEFINE_string('images_path', '','Give path for folder. eg: python find_phone.py --images_path=test_images')
FLAGS=flags.FLAGS

assert FLAGS.images_path, '`images_path` is missing.'
if(FLAGS.images_path != 'test_images'):
  sys.exit("Incorrect Test Images Path")

# This is needed to display the images.
#get_ipython().magic('matplotlib inline')
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util
from utils import visualization_utils as vis_util

#"SSD with Mobilenet" model here. 
MODEL_NAME = 'find_phone_model'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
NUM_CLASSES = 1

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names.

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection
# If you want to test the code with your images, just add images to the test_images folder.
if FLAGS.images_path:
  PATH_TO_TEST_IMAGES_DIR = FLAGS.images_path

#PATH_TO_TEST_IMAGES_DIR = 'test_images'  
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 7) ]
TEST_IMAGE_PATHS=[i for i in glob.glob(PATH_TO_TEST_IMAGES_DIR+'/*.jpg')]
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0') 

    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      image_path=image_path.split('\\')
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      mid_y=(boxes[0][0][0]+boxes[0][0][2])/2
      mid_x=(boxes[0][0][1]+boxes[0][0][3])/2
      print('{:.3f},{:.3f}'.format(mid_x,mid_y))
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)

      #Converting array to image and displaying
      dis_img = Image.fromarray(image_np, 'RGB')
      dis_img.show()
      dis_img.save('Results/image_'+image_path[1])
