"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from generators.common import Generator
import os
import numpy as np
# from pycocotools.coco import COCO
import cv2
import json


CATEGORIES = [
      "header",
      "question-answer",
      "question",
      "question_label",
      "A",
      "A_label",
      "B",
      "B_label",
      "C",
      "C_label",
      "D",
      "D_label"
  ]

CATEGORY_TO_ID = dict([(cat, id) for id, cat in enumerate(CATEGORIES)] )

def anno_from_json(json_file, data_dir):
    with open(json_file) as f:
        anno = json.load(f)
    for obj in anno:
        obj["file_name"] = os.path.join(data_dir, obj["file_name"])
    return anno

class CocoGenerator(Generator):
    """
    Generate data from the COCO dataset.
    See https://github.com/cocodataset/cocoapi/tree/master/PythonAPI for more information.
    """

    def __init__(self, data_dir, set_name, **kwargs):
        """
        Initialize a COCO data generator.

        Args
            data_dir: Path to where the COCO dataset is stored.
            set_name: Name of the set to parse.
        """
        self.data_dir = data_dir #"/content/drive/MyDrive/Colab Notebooks/idris-teacher-assistant"
        self.set_name = set_name #[train, val, test]

        self.load_classes()
        self.anno = anno_from_json(f"{data_dir}/anno/{set_name}_anno.json", f"{data_dir}/detection_data_with_qa_labels")
        self.image_ids = list(range(len(self.anno)))

        super(CocoGenerator, self).__init__(**kwargs)



    def load_classes(self):
        """
        Loads the class to label mapping (and inverse) for COCO.
        """

        self.classes = CATEGORY_TO_ID # name to label
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def size(self):
        """ Size of the COCO dataset.
        """
        return len(self.image_ids)

    def num_classes(self):
        """ Number of classes in the dataset. For COCO this is 80.
        """
        return len(CATEGORIES)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def coco_label_to_label(self, coco_label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return coco_label

    def coco_label_to_name(self, coco_label):
        """ Map COCO label to name.
        """
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        """ Map label as used by the network to labels as used by COCO.
        """
        return label

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        image = self.anno[image_index]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        Assuming image_index is a number in range(len(self.image_ids))
        """
        path = self.anno[image_index]['file_name']
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        anno = self.anno[image_index]["annotations"]
        labels = np.array([a["category_id"] for a in anno])
        bboxes = np.array([np.array(a["bbox"]) for a in anno])
        annotations = {'labels': labels, 'bboxes': bboxes}

        return annotations
