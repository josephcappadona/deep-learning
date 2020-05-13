import json
from PIL import Image
import os
import utils


def load_label_template(label_template_fp):
    return utils.read_json(label_template_fp)

def adapt_label_template(template_json, img_filepath):

    img_json = template_json.copy()
    img_json['imagePath'] = utils.get_filename(img_filepath)
    img_json['imageData'] = None
    im = Image.open(img_filepath)
    img_width, img_height = im.size
    img_json['imageWidth'] = img_width
    img_json['imageHeight'] = img_height

    # write new label .json file
    new_label_fp = utils.remove_extension(img_filepath) + '.json'
    utils.write_json(new_label_fp, img_json)

