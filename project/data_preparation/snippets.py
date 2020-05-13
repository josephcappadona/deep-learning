import os
import numpy as np
import random


def create_snippet(video_name, im, bbox_label, bbox_coords, is_neg=False, output_dir='output'):

    bbox_img = im.copy().crop(bbox_coords)

    snippets_dir = os.path.join(output_dir, video_name, 'snippets', bbox_label)
    os.makedirs(snippets_dir, exist_ok=True)

    snippet_id = str(random.randint(10**7, 10**10))
    if not is_neg:
        snippet_fn = bbox_label + '.' + snippet_id + '.png'
    else:
        snippet_fn = 'NEGATIVE' + '.' + snippet_id + '.png'
    snippet_fp = os.path.join(snippets_dir, snippet_fn)
    bbox_img.save(snippet_fp)
    return True

def is_blank(im):
    im = np.array(im)
    top_left_pixel = im[0, 0]
    is_not_blank = im[im != top_left_pixel].any()
    return (not is_not_blank)

def create_negative_snippets(video_name, im, im_fp, label, bboxes, output_dir='output', neg_to_pos_ratio=1, verbose=False):

    # mark off areas containing positive label bounding boxes
    occupied = np.zeros(im.size[::-1], dtype=np.uint8)
    for bbox_coords_list in bboxes.values():
        for bbox_coords in bbox_coords_list:
            x_min, y_min, x_max, y_max = bbox_coords
            occupied[y_min:y_max+1, x_min:x_max+1] = 1

    w_im, h_im = im.size
    count = 0
    for bbox_label, bbox_coords_list in bboxes.items():
        for bbox_coords in bbox_coords_list:
            
            # (try to) make `neg_to_pos_ratio` negative snippets for each positive snippet
            for k in range(neg_to_pos_ratio):
                
                x_min, y_min, x_max, y_max = bbox_coords
                w_bbox, h_bbox = (x_max - x_min), (y_max - y_min)

                N = 10
                k = 0.9
                # try at most N times to find a viable negative image somewhere random in the image
                for i in range(N):
                    w_rand_max = int(w_im * (k ** (i + 1)))
                    w_rand_min = int(0.75 * w_rand_max)
                    h_rand_max = int(h_im * (k ** (i + 1)))
                    h_rand_min = int(0.75 * h_rand_max)
                    w_rand = random.randint(w_rand_min, w_rand_max)
                    h_rand = random.randint(h_rand_min, h_rand_max)
                    x_rand = random.randint(0, w_im - w_rand)
                    y_rand = random.randint(0, h_im - h_rand)
                    x_rand_min, x_rand_max = x_rand, min(w_im - 1, x_rand + w_rand)
                    y_rand_min, y_rand_max = y_rand, min(h_im - 1, y_rand + h_rand)

                    # if random bbox overlaps with any label bbox
                    # or subimage is blank
                    # then try to find new random bbox
                    does_overlap_label_bbox = occupied[y_rand_min:y_rand_max,
                                                       x_rand_min:x_rand_max].any()
                    subimage = im.crop((x_rand_min, y_rand_min, x_rand_max, y_rand_max))
                    if does_overlap_label_bbox or is_blank(subimage):
                        
                        if i == N-1 and verbose:
                            print('\tCould not find negative box for label \'%s\'' + 
                                  ' in image \'%s\'.\n' % (bbox_label, im_fp))
                        continue

                    # found good box!
                    neg_bbox_coords = (x_rand_min, y_rand_min, x_rand_max, y_rand_max)
                    create_snippet(video_name, im, label, neg_bbox_coords, is_neg=True)
                    count += 1
                    break
    return count

