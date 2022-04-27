import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import imp

resolution_map = {1.0: 'full', 0.5: 'half', 0.25: 'quarter'}


def resample(img, scale_factor=1.0, interpolation=Image.BILINEAR):
    width, height = img.size
    basewidth = width * scale_factor
    basewidth = int(basewidth)
    wpercent = (basewidth / float(width))
    hsize = int((float(height) * wpercent))
    return img.resize((basewidth, hsize), interpolation)


def recursive_mkdir(nested_dir_list):
    nested_dir = ''
    for dir in nested_dir_list:
        nested_dir = os.path.join(nested_dir, dir)
        if not os.path.isdir(nested_dir):
            os.mkdir(nested_dir)
    return


def preprocess(cf):
    for set in list(cf.settings.keys()):
        print('Processing {} set.'.format(set))

        # image dir
        image_dir = os.path.join(cf.raw_data_dir, 'leftImg8bit', set)
        city_names = os.listdir(image_dir)

        for city in city_names:
            print('Processing {}'.format(city))
            city_dir = os.path.join(image_dir, city)
            image_names = os.listdir(city_dir)
            image_specifiers = ['_'.join(img.split('_')[:3]) for img in image_names]

            for img_spec in tqdm(image_specifiers):
                for scale in cf.settings[set]['resolutions']:
                    recursive_mkdir([cf.out_dir, resolution_map[scale], set, city])

                    # image
                    img_path = os.path.join(city_dir, img_spec + '_leftImg8bit.png')
                    img = Image.open(img_path)
                    if scale != 1.0:
                        img = resample(img, scale_factor=scale, interpolation=Image.BILINEAR)
                    img_out_path = os.path.join(cf.out_dir, resolution_map[scale], set, city,
                                                img_spec + '_leftImg8bit.npy')
                    img_arr = np.array(img).astype(np.float32)

                    channel_axis = 0 if img_arr.shape[0] == 3 else 2
                    if cf.data_format == 'NCHW' and channel_axis != 0:
                        img_arr = np.transpose(img_arr, axes=[2, 0, 1])
                    np.save(img_out_path, img_arr)

                    # labels
                    for label_density in cf.settings[set]['label_densities']:
                        label_dir = os.path.join(cf.raw_data_dir, label_density, set, city)
                        for mod in cf.settings[set]['label_modalities']:
                            label_spec = img_spec + '_{}_{}'.format(label_density, mod)
                            label_path = os.path.join(label_dir, label_spec + '.png')
                            label = Image.open(label_path)
                            if scale != 1.0:
                                label = resample(label, scale_factor=scale, interpolation=Image.NEAREST)
                            label_out_path = os.path.join(cf.out_dir, resolution_map[scale], set, city,
                                                          label_spec + '.npy')
                            label_arr = np.array(label).astype(np.uint8)
                            if cf.data_format == 'NCHW' and channel_axis != 0:
                                label_arr = label_arr[np.newaxis, :, :]
                            np.save(label_out_path, label_arr)


if __name__ == "__main__":
    cf = imp.load_source('cf', 'preprocess_config.py')
    os.chdir('../')
    preprocess(cf)
