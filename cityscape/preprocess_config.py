import os

raw_data_dir = './TestData/City'
out_dir = './TestData/postprocess/city'

# settings:
settings = {
    'train': {'resolutions': [1.0, 0.5, 0.25], 'label_densities': ['gtFine'],
              'label_modalities': ['labelIds']},
    'val': {'resolutions': [1.0, 0.5, 0.25], 'label_densities': ['gtFine'],
            'label_modalities': ['labelIds']},
}

data_format = 'NCHW'
