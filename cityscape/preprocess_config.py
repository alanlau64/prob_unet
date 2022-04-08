raw_data_dir = 'SET_INPUT_DIRECTORY_ABSOLUTE_PATH_HERE'
out_dir = 'SET_OUTPUT_DIRECTORY_ABSOLUTE_PATH_HERE'

# settings:
settings = {
    'train': {'resolutions': [1.0, 0.5, 0.25], 'label_densities': ['gtFine'],
              'label_modalities': ['labelIds']},
    'val': {'resolutions': [1.0, 0.5, 0.25], 'label_densities': ['gtFine'],
            'label_modalities': ['labelIds']},
}

data_format = 'NCHW'
