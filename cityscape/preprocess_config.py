raw_data_dir = 'D:\demo\\523\\final_pro\data\City'
out_dir = 'D:\demo\\523\\final_pro\data\City\\afterPre'

# settings:
settings = {
    'train': {'resolutions': [1.0, 0.5, 0.25], 'label_densities': ['gtFine'],
              'label_modalities': ['labelIds']},
    'val': {'resolutions': [1.0, 0.5, 0.25], 'label_densities': ['gtFine'],
            'label_modalities': ['labelIds']},
}

data_format = 'NCHW'
