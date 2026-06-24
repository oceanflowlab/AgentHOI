import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset_file', default='hico', choices=['hico', 'swig'])
    parser.add_argument('--repeat_factor_sampling', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='apply repeat factor sampling to increase the rate at which tail categories are observed')
    parser.add_argument('--zero_shot_exp', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='[specific for hico], treat 120 rare interactions as zero shot')
    parser.add_argument('--ignore_non_interaction', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='[specific for hico], ignore <non_interaction> category')
    # parser.add_argument('--ignore_non_interaction', default=False, type=lambda x: (str(x).lower() == 'true'),
    #                     help='[specific for hico], ignore <non_interaction> category')
    parser.add_argument('--enable_softmax', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--test_score_thresh', default=0.1, type=float,
                        help="threshold to filter out HOI predictions")
    # parser.add_argument('--bbox_lambda', default=0, type=float)
    parser.add_argument('--bbox_lambda', default=0.3, type=float)
    parser.add_argument('--zero_shot_type', default="rare_first", type=str,
                        choices=["default", "uc0", "uc1", "uc2", "uc3", "uc4",
                                 "rare_first", "non_rare_first", "unseen_object", "unseen_verb"], )

    
    # * Log and Device
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    return parser