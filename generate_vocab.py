import os
from lib.vocab import *

def main(work_dir, vocab_file, attachment_counter_file, attachment_threshold, attachment_collapse_threshold):
    vocab = Vocab(attachment_counter_file, attachment_threshold, attachment_collapse_threshold, save_path=vocab_file)


if __name__ == '__main__':
    import warnings
    import argparse
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--work_dir", type = str, required=True)
    parser.add_argument('-v', "--vocab_file_name", type = str, default='')
    parser.add_argument('-c', '--attachment_counter_file_name', type = str, default='')
    parser.add_argument('-att_thresh', '--attachment_threshold', type = int, required=True)
    parser.add_argument('-att_c_thresh', '--attachment_collapse_threshold', type = int, required=True)

    args = parser.parse_args()

    work_dir = args.work_dir
    vocab_file = os.path.join(work_dir, args.vocab_file_name)
    attachment_counter_file = os.path.join(work_dir, 'preprocess', args.attachment_counter_file_name)
    attachment_threshold = args.attachment_threshold
    attachment_collapse_threshold = args.attachment_collapse_threshold
    
    main(
        work_dir=work_dir,
        vocab_file=vocab_file,
        attachment_counter_file=attachment_counter_file,
        attachment_threshold=attachment_threshold,
        attachment_collapse_threshold=attachment_collapse_threshold,
    )