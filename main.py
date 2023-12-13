import argparse

from recbole_pjf.quick_start import run_recbole_pjf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='KAERR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='zhaopin_kg', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--device', type=str, default=None, help='gpu id')
    parser.add_argument('--max_path_num', type=int, default=16, help='max number of metapath')


    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    if args.device:
        config_dict = {'gpu_id': args.device, 'max_path_num': args.max_path_num}
        run_recbole_pjf(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=config_dict)
    else:
        run_recbole_pjf(model=args.model, dataset=args.dataset, config_file_list=config_file_list)