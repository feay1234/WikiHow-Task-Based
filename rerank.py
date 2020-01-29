import argparse
import train
import Data


def main_cli():
    parser = argparse.ArgumentParser('CEDR model re-ranking')
    parser.add_argument('--model', choices=train.MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--run', type=argparse.FileType('rt'))
    parser.add_argument('--model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--out_path', type=argparse.FileType('wt'))
    args = parser.parse_args()
    model = train.MODEL_MAP[args.model]().cuda()
    dataset = Data.read_datafiles(args.datafiles)
    run = Data.read_run_dict(args.run)
    if args.model_weights is not None:
        model.load(args.model_weights.name)
    train.run_model(model, dataset, run, args.out_path.name, desc='rerank')


if __name__ == '__main__':
    main_cli()
