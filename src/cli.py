import argparse
from src.train.loop import train_main
from src.evaluation.eval_single import eval_main

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", default="configs/base.yaml")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--exp_dir", required=True)

    args = parser.parse_args()
    if args.cmd == "train":
        train_main(args.config)   # 你在 loop.py 里实现 train_main(config_path)
    elif args.cmd == "eval":
        eval_main(args.exp_dir)   # 你在 eval_single.py 里实现 eval_main(exp_dir)

if __name__ == "__main__":
    main()