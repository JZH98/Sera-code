from cross_validation import *
from prepare_data_DREAMER import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--dataset', type=str, default='DREAMER')
    parser.add_argument('--data-path', type=str, default='DREAMER/')
    parser.add_argument('--subjects', type=int, default=23)
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='V', choices=['A', 'V', 'D', 'L'])###1
    parser.add_argument('--segment', type=int, default=4)
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--scale-coefficient', type=float, default=1)
    parser.add_argument('--input-shape', type=tuple, default=(1, 14, 512))
    parser.add_argument('--data-format', type=str, default='eeg')
    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=2025)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--patient', type=int, default=200)
    parser.add_argument('--patient-cmb', type=int, default=8)
    parser.add_argument('--max-epoch-cmb', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--LS', type=bool, default=False, help="Label smoothing")
    parser.add_argument('--LS-rate', type=float, default=0.1)

    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--load-path-final', default='./save/final_model.pth')
    parser.add_argument('--gpu', default=0)# 0,1
    parser.add_argument('--save-model', type=bool, default=True)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='SERA')
    parser.add_argument('--pool', type=int, default=2)
    parser.add_argument('--pool-step-rate', type=float, default=0.25)
    parser.add_argument('--T', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--patch-stride', type=int, default=8)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--num-head', type=int, default=32)
    parser.add_argument('--graph-type', type=str, default='gen', choices=['fro', 'gen', 'hem', 'BL'])###1
    parser.add_argument('--hidden', type=int, default=32)

    ######## Reproduce the result using the saved model ######
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    sub_to_run = np.arange(args.subjects)
    pd = PrepareData(args)
    pd.run(sub_to_run, split=True, expand=True)
    cv = CrossValidation(args)
    seed_all(args.random_seed)
    cv.leave_sub_out(subject=sub_to_run)
