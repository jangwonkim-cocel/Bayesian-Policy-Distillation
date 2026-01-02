from BPD import BPDAgent
from utils import set_seed, get_learning_info, get_compression_ratio, load_buffer
import pickle
import inspect
import os
import argparse
import gdown
import time
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--env-name", default="Hopper-v3")      # OpenAI gym environment name
    parser.add_argument("--level", default="expert")            # expert or medium
    parser.add_argument("--random-seed", default=1, type=int)
    parser.add_argument("--eval-freq", default=5000, type=int)
    parser.add_argument("--max-teaching-count", default=1000000, type=int)
    parser.add_argument('--num-test-epi', default=10, type=int)
    parser.add_argument("--teacher-hidden-dims", default=(400, 300), type=tuple)
    parser.add_argument("--student-hidden-dims", default=(128, 128), type=tuple)

    parser.add_argument("--batch-size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)             # Discount factor
    parser.add_argument("--tau", default=0.005)                 # Target network update rate
    parser.add_argument("--noise-clip", default=0.5)            # Range to clip target policy noise
    parser.add_argument("--policy-freq", default=2, type=int)   # Frequency of delayed policy updates

    parser.add_argument("--h", default=0.5, type=float)
    parser.add_argument("--nu", default=4, type=float)
    parser.add_argument("--theta-threshold", default=0, type=float)
    parser.add_argument("--alpha-threshold", default=2, type=float)
    parser.add_argument("--init-kl-weight", default=0, type=float)
    parser.add_argument("--kl-max-coef", default=2, type=int)
    parser.add_argument("--datasize", default=1000000, type=int)

    args = parser.parse_args()

    # MuJoCo Environment Variable & Device Setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # STEP 1: Make Instances & Variables
    max_avg_return = 0
    seed = set_seed(args.random_seed)
    args.random_seed = seed
    learning_info = get_learning_info(args, seed)

    agent = BPDAgent(**learning_info)
    kl_weight = args.init_kl_weight

    # STEP 2: Load Dataset (=teacher buffer)
    buffer = load_buffer(args.env_name, args.level, args.datasize)

    # STEP 3: Training
    print(f"Distilling Start! | env_name: {args.env_name} | level: {args.level} | seed: {seed}")
    time_start = time.time()
    return_list = []
    for teaching_cnt in range(1, args.max_teaching_count + 1):
        kl_weight = (args.nu / args.max_teaching_count) * teaching_cnt
        kl_weight = min(kl_weight, args.kl_max_coef)
        agent.set_kl_weight(kl_weight)
        transitions = buffer.sample(batch_size=args.batch_size)
        agent.train(transitions)

        if teaching_cnt % args.eval_freq == 0:
            avg_student_return, max_student_return, min_student_return = agent.test()
            return_list.append(avg_student_return)
            print(f"[INFO] Teaching Count: [{teaching_cnt}/{args.max_teaching_count}]  |  Average Student Return:"
                  f" {avg_student_return:.3f}  |  Max Student Return: {max_student_return:.3f}  |  Min Student Return:"
                  f" {min_student_return:.3f}", end='')

            for i, c in enumerate(agent.actor.children()):
                temp = (torch.abs(c.get_pruned_weights()) == 0).float().data.cpu().numpy().mean()
                if hasattr(c, 'kl_reg'):
                    print(f"  |  sp_{i}: {1-temp:.3f}", end='')
                del temp
            print()

    return_sum = 0
    for i in range(10):
        return_sum += return_list[-1 - i]
    return_avg = return_sum / 10

    time_end = time.time()
    print(f"\nDistilling Finish!  |  Seed: {seed}  |  Consumed Time (sec): {time_end - time_start}")
    print("Average Return of the Last 10 Episode: {}".format(return_avg))
    cr = get_compression_ratio(learning_info["num_teacher_param"], agent)
    print('Compression ratio (kep_w/all_w)=', cr)
    print("-----------------------------------------------------------\n")
