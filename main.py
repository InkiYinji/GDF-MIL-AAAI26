import os
from models import process
from datasets.MIL import MIL
from utils.reader import read_yaml
from utils.writer import RecordWriter
from utils.basic import get_time


def main():

    if args.dataset in ["musk1", "musk2", "elephant", "tiger", "fox"] or args.dataset in ["musk1+", "musk2+", "elephant+", "tiger+", "fox+"]:
        data_cata = "benchmark"
    elif args.dataset in ["messifor", "ucsb_breast"] or args.dataset in ["messifor+", "ucsb_breast+"]:
        data_cata = "medical"
    elif args.dataset in ["web%s" % i for i in range(1, 10)] or args.dataset in ["web%s+" % i for i in range(1, 10)]:
        data_cata = "web"
    else:
        data_cata = "news"
    log_dir = os.path.join('records', r'%s/%s/%s' % (data_cata, args.dataset, model))
    args['log_dir'] = log_dir
    os.makedirs(log_dir, exist_ok=True)
    record_writer = RecordWriter(os.path.join(log_dir, experiments_type + "_seed" +
                                              str(args.general.seed) + "_" + get_time() + ".txt"))
    # record_writer = RecordWriter(os.path.join(log_dir, "seed" + str(args.general.seed) + ".txt"))
    record_writer_best = RecordWriter(os.path.join(log_dir, experiments_type + "_seed" +
                                                   str(args.general.seed) + "_best.txt"))
    record_writer.write2file("Dataset: %s" % args.dataset)
    record_writer.write2file("Model name: %s" % model)
    os.makedirs(log_dir, exist_ok=True)
    if args.general.model_name == "DTFDMIL":
        process.train_kcv_tdfd(args, mil, record_writer, record_writer_best)
    else:
        process.train_kcv(args, mil, record_writer, record_writer_best)


if __name__ == '__main__':
    """Configs"""
    model = "MILGNN"
    data_list = [
        "Drug/musk1.mat",
        "Drug/musk2.mat",
        "Image/elephant.mat",
        "Image/tiger.mat",
        "age/fox.mat",
        "Text/alt_atheism.mat",
        "Text/comp_graphics.mat",
        "Text/comp_os_ms-windows_misc.mat",
        "Text/comp_sys_ibm_pc_hardware.mat",
        "Text/comp_sys_mac_hardware.mat",
        "Text/comp_windows_x.mat",
        "Text/misc_forsale.mat",
        "Text/rec_autos.mat",
        "Text/rec_motorcycles.mat",
        "Text/rec_sport_baseball.mat",
        "Text/rec_sport_hockey.mat",
        "Text/sci_crypt.mat",
        "Text/sci_electronics.mat",
        "Text/sci_med.mat",
        "Text/sci_religion_christian.mat",
        "Text/sci_space.mat",
        "Text/talk_politics_guns.mat",
        "Text/talk_politics_mideast.mat",
        "Text/talk_politics_misc.mat",
        "Text/talk_religion_misc.mat",
        "Web/web1+.mat",
        "Web/web2+.mat",
        "Web/web3+.mat",
        "Web/web4+.mat",
        "Web/web5+.mat",
        "Web/web6+.mat",
        "Web/web7+.mat",
        "Web/web8+.mat",
        "Web/web9+.mat",
    ]
    data_type = 'mat'
    num_classes = 2
    k = 5
    experiments_type = "train_%dcv" % k  # train_test, "train_%dcv" % k

    # Load configs
    args = read_yaml('configs/%s.yaml' % model)

    for data_path in data_list:
        args['dataset'] = data_path.split('/')[-1].split('.')[0]
        args['k'] = k
        args['experiments_type'] = experiments_type
        mil = MIL(data_path)
        # args.model.topk = min(min(mil.bag_size), args.model.topk)

        args['in_dim'] = mil.d
        args['retrain'] = True
        args['data_path'] = data_path
        args.general.num_classes = num_classes

        main()

