import torch
import argparse
import multiprocessing as mp
import os
import time
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


class Runner(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="cascade_rcnn.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--img_list",
        type=str,
        default='',
    )
    parser.add_argument(
        "--output",
        help="outputfolder",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--save_folder', default='./testout/', type=str, help='Dir to save txt results')

    return parser


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    runner = Runner(cfg)
    if args.img_list != '':
        f_list = open(args.img_list, 'r')
        test_dataset = f_list.readlines()
        test_dataset = [_d.strip() for _d in test_dataset]
    print(f'done preparing dataset!, N={len(test_dataset)}', flush=True)

    for image_path in tqdm.tqdm(test_dataset):
        # use PIL, to be consistent with evaluation
        img_name = image_path.split('/')[-1]
        subfolder = image_path.split('/')[-2]
        save_name = os.path.join(args.save_folder, subfolder, img_name.split('.')[0] + ".txt")

        img = read_image(image_path, format="BGR")
        start_time = time.time()
        predictions = runner.predictor(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                img_name,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
                )
        )
        classes, scores = predictions["instances"].pred_classes.cpu(), predictions["instances"].scores.cpu()
        # --------------------------------------------------------------------
        name_ifperson = os.path.join(args.save_folder, 'person_index', subfolder,  img_name)
        name_ifnoperson = os.path.join(args.save_folder, 'noperson_index', subfolder,  img_name )

        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            has_person = 1 if any([int(c) == 0 for c in classes]) else 0
            fd.write(str(has_person) + " \n")
            for _c, _s in zip(classes, scores):
                line = str(_c.item()) + ":" + str(_s.item()) + " \n"
                fd.write(line)
        if has_person:
            if not os.path.isdir(os.path.dirname(name_ifperson)):
                 os.makedirs(os.path.dirname(name_ifperson))
            touch(name_ifperson)
        else:
            if not os.path.isdir(os.path.dirname(name_ifnoperson)):
                os.makedirs(os.path.dirname(name_ifnoperson))
            touch(name_ifnoperson)