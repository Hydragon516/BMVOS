import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from dataset_loaders import *
import evaluation
import models
import utils
from local_config import config
np.seterr(divide='ignore', invalid='ignore')


torch.cuda.set_device(0)


def test_model(model, save):
    datasets = {
        'DAVIS16_val': DAVIS_Test(config['davis_path'], '2016', 'val'),
        'DAVIS17_val': DAVIS_Test(config['davis_path'], '2017', 'val'),
        'DAVIS17_test-dev': DAVIS_Test(config['davis_path'], '2017', 'test-dev'),
        # 'YTVOS_val': YTVOS_Test(config['ytvos_path'])
    }

    for key, dataset in datasets.items():
        evaluator = evaluation.VOSEvaluator(dataset, save)
        result_fpath = os.path.join(config['output_path'], os.path.splitext(os.path.basename(__file__))[0])
        evaluator.evaluate(model, os.path.join(result_fpath, key))


def main():
    model = models.BMVOS()
    print('Network model {} loaded, (size: {})'.format(model.__class__.__name__, utils.get_model_size_str(model)))
    torch.backends.cudnn.deterministic = False
    model.output_scores = False
    model.output_segs = True
    model.load_state_dict(torch.load('../trained_model/BMVOS_davis.pth', map_location='cuda:0'))
    test_model(model, save=True)


if __name__ == '__main__':
    main()
