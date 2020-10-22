import argparse
from datetime import datetime
import yaml
import backdoor_tasks
from utils.helper import Helper
from utils.utils import *



def run(helper: Helper):

    for epoch in helper.epochs:
        train()
        test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit', required=True)

    args = parser.parse_args()
    timestamp = datetime.now().strftime('%b.%d_%H.%M.%S')
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['commit'] = args.commit
    params['name'] = args.name
    params['timestamp'] = timestamp

    helper = Helper(params, timestamp)

    if helper.log:
        logger = create_logger()
        fh = logging.FileHandler(filename=f'{helper.folder_path}/log.txt')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.warning(f'Logging things. current path: {helper.folder_path}')
        logger.error(
            f'LINK: <a href="https://github.com/ebagdasa/backdoors/tree/{helper.commit}">https://github.com/ebagdasa/backdoors/tree/{helper.commit}</a>')

        helper.params['tb_name'] = args.name
        with open(f'{helper.folder_path}/params.yaml.txt', 'w') as f:
            yaml.dump(helper.params, f)
    else:
        logger = create_logger()

    run(helper)
