import os
import json
from datetime import datetime
from time import sleep

def init_json(args, save_path, save_name, model_name, algorithm = False, dataset = 'wikitext2', metric = 'ppl'):
    data = {
    'file' : f'created by {os.path.abspath(__file__)}',
    'start_time' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'end_time' : None,
    'args' : vars(args),
    'archive' : []
    }

    save_path = os.path.abspath(save_path)

    os.makedirs(f'{save_path}/{save_name}', exist_ok=True)
    json_name = f'{"algorithm" if algorithm else "replace"}_{model_name}_{dataset}_{metric}.json'

    if not os.path.exists(f'{save_path}/{save_name}/{json_name}'):
        with open(f'{save_path}/{save_name}/{json_name}', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    result_path = f'{save_path}/{save_name}/{json_name}'

    return result_path


def write_json(result_path, result):
    assert os.path.exists(result_path), f'{result_path} does not exist'
    assert '.json' in result_path, f'{result_path} is not json file'
    
    for _ in range(5):
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
                data['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                data['archive'].append(result)
            with open(result_path, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            break
        except Exception as e:
            print(e)
            sleep(5)