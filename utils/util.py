import json
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import torch

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()

def load_model(model,n_gpu,model_ckpt):
    checkpoint = torch.load(model_ckpt)
    state_dict = checkpoint['state_dict']
    try:
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k]=v

        model.load_state_dict(new_state_dict)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # model.load_state_dict(new_state_dict) 
    return model     