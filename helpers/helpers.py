import numpy as np
import os
from tensorpack.callbacks import *
from tensorpack import *

def get_tensors_from_graph(G, filter_fn):
    ops = [(n, n.name) for n in G.get_operations()]
    ops = [n for n, name in ops if filter_fn(name)]
    return [m.values()[0] for m in ops]

class DumpTensorsOnce(Callback):
    """
    Dump some tensors to a file.
    Every step this callback fetches tensors and write them to a npz file under ``logger.LOG_DIR``.
    The dump can be loaded by ``dict(np.load(filename).items())``.
    """
    def __init__(self, model, filename='saved_tensors'):
        """
        Args:
            names (list[str]): names of tensors
        """

        self._model = model
        self.save_dir = logger.LOG_DIR

        def fn(*args):
            dic = {}
            for tensor, val in zip(self._model.ternary_weight_tensors, args):
                # print("saving", tensor.op.name, val.shape)
                dic[tensor.op.name] = val
            fname = os.path.join(self.save_dir, '{}.npz'.format(filename))
            np.savez(fname, **dic)

        self._fn = fn

    def _trigger(self):
        vals = self.trainer.sess.run(self._model.ternary_weight_tensors)
        self._fn(*vals)

def read_labels(label_file):
    labels = np.zeros(50000).astype(np.uint8)
    with open(label_file, 'r') as f:
        n, header_seen = 0, False
        for line in f:
            if not header_seen:
                header_seen = True
                continue
            labels[n] = int(line.strip().split(',')[1])
            n += 1
    return labels


def _mkdir(newdir):
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            _mkdir(head)
        #print "_mkdir %s" % repr(newdir)
        if tail:
            os.mkdir(newdir)
