import numpy as np
import torch
import random


'''
random seed
'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

'''
write
'''
def writetxt(result_list,save_file):
    with open(save_file,"w") as f:
        for i in result_list:
            #print(i)
            f.write('''{} {} {}'''.format(i[0],i[1],i[2]))
            f.write("\n")
    f.close()

'''
It is used to watch the parameter of grad.
Example: watch the loss grad
    loss.register_hook(save_grad('loss'))
    loss.backward()
    print(grads['loss'])
'''
grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook
 
