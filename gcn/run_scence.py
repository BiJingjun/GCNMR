

import sys
import os
import copy
import json
import datetime

opt = dict()

#opt['dataset'] = "citeseer"
#opt['weight_decay'] = 5e-2
# opt['losslr1'] = 0.00001
# opt['decay_lr'] = 1.0
# opt['lr2']=0.01

def generate_command(opt):
    cmd = 'python train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))





#sname = ("citeseer1", "citeseer2", "citeseer3", "citeseer4", "citeseer5", "citeseer6", "citeseer7", "citeseer8", "citeseer9", "citeseer10" )
sname = ("scence1", "scence2", "scence3", "scence4", "scence5", "scence6", "scence7", "scence8", "scence9", "scence10" )

#laList = [0, 0.1, 1, 10, 50, 100,500,1000,10000,50000]
#
laList = [0,0.1,1,10,100,500,1000]
#

for dname in sname:
    for lamb in laList:
        opt['lamb'] = lamb

        opt['dataset'] = dname

        run(opt)