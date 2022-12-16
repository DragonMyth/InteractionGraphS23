'''
Descrition:
    This module extracts BVH files from AMASS data set.
    The average body shape will be used for the skeleton of BVH files.
Examples:
    python3 bvh_from_amass.py --files xxx
    python3 bvh_from_amass.py --dirs xxx --fps 30 --fix_height
'''


import sys, os
import numpy as np
import torch

from fairmotion.ops import conversions
from fairmotion.ops import math
from fairmotion.core import motion as mo_core
from fairmotion.ops import motion as mo_ops

from fairmotion.data import amass
from fairmotion.data import bvh,asfamc

from fairmotion.utils import utils
# from fairmotion.utils import multiprocessing as mp
from fairmotion.viz.utils import TimeChecker

from heapq import nsmallest

import argparse

import re

def arg_parser():
    parser = argparse.ArgumentParser()
    ''' Files to process '''
    parser.add_argument(
        '--asf', action='append', default="/private/home/yzhang3027/ScaDive/data/motion/salsa/60/60.asf")
    ''' Directories to process '''
    parser.add_argument(
        '--amc', action='append', default="/private/home/yzhang3027/ScaDive/data/motion/salsa/60/60_01.amc")
    
    return parser

if __name__ == '__main__':

    args = arg_parser().parse_args()

    asf = args.asf
    amc = args.amc

    motion = asfamc.load(file=asf,motion=amc)
    name = "/private/home/yzhang3027/ScaDive/data/motion/salsa/60/cmu_60_01.bvh"
    bvh.save(motion, (name))
