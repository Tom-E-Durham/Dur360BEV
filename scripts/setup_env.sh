#!/bin/bash

python3 -m venv dur360bev
source dur360bev/bin/activate
pip install -r requirements.txt
cd coarse_fine/src/ops/gs; python setup.py build install; cd -