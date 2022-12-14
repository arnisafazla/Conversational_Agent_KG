#!/usr/bin/env bash
# -- coding: utf-8 --

# Example call:
#   bash query/run.sh

pip install -r ./query/requirements.txt
pip install -U sentence-transformers
pip install transformers
python3 ./query/main.py