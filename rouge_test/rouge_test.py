import os
import pyrouge
import logging
import tensorflow as tf


def rouge_eval(ref_dir, dec_dir):
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


if __name__ == '__main__':
  print(rouge_eval("./ref/", "./dec/"))