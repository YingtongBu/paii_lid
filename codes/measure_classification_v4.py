#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from collections import defaultdict, namedtuple, Counter

import abc
import datetime
import functools
import glob
import heapq
import itertools
import logging
import math
import multiprocessing as mp
import numpy as np
import operator
import optparse
import os
import pickle
import pprint
import queue
import random
import re
import struct
import sys
import tempfile
import time
import typing
import scipy
import socket

EPSILON     = 1e-6

class Measure:
  @staticmethod
  def stat_data(true_labels: list):
    labels = Counter(true_labels)
    result = [
      f"#label: {len(set(true_labels))}",
      f"#sample: {len(true_labels)}",
    ]
    for label in sorted(labels.keys()):
      c = labels[label]
      ratio = c / len(true_labels)
      result.append(
        f"label[{label}]: (count={c}, percent={ratio * 100:.4} %)"
      )

    return " ".join(result)

  @staticmethod
  def calc_classification(true_labels: list, preded_labels: list):
    ret = Measure.calc_precision_recall_fvalue(true_labels, preded_labels)
    # ret["kappa_coefficient"] = Measure.calc_kappa_coefficient(
    #   true_labels, preded_labels
    # )
    return ret

  @staticmethod
  def calc_precision_recall_fvalue(true_labels: list, preded_labels: list):
    '''
    :return (recall, precision, f) for each label, and
    (average_f, weighted_f, precision) for all labels.
    '''
    assert len(true_labels) == len(preded_labels)
    true_label_num = defaultdict(int)
    pred_label_num = defaultdict(int)
    correct_labels = defaultdict(int)

    for t_label, p_label in zip(true_labels, preded_labels):
      true_label_num[t_label] += 1
      pred_label_num[p_label] += 1
      if t_label == p_label:
        correct_labels[t_label] += 1

    result = dict()
    label_stat = Counter(true_labels)
    for label in label_stat.keys():
      correct = correct_labels.get(label, 0)
      recall = correct / (true_label_num.get(label, 0) + EPSILON)
      prec = correct / (pred_label_num.get(label, 0) + EPSILON)
      f_value = 2 * (recall * prec) / (recall + prec + EPSILON)
      result[label] = {
        "recall": round(recall, 4),
        "precision": round(prec, 4),
        "f": round(f_value, 4),
      }

    total_f = sum([result[label]["f"] * label_stat.get(label, 0)
                   for label in label_stat.keys()])
    if len(true_labels) !=0:
      weighted_f_value = total_f / len(true_labels)
    else:
      weighted_f_value = 0
    result["weighted_f"] = round(weighted_f_value, 4)
    if len(true_labels) != 0:
      result["accuracy"] = round(
        sum(correct_labels.values()) / len(true_labels), 4
      )
    else:
      result["accuracy"] = 0
    result["data_description"] = Measure.stat_data(true_labels)

    return result

  @staticmethod
  def calc_kappa_coefficient(true_labels: list, preded_labels: list):
    '''https://en.wikipedia.org/wiki/Cohen%27s_kappa'''
    assert len(true_labels) == len(preded_labels)
    size = len(true_labels)

    p0 = sum([e1 == e2 for e1, e2 in zip(true_labels, preded_labels)])
    if len(true_labels) !=0 :

      p0 /= size
    else:
      p0 = 0

    counter_true = Counter(true_labels)
    counter_pred = Counter(preded_labels)
    pe = 0
    for k, c1 in counter_true.items():
      c2 = counter_pred.get(k, 0)
      pe += c1 * c2
    pe /= (size * size)

    value = (p0 - pe) / (1 - pe + EPSILON)

    return value

def find_best_combination(true_labels: list, pred_labels: list, label_num: int):
  assert 3 <= label_num <= 5
  values = []
  for labels in itertools.combinations(range(6), 6 - label_num + 1):
    target_label = min(labels)
    true_labels_replace = [target_label if d in labels else d
                           for d in true_labels]
    pred_labels_replace = [target_label if d in labels else d
                           for d in pred_labels]
    measure = Measure.calc_classification(true_labels_replace,
                                          pred_labels_replace)
    values.append([measure["accuracy"], measure["weighted_f"], labels, measure])

  opt_value1 = max(values)
  opt_value2 = max(values, key=lambda item: item[1])
  if  opt_value1[2] == opt_value2[2]:
    print(title_str(f"optimal combination for {label_num} classes"),
          f"by [accuracy] and [weighted_f]")
    print(f"Combing classes: {opt_value1[2]}: ")
    beautify_result(opt_value1[3])
    print()

  else:
    print(f"---optimal combination for {label_num} classes, by [accuracy]")
    print(f"Combing classes: {opt_value1[2]}: {opt_value1[0]}")
    beautify_result(opt_value1[3])
    print()

    print(f"---optimal combination for {label_num} classes, by [weighted_f]")
    print(f"Combing classes: {opt_value2[2]}: {opt_value2[1]}")
    beautify_result(opt_value2[3])
    print()

def beautify_result(result: dict):
  print(f"weighted_f: {result['weighted_f']}, accuracy: {result['accuracy']}")
  label_names = {
    0: "0[us]",
    1: "1[uk]",
    2: "2[neutral]",
    3: "3[ligh]",
    4: "4[moderate]",
    5: "5[heavy]",
  }
  print()
  for d in range(6):
    if d in result:
      print(f"{label_names[d]}: {result[d]}")

  print()
  print("data desc:", result["data_description"])

def title_str(title: str):
  w = (48 - (len(title) + 2)) // 2
  return w * "-" + f" {title} " + w * "-"

def print_300_dist_result(true_labels: list, preded_labels: list):
  dist_300 = [127, 59, 42, 35, 23, 14]
  samples = defaultdict(list)
  for l1, l2 in zip(true_labels, preded_labels):
    samples[l1].append(l2)
  ratio = min([int(len(samples[c]) / dist_300[c]) for c in range(6)])
  print(f"actual data: {ratio} x 300")

  filtered_true_labels = []
  filtered_preded_labels = []
  for c in range(6):
    exp_c_num = dist_300[c] * ratio
    filtered_true_labels.extend([c] * exp_c_num)
    filtered_preded_labels.extend(samples[c][: exp_c_num])

  result = Measure.calc_classification(filtered_true_labels,
                                       filtered_preded_labels)
  beautify_result(result)
  print()

def main(in_file):
  lines = open(in_file).readlines()[1:]
  labels = [int(token) for token in " ".join(lines).split()]
  true_labels = labels[::2]
  preded_labels = labels[1::2]

  print("*" * 60)
  print("version 4:")
  print(title_str("All 6 classes"))
  result = Measure.calc_classification(true_labels, preded_labels)
  beautify_result(result)
  print()

  print(title_str("Result in 300 data distribution"))
  print_300_dist_result(true_labels, preded_labels)

  print(title_str("4 non-native classes"))
  true_labels_4, preded_labels_4 = list(zip(
    *[[d1, d2] for d1, d2 in zip(true_labels, preded_labels)
      if d1 not in [0, 1]]
  ))
  result = Measure.calc_classification(true_labels_4, preded_labels_4)
  beautify_result(result)
  print()

  find_best_combination(true_labels, preded_labels, 5)
  find_best_combination(true_labels, preded_labels, 4)
  find_best_combination(true_labels, preded_labels, 3)

  print("*" * 32)

if __name__ == "__main__":
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--input_file", default=None)
  (options, args) = parser.parse_args()

  assert options.input_file is not None

  main(options.input_file)

