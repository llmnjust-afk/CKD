#!/usr/bin/env python3
"""Collect every run under save/student_data/ into a markdown table:
per run -> loss weights / temperature / fusion settings / best accuracy."""
from __future__ import print_function

import glob
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else './save/student_data'
rows = []
for cfg_path in sorted(glob.glob(os.path.join(ROOT, '*', 'config.json'))):
    folder = os.path.dirname(cfg_path)
    with open(cfg_path) as f:
        c = json.load(f)
    res_path = os.path.join(folder, 'result.txt')
    acc = ''
    if os.path.exists(res_path):
        with open(res_path) as f:
            for line in f:
                if line.startswith('best_acc'):
                    acc = line.split()[1]
    name = os.path.basename(folder)
    rows.append((name, c, acc))

cols = ['run', 'dataset', 'model_s', 'method', 'alpha', 'beta', 'lam',
        'gamma', 'eta', 'kd_T', 'kd_t2', 'full_grad_kl', 'fusion_arch',
        'fusion_dim', 'learning_rate', 'epochs', 'batch_size', 'best_acc']
print('| ' + ' | '.join(cols) + ' |')
print('|' + '---:|' * len(cols))
for name, c, acc in rows:
    vals = [name]
    for k in cols[1:-1]:
        vals.append(str(c.get(k, '')))
    vals.append(acc)
    print('| ' + ' | '.join(vals) + ' |')

summary = os.path.join(ROOT, '..', 'sweep', 'all_runs_table.md')
try:
    os.makedirs(os.path.dirname(summary), exist_ok=True)
    with open(summary, 'w') as f:
        f.write('# All CKD runs\n\n')
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('|' + '---:|' * len(cols) + '\n')
        for name, c, acc in rows:
            vals = [name] + [str(c.get(k, '')) for k in cols[1:-1]] + [acc]
            f.write('| ' + ' | '.join(vals) + ' |\n')
    print('\nwritten to', summary)
except OSError:
    pass
