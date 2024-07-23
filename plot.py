# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pyformat: disable

import os
import scipy.stats

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import functools
import torch
# Look at me being proactive!
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    print("about to load datmodel scores")
    print("score shape: " + str(score.shape))
    distances = inputs = np.load("outlier_distances.npy")
    print("distances shape: " + str(distances.shape))
    corr = np.corrcoef(distances,score)
    print("corr: " + str(corr))
    fpr2,tpr2,thresholds = roc_curve(x,distances)
    fpr, tpr, thresholds = roc_curve(x, -score)
    acc = np.max(1-(fpr2+(1-tpr2))/2)
    return fpr2, tpr2, auc(fpr2, tpr2), acc

def load_data(p):
    """
    Load our saved scores and then put them into a big matrix.
    """
    global scores, keep
    scores = []
    keep = []

    for root,ds,_ in os.walk(p):
        for f in ds:
            print("f: " + str(f))
            if not f.startswith("experiment"): continue
            if not os.path.exists(os.path.join(root,f,"scores")): continue
            last_epoch = sorted(os.listdir(os.path.join(root,f,"scores")))
            if len(last_epoch) == 0: continue
            scores.append(np.load(os.path.join(root,f,"scores",last_epoch[-1])))
            keep.append(np.load(os.path.join(root,f,"keep.npy")))
            print("scores len: " + str(len(scores[0])) + ", keep len: " + str(len(keep[0])))
    scores = np.array(scores)
    keep = np.array(keep)[:,:scores.shape[1]]
    print("final scores len: " + str(len(scores)) + ", keep len: " + str(len(keep)))
    return scores, keep

def generate_ours(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000,
                  fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    print("ck size: " + str(len(check_keep)) + ", " +str(len(check_keep[0])))
    print("k size: " + str(len(keep)) + ", " +str(len(keep[0])))
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        if j==0:
            print("keep[:,j]: "+str(keep[:,j]) +", scores[keep[:,j],j,:]: "+str(scores[keep[:,j],j,:]))
        dat_in.append(scores[keep[:,j],j,:])
        dat_out.append(scores[~keep[:,j],j,:])
    in_size = min(min(map(len,dat_in)), in_size)
    out_size = min(min(map(len,dat_out)), out_size)
    print("in size: " + str(in_size) + ", out size: " + str(out_size))
    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])
    #print("dat in size: " + str(len(dat_in)) + ", " + str(len(dat_in[0])) + ", " + str(len(dat_in[0][0]))+ ", " + str(len(dat_in[0][0][0])))
    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)
    print("mean len: " + str(len(mean_in)) + ", " + str(len(mean_in[0])))
    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    printcount=0
    for ans, sc in zip(check_keep, check_scores):
        printcount+=1
        if printcount==1:
            print("len of sc: " + str(len(sc))+ ", len of output: " + str(len(scipy.stats.norm.logpdf(sc, mean_in, std_in+1e-30))))
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in+1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)
        score = pr_in-pr_out
        if printcount==1:
            print("score: " + str(score))
        prediction.extend(score.mean(1))
        answers.extend(ans)

    return prediction, answers

def generate_ours_offline(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000,
                          fix_variance=False):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    out_size = min(min(map(len,dat_out)), out_size)

    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)

        prediction.extend(score.mean(1))
        answers.extend(ans)
    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc.mean(1))
        answers.extend(ans)

    return prediction, answers

def do_plot(fn, keep, scores, ntest, legend='', metric='auc', sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(keep[:-ntest],
                             scores[:-ntest],
                             keep[-ntest:],
                             scores[-ntest:])

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr<.001)[0][-1]]

    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f'%(legend, auc,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text, **plot_kwargs)
    return (acc,auc)


def fig_fpr_tpr():

    plt.figure(figsize=(4,3))

    do_plot(generate_ours,
            keep, scores, 1,
            "Ours (online)\n",
            metric='auc'
    )

    do_plot(functools.partial(generate_ours, fix_variance=True),
            keep, scores, 1,
            "Ours (online, fixed variance)\n",
            metric='auc'
    )

    do_plot(functools.partial(generate_ours_offline),
            keep, scores, 1,
            "Ours (offline)\n",
            metric='auc'
    )

    do_plot(functools.partial(generate_ours_offline, fix_variance=True),
            keep, scores, 1,
            "Ours (offline, fixed variance)\n",
            metric='auc'
    )

    do_plot(generate_global,
            keep, scores, 1,
            "Global threshold\n",
            metric='auc'
    )

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig("/tmp/fprtpr2.png")
    plt.show()


if __name__ == '__main__':
    load_data("exp/cifar10/")
    fig_fpr_tpr()
