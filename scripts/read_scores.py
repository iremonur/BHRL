import os
import numpy as np
import matplotlib.pyplot as plt

seq = "person20"


def choose_scores(seq):
    cnt=0
    scores = np.load("/truba/home/ionur/BHRL/target_update_studies/{}/scores.npy".format(seq))
    for idx, score in enumerate(scores):
        if score < 0.7:
            print(idx, score)
            cnt+=1
    print(cnt)



def plot_scores(seq):

    scores = np.load("/truba/home/ionur/BHRL/target_update_studies/{}/scores.npy".format(seq))
    print(scores)
    print(len(scores))

    indices_to_change = scores == -100
    scores[indices_to_change] = 0
    indexes = np.arange(len(scores))
    plt.scatter(range(len(scores)), scores, c=np.where(indices_to_change, 'red', 'blue'), s=2)
    plt.scatter(range(len(scores)), np.where(indices_to_change, scores, np.nan), c='red', s= 1, label='Target is not found \n for {} frames !'. format(np.sum(indices_to_change)))
    plt.legend(fontsize='x-small')
    plt.xlabel("Frame Index")
    plt.ylabel("Score")
    plt.title("{} - Scores". format(seq))
    plt.savefig('/truba/home/ionur/BHRL/target_update_studies/{}/{}_scores.png'.format(seq, seq))

def read_preds(seq):
    bboxes = np.load("/truba/home/ionur/BHRL/target_update_studies/{}/preds.npy".format(seq))
    print(bboxes[0][:4])

    for idx, bbox in enumerate(bboxes):
        if all(item == 0 for item in bbox[:4]):
            print("yes")
        else:
            print(bbox[:4])
    print(len(bboxes))

read_preds(seq)