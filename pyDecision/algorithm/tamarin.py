###############################################################################

# Required Libraries
import matplotlib.pyplot as plt
import numpy as np

###############################################################################

# Function: Rank


def ranking(flow):
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i, 0])), size=12, ha='center',
                 va='center', bbox=dict(boxstyle='round', ec=(0.0, 0.0, 0.0), fc=(0.8, 1.0, 0.8),))
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1],
                  head_width=0.01, head_length=0.2, overhang=0.0, color='black', linewidth=0.9, length_includes_head=True)
    axes = plt.gca()
    axes.set_xlim([-1, +1])
    ymin = np.amin(rank_xy[:, 1])
    ymax = np.amax(rank_xy[:, 1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.show()
    return

# Function: ARAS (Additive Ratio Assessment)


def tamarin_method(dataset, weights, blurs, criterion_type, graph=True, verbose=True):
    # an alternative is an array of n values
    # dataset is an array of m alternatives
    # Weights is an array of n values
    # blurs is an array of n values
    # criterion_type is an array of n "min" or "max" strings

    # Normalize the dataset
    minValue = np.min(dataset, axis=0)
    maxValue = np.max(dataset, axis=0)
    normalized_dataset = (dataset - minValue) / (maxValue - minValue)
    # dominance
    dominance = np.copy(dataset)*0

    # for each alternatives, for each value compute how many are dominated
    for alternative_1 in range(0, dataset.shape[0]):
        for alternative_2 in range(0, dataset.shape[0]):
            for criteria in range(0, dataset.shape[1]):
                # skip
                if (alternative_1 == alternative_2):
                    continue
                if (criterion_type[criteria] == 'max'):
                    if (normalized_dataset[alternative_1][criteria] - blurs[criteria] > normalized_dataset[alternative_2][criteria]):
                        dominance[alternative_1][criteria] += 1
                elif (criterion_type[criteria] == 'min'):
                    if (normalized_dataset[alternative_1][criteria] + blurs[criteria] < normalized_dataset[alternative_2][criteria]):
                        dominance[alternative_1][criteria] += 1

    # normalisation
    maximumDominance = [max(dominance[:, i])
                        for i in range(0, dominance.shape[1])]
    dominance = dominance / maximumDominance

    sumWeight = sum(weights)
    score = [sum(dominance[alternative] * weights) /
             sumWeight for alternative in range(0, dominance.shape[0])]

    score = np.array(score)
    flow = np.copy(score)
    flow = np.reshape(flow, (score.shape[0], 1))
    flow = np.insert(flow, 0, list(range(1, score.shape[0]+1)), axis=1)
    if (verbose == True):
        for i in range(0, flow.shape[0]):
            print('a' + str(int(flow[i, 0])) +
                  ': ' + str(round(flow[i, 1], 3)))
    if (graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return flow

###############################################################################
