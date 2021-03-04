import matplotlib.pyplot as plt
import numpy as np

class scoreGraph:
    """Used for generating bar graphs for scores"""

    def __init__(self, scores, title=""):
        fontSize = 20
        figSize = (10, 5)
        barWidth = 0.2
        yRange = np.arange(0, 1.1, 0.1)

        plt.rcParams.update({'font.size': fontSize})

        fig, ax = plt.subplots(figsize=figSize)
        bars = plt.bar(scores.keys(), scores.values(), width=barWidth)

        for bar in bars:
            height = bar.get_height()
            label_x_pos = bar.get_x() + bar.get_width() / 2
            ax.text(label_x_pos, height, s=f'{height}', ha='center', va='bottom')
        plt.yticks(yRange)
        plt.title(title + " Scores On Test-Set")
        plt.show()
