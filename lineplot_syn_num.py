import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ucf = [46.55, 46.52, 46.67, 46.58, 47.14, 47.69, 47.41, 46.89]
hmdb = [36.08, 35.85, 35.84, 35.94, 35.78, 35.84, 35.70, 35.76]

syn_num = [200, 400, 600, 800, 1000, 1200, 1400, 1600]

# Make a data frame
df = pd.DataFrame({'x': syn_num, 'UCF101': ucf, 'HMDB51': hmdb})

# Change the style of plot
plt.style.use('seaborn-darkgrid')

# Create a color palette
palette = plt.get_cmap('Set1')

num = 0
for column in df.drop('x', axis=1):
    num += 1
    # Find the right spot on the plot
    plt.subplot(2, 1, num)

    # Plot the lineplot
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)

    # Add title
    #plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num))
    plt.legend([column])
    plt.ylabel("Average Accuracy")

# general title
#plt.suptitle("The ZSAR performances", fontsize=10, fontweight=0, color='black', y=1)

# Axis titles
#plt.text(2, 2, 'Time', ha='center', va='center')
#plt.text(4, 4, 'Note', ha='center', va='center', rotation='vertical')
plt.xlabel("The number of synthesised unseen visual embeddings")

# Show the graph
#plt.show()
plt.savefig('syn_num.png', dpi=1000)
