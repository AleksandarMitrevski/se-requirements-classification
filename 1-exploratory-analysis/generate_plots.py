import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

classes = ['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']

def get_counts(df, class_col_name):
    counts = {key:0 for key in classes}
    for key in counts.keys():
        counts[key] = df[df[class_col_name] == key].shape[0]
    return counts

# read datasets and count
nfr_df = pd.read_csv('./data/nfr.csv', sep=',', header=0, quotechar = '"')
nfr_counts = get_counts(nfr_df, 'class')

promise_exp_df = pd.read_csv('./data/PROMISE_exp.csv', sep=',', header=0, quotechar = '"', doublequote=True)
promise_exp_counts = get_counts(promise_exp_df, '_class_')

# calculate values for plot for all classes
nfr_values = [nfr_counts[key] for key in classes]
promise_exp_values = [promise_exp_counts[key] for key in classes]
promise_exp_diffs = [a1 - a2 for (a1, a2) in zip(promise_exp_values, nfr_values)]
indeces = np.arange(len(classes))
bar_width = 0.45

y1 = plt.bar(indeces, nfr_values, bar_width)
y2 = plt.bar(indeces, promise_exp_diffs, bar_width, bottom=nfr_values)

# draw plot for all classes

plt.title('PROMISE_exp class distribution')
plt.xlabel('Requirement type')
plt.ylabel('Count')
plt.xticks(indeces, nfr_counts.keys())
plt.yticks(np.arange(0, 451, 50))
plt.legend((y1[0], y2[0]), ('NFR', 'PROMISE_exp'))

plt.savefig('./stacked_plot_classes.png')

plt.close()

# calculate values for plot for FR vs NFR
nfr_count_nfr = sum([nfr_counts[key] for key in classes[1:]])
nfr_values = [nfr_counts['F'], nfr_count_nfr]
promise_exp_count_nfr = sum([promise_exp_counts[key] for key in classes[1:]])
promise_exp_diffs = [promise_exp_counts['F'] - nfr_counts['F'], promise_exp_count_nfr - nfr_count_nfr]
indeces = np.arange(2)
bar_width = 0.6

y1 = plt.bar(indeces, nfr_values, bar_width)
y2 = plt.bar(indeces, promise_exp_diffs, bar_width, bottom=nfr_values)

# draw plot for FR vs NFR

plt.title('PROMISE_exp class distribution - FR versus NFR')
plt.xlabel('Requirement type')
plt.ylabel('Count')
plt.xticks(indeces, ['FR', 'NFR'])
plt.yticks(np.arange(0, 501, 50))
plt.legend((y1[0], y2[0]), ('NFR', 'PROMISE_exp'))

fig = plt.gcf()
fig.set_size_inches(4.5, 4.8)
fig.savefig('stacked_plot_fr_nfr.png')
