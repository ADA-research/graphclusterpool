import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle 
import sys
import os


if len(sys.argv) != 2:
    print("Arguments not recognized: ", sys.argv)
    print("Please only supply path to dictionary with experiment results.")
    sys.exit(-1)

filepath = sys.argv[1]

if not os.path.exists(filepath):
    print("Invalid path: ", filepath)
    sys.exit(-1)

if not os.path.isfile(filepath):
    print("Path is not a file: ", filepath)

data = None

with open(filepath, 'rb') as pkl:
    data = pickle.load(pkl)

clfName, poolLayer, widthString = data["description"][0], data["description"][1], data["description"][2]
train_loss_f, train_acc_f = data["train_loss_folds"], data["train_acc_folds"]
validation_loss_f, validation_acc_f = data["validation_loss_folds"], data["validation_acc_folds"]

sns.set()
sns.set_style("darkgrid")
fig, axes = plt.subplots(2, 2)
fig.tight_layout(pad=1.0)
fig.suptitle(f"Training metrics {clfName} {poolLayer})")
fig.subplots_adjust(top=0.9)

tloss = np.mean(train_loss_f, axis=0)
vloss = np.mean(validation_loss_f, axis=0)
tacc = np.mean(train_acc_f, axis=0)
vacc = np.mean(validation_acc_f, axis=0)

min_tloss = np.min(train_loss_f, axis=1)
min_vloss = np.min(validation_loss_f, axis=1)
avg_min_t_loss = np.mean(min_tloss)
avg_min_v_loss = np.mean(min_vloss)
std_min_t_loss = np.std(min_tloss)
std_min_v_loss = np.std(min_vloss)

max_tacc = np.max(train_acc_f, axis=1)
max_vacc = np.max(validation_acc_f, axis=1)
avg_max_t_acc = np.mean(max_tacc)
avg_max_v_acc = np.mean(max_vacc)
std_max_t_acc = np.std(max_tacc)
std_max_v_acc = np.std(max_vacc)

axes[0][0].set_title(f"Training loss ({avg_min_t_loss:.4f} ± {std_min_t_loss:.4f})")
axes[0][1].set_title(f"Validation loss ({avg_min_v_loss:.4f} ± {std_min_v_loss:.4f})")
axes[1][0].set_title(f"Training Accuracy ({avg_max_t_acc:.4f} ± {std_max_t_acc:.4f})")
axes[1][1].set_title(f"Validation Accuracy ({avg_max_v_acc:.4f} ± {std_max_v_acc:.4f})")

def plot_line_with_std(ax, mean, std, color):
    lower_bound = [M_new - Sigma for M_new, Sigma in zip(mean, std)]
    upper_bound = [M_new + Sigma for M_new, Sigma in zip(mean, std)]
    sns.lineplot(mean, ax=ax, color=color)
    ax.fill_between([x for x in range(len(lower_bound) )], lower_bound, upper_bound, color=color, alpha=.3)

plot_line_with_std(axes[0][0], tloss, np.std(train_loss_f, axis=0), color="red")
plot_line_with_std(axes[0][1], vloss, np.std(validation_loss_f, axis=0), color="red")
plot_line_with_std(axes[1][0], tacc, np.std(train_acc_f, axis=0), color="blue")
plot_line_with_std(axes[1][1], vacc, np.std(validation_acc_f, axis=0), color="blue")

plt.show()