import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle 
import sys
import os
from pathlib import Path


sns.set_theme()
#sns.set_style("darkgrid")
printdata = False
if len(sys.argv) == 1:
    [y for x in Path("results").iterdir() for y in x.iterdir()]
    paths = [(y / "results_dictionary.pkl") for x in Path("results").iterdir() for y in x.iterdir()
             if y.is_dir() and not any([file for file in y.iterdir() if file.suffix == ".png"])]
    paths = [x for x in paths if x.exists()]
    if len(paths) == 0:
        print("Could not auto detect any plots still to be made. Exiting.")
        sys.exit()
else:
    if len(sys.argv) == 3:
        printdata = True
    filepath = Path(sys.argv[1])

    if not filepath.exists():
        print("Invalid path: ", filepath)
        sys.exit(-1)

    if not filepath.is_file():
        files = [x for x in os.listdir(filepath) if x.endswith(".pkl")]
        filepath = filepath / files[0]
        if not filepath.is_file():
            print("Path is not a file: ", filepath)
    paths = [filepath]

for filepath in paths:
    data = None
    with open(filepath, 'rb') as pkl:
        try:
            data = pickle.load(pkl)
        except Exception as err:
            print(f"Couldn't load file {filepath} because of Exception: {err}")
            continue

    clfName, poolLayer, widthString = data["description"][0], data["description"][1], data["description"][2]
    train_loss_f, train_acc_f = data["train_loss_folds"], data["train_acc_folds"]
    validation_loss_f, validation_acc_f = data["validation_loss_folds"], data["validation_acc_folds"]
    if "test_set_scores" in data.keys() and len(data["test_set_scores"]) > 0:
        test_data = data["test_set_scores"]
        print(f"Test set score for {filepath.parent.name} experiment: {np.mean(test_data)} +/- {np.std(test_data)} [Out of {len(test_data)} folds]")
        print(test_data)
        print()
    if printdata:
        print(data["description"])
        bve = [np.argmax(f) for f in data["validation_acc_folds"]]
        maxval = [np.round(np.max(f), 4) for f in data["validation_acc_folds"]]
        print(f"Best validation epochs ({np.min(bve)}-{np.max(bve)}):\n {bve}\n{maxval}")
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

    plt.savefig(str(filepath) + ".png")
    plt.show()