import os

import seaborn as sns

results_dir = "/Volumes/Macintosh HD/Users/khurramjaved96/Results_ICML/All_plots/Plots_with_seeds"
results_dict = {}
std_dict = {}
import pandas as pd
import json
import matplotlib.pyplot as plt

data = {}
folders = []
exeriments_data = []
for experiment_name in os.listdir(results_dir):
    if "DS_St" not in experiment_name:
        for rank in os.listdir(os.path.join(results_dir, experiment_name)):
            if "DS_St" not in rank:
                for seed in os.listdir(os.path.join(os.path.join(results_dir, experiment_name), rank)):
                    if "DS_St" not in seed:
                        with open(os.path.join(results_dir, experiment_name, rank, seed, "metadata.json")) as json_file:

                            try:
                                data_temp = json.load(json_file)
                                exeriments_data.append(data_temp)
                                print(experiment_name)
                                if experiment_name in data:

                                    data[experiment_name].append(data_temp['results']['Adaptation loss'][0:40000])
                                else:
                                    folders.append(experiment_name + rank)
                                    data[experiment_name] = []
                                    data[experiment_name].append(data_temp['results']['Adaptation loss'][0:40000])
                            except:
                                pass
                    # print(data)e
    # quit()
    # print(f)

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=0.4, rc={"lines.linewidth": 2.0})

mem = '0'
counter = 0
for current_experimnet_data in data:
    dictionary_temp = {}
    dictionary_temp["x"] = []
    dictionary_temp["y"] = []
    for run in data[current_experimnet_data]:
        run_temp = []
        running_val = run[0]
        for val in run:
            run_temp.append(running_val)
            running_val = 0.90*running_val + 0.10*val
        dictionary_temp["x"] = dictionary_temp["x"] + list(range(len(run)))
        dictionary_temp["y"] = dictionary_temp["y"] + run_temp
    # print(results_dict[folder])
    df = pd.DataFrame(dictionary_temp)
    print(df)
    # df['#Classes'] = df.index
    # print(df)
    # print(df.columns)
    # df = pd.DataFrame.from_dict(results_dict[folder], orient='index')
    # df['Episode'] = df.index
    # df = df.melt('#Classes', var_name='cols', value_name='Accuracy')
    # print(df)
    # # print(df['vals'])
    plt.ylim(0, 0.5)

    sns.lineplot(x='x', y='y', data=df, legend='full', label=folders[counter], ci="sd")
    counter += 1
    #

    #     # plt.errorbar(list(results_dict[folder].keys()), list(results_dict[folder].values()), yerr= list(std_dict[folder].values()) , marker='s')

plt.tight_layout()
plt.savefig("plots/rebuttal_adaptation_smooth.png", dpi=300, format="png")
quit()