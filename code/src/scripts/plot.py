# TODO refactor this huge file with duplication into config-based plotting 

import os
import typing

import sys 

import numpy
import pandas
from fairlearn.metrics import demographic_parity_difference
from matplotlib import interactive, pyplot
from matplotlib import rc
import argparse
from texttable import Texttable

from lib import os_utils
# matplotlib config
from src.common.datasets.adult import load_adult
from src.scripts.lp import get_optimal_front

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 20})
rc('text', usetex=True)
rc('figure', figsize=[10.0, 10.0], dpi=100)
rc('axes', titlesize=24, labelsize=24)
rc('xtick', labelsize=24)
rc('ytick', labelsize=24)

# CONSTANTS
TITLE_PAD = 20

COLOR = ["#e41a1c", "#984ea3", "#ff7f00", "#377eb8", "#000000", "#4daf4a", "#964B00", "#f781bf", "#000000", "#609"]

#MARKER = ["o", "s", "X", "d", "8", "P", "*", "^"]
MARKER = ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o", "o"]

FIGURES_FOLDER = "plots"
METRIC_TO_READ = ["acc", "dp", "eo", "eopp"]
FORMAT = "pdf"


def get_dataframe_from_results(results, has_ub=False, metrics=METRIC_TO_READ):
    data = []
    columns = []
    count = 0
    # remove checksums
    if results.get("checksums") is not None:
        del results["checksums"]
    read_column_name = False

    for idx, (name, result) in enumerate(results.items()):

        if len(result) > 1:
            print(
                "Warning: More than one result found for a folder. Currently just using the first one")

        if len(result) == 0:
            print(f"skipping {name}")
            continue

        inner_res = result[0].result
        has_metric_ub = {} 
        for m in metrics:
            has_metric_ub[m] = f'{m}_ub' in list(inner_res.values())[0]

        if len(result[0].result.keys()) == 0:
            continue
        elif read_column_name is False:
            for key in sorted(result[0].result.keys()):
                for metric in metrics:
                    columns.append(f"{key}_{metric}")
                    if has_metric_ub[metric]:
                        columns.append(f"{key}_{metric}_ub")

            count += len(columns)
            for key in sorted(result[0].params.keys()):
                columns.append(key)
            # print(result[0].params.keys())
            read_column_name = True
            # print(f"Column name read at {idx}")
        # adding result data here
        arr = []
        for model_result in sorted(result[0].result):
            # model_result is model type : nn, logistic reg etc.
            for metric in metrics:
                # aggregate
                if metric in ["dp", "dp_soft", "eo", "eopp"]:
                    val = numpy.max(
                        list(map(lambda k: numpy.mean(k.result[model_result][metric]), result)))
                else:
                    val = numpy.mean(
                        list(map(lambda k: numpy.mean(k.result[model_result][metric]), result)))

                arr.append(val)

                if has_metric_ub[metric]:
                    val = numpy.mean(list(map(lambda k: k.result[model_result][f'{metric}_ub'], result)))
                    arr.append(val)


        for i in range(count, len(columns)):
            arr.append(result[0].params[columns[i]])
        # read params and add that here
        data.append(arr)
    # breakpoint()
    df = pandas.DataFrame(data, columns=columns)
    return df


def get_pareto(accs, unfairness):
    # Plot only pareto of given points 
    tmp = []

    n = accs.shape[0]
    for i in range(n):
        on_pareto = True
        for j in range(n):
            if i == j:
                continue
            p1 = accs[j] > accs[i]
            p2 = unfairness[j] < unfairness[i]
            if p1 and p2:
                #print(f'Point {i} killed by {j}')
                on_pareto = False
                break 
        if on_pareto:
            tmp.append((accs[i], unfairness[i]))

    # connect with lines only the pareto points
    vals = sorted(tmp, key= lambda x: (-x[0])*1000000 - x[1]) ###############
    accsr = [x[0] for x in vals]
    unfairnessr = [x[1] for x in vals]
    return accsr, unfairnessr 

##########################################
##########################################

def figure_diff_models(args):
    plot_acc = 0.793 
    plot_dp = 0.262 
    plot_ub = 0.391
    filename = f"diff_models.pdf"


    t = numpy.load(f"result/_eval/tree_health_notransfer.npy", allow_pickle=True).item()
    # this point is from health_notransfer, a representative one 
    t = dict(dict(dict(t)['k=60,ni=60,a=0.5,s=0.5'][0])['result'])
    # alpha = 0.5, s = 0.5, ni = 60, k = 60 

    accs = []
    dps = []
    
    for method, vals in t.items(): # 24
        acc = list(vals.acc)[0] 
        dp = list(vals.dp)[0]
        aux_acc = vals.aux_acc 
        if 'adversary' in method:
            accs.append(aux_acc)
        else:
            accs.append(acc)
        dps.append(dp)
        print(f'{method}: {acc} vs {dp} and aux = {aux_acc}')
        
    pink = "#f781bf"
    gray = "#555555"
    lb_marker = "^"
    ub_marker = "v"
    MARKER_LW = 2 

    SCATTER_MARKERSIZE = 85

    figure = pyplot.figure(figsize=(5.5, 6.3))
    ax = figure.add_subplot(1, 1, 1)

    ax.scatter(x=[plot_acc], y=[plot_ub], c="none", marker="v", edgecolors=pink, s=SCATTER_MARKERSIZE, linewidth=MARKER_LW, alpha=1)
    #ax.scatter(x=[plot_acc], y=[plot_dp], label='FARE (Empirical)', c="none", marker="^", edgecolors=pink, zorder=-20, s=SCATTER_MARKERSIZE, linewidth=MARKER_LW, alpha=1)

    L = 0.15
    R = 0.85
    ax.set_ylim(bottom=0.23, top=0.44)
    ax.set_xlim(left=L, right=R)
    ax.set_yticks([0.25, 0.3, 0.35, 0.4])
    ax.set_xticks([0.25, 0.5, 0.75])

    ax.hlines(y=plot_ub, xmin=L, xmax=R, color=pink, linewidth=2, linestyle=(0,(2,4)), alpha=0.9, zorder=-20, label='FARE (Upper Bound)')
    #ax.vlines(x=plot_acc, ymin=plot_dp+0.0015, ymax=plot_ub-0.0015, color=pink, linewidth=1.2, linestyle=(0,(2,4)), alpha=0.9, zorder=-20)

    ax.scatter(x=accs, y=dps, label='Downstream Classifiers', c="none", marker="o", edgecolors=gray, s=SCATTER_MARKERSIZE, linewidth=2, alpha=1)

    ax.scatter(x=[plot_acc], y=[plot_dp], c="none", marker="^", edgecolors=pink, s=SCATTER_MARKERSIZE, linewidth=MARKER_LW, alpha=1)

    ax.set_xlabel(f"Accuracy")
    ax.set_ylabel("Demographic Parity Distance")

    ax.legend(fancybox=True, framealpha=0., loc='upper right', prop={'size': 18})
    ax.grid(alpha=0.4)
    #ax.set_title('All Classifiers', pad=TITLE_PAD)

    os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "main"))

    if args.i:
        pyplot.show()
    else:
        pyplot.savefig(os.path.join(FIGURES_FOLDER, "main", filename), bbox_inches='tight')
        pyplot.close()


# Ns plot
def figureNs(args):

    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    prop_idxs = {
        "2": 0, "4": 3, "8": 2, "16": 5, "32": 1,
        '1': 7, 'tree': 7, 'noop': -1 # ?
    }

    interactive = args.i 
    pareto = args.p 
    
    # These are some points from CA 
    for data in ["ACSIncome-CA-2014"]:

        method_array = ["tree"]

        labels = {
            '1': 'M=1',
            '2': 'Upper Bound with M=2',
            '4': 'Upper Bound with M=4',
            '8': 'Upper Bound with M=8',
            '16': 'Upper Bound with M=16',
            '32': 'Upper Bound with M=32',
        }

        dfs = {} 

        t = numpy.load(f"result/_eval/{data}/tree.npy", allow_pickle=True).item()
        dfs['tree'] = get_dataframe_from_results(t, has_ub=True)

        t = numpy.load(f"result/_eval/{data}/noop.npy", allow_pickle=True).item()
        dfs['noop'] = get_dataframe_from_results(t, has_ub=False)

        pln = numpy.load('result/_eval/Ns.npy',allow_pickle=True)
        cs = ['k','ni','a', 's', 'N', 'N_size', 'ub']
        df_plot = pd.DataFrame(data = pln,columns = cs)
        good_idx = dfs['tree'].reset_index().merge(df_plot,how='right').set_index('index').index.unique().to_numpy()
        dfs['tree'] = dfs['tree'].iloc[good_idx]
        df_plot_merged = pd.merge(dfs['tree'], df_plot, on=['k', 'ni', 'a', 's'], how='inner')
        check = df_plot_merged[df_plot_merged['N'] == 1]
        assert numpy.all( (check['nn_1_layer_normalized_dp_ub'] == check[ 'ub']).to_numpy() )
        arrs = { str(i): df_plot_merged[df_plot_merged['N']==i][['nn_1_layer_normalized_acc','ub']] for i in [1,2,4,8,16,32] }
        arrs = { k: arrs[k].rename(columns={'ub':'nn_1_layer_normalized_dp'}) for k in arrs.keys() }
        method_array = list(arrs.keys())
        method_array.append('tree')
        method_array.append('noop')
        arrs['tree'] = dfs['tree']
        arrs['noop'] = dfs['noop']
        dfs = arrs

        for idx, key in enumerate(["nn_1_layer"]):
            figure = pyplot.figure(figsize=(10, 10))
            ax = figure.add_subplot(1, 1, 1)

            # limits 
            XL = 0.69
            ax.set_xticks([0.70, 0.73, 0.76, 0.79, 0.82])

            # also filter points under XL 
            for method in method_array:
                n_before = dfs[method].shape[0]
                dfs[method] = dfs[method][dfs[method]['nn_1_layer_normalized_acc'] > XL]
                n_after = dfs[method].shape[0]
                if n_after < n_before:
                    print(f'[{method}] {n_before} -> {n_after} pruned because of {XL}')

            MARKER_LW = 2 
            SCATTER_MARKERSIZE = 55
            
            # plot all else
            for i, method in enumerate(method_array):
                print(f'\n====================== METHOD: {method}\n')
                print(dfs[method])

                dataframe = dfs[method]
                color_idx = prop_idxs[method]

                tmp = dataframe[['nn_1_layer_normalized_acc', 'nn_1_layer_normalized_dp']].to_numpy() 

                if pareto:
                    xs, ys = get_pareto(tmp[:,0], tmp[:,1])
                    ax.plot(xs, ys, c=COLOR[color_idx], linewidth=2, alpha=1, zorder=-20)
                else:
                    xs, ys = tmp[:,0], tmp[:,1]

                order = numpy.argsort( xs[:] ) 
                xs, ys = numpy.asarray(xs)[ order ], numpy.asarray(ys)[ order ]
                if method == 'tree':
                    sc = ax.plot(xs, ys, c=COLOR[color_idx], linewidth=2)
                    sc2 = ax.scatter(x=xs, y=ys, label='FARE (Empirical)', c="none", marker="^", edgecolors=COLOR[7], s=SCATTER_MARKERSIZE, linewidth=MARKER_LW, alpha=1, zorder=-20)
                elif method == 'noop':
                    pass
                else:
                    if method == '1':
                        sc = ax.plot(xs, ys, c=COLOR[color_idx], linewidth=2)
                        sc1 = ax.scatter(x=xs, y=ys, label='FARE (Upper Bound)', c="none", marker="v", edgecolors=COLOR[7], s=SCATTER_MARKERSIZE, linewidth=MARKER_LW, alpha=1, zorder=-20)
                    else:
                        sc = ax.plot(xs, ys, label=labels[method], c=COLOR[color_idx], linewidth=2)

            ax.set_xticks([0.7, 0.73, 0.76, 0.79, 0.82])
            ax.set_xlim(left = 0.69, right = 0.83)

            if XL > 0:
                ax.set_xlim(left=XL, )
            ax.set_ylim(bottom=-0.002)

            ax.set_xlabel(f"Accuracy")
            ax.set_ylabel("Demographic Parity Distance")

            ax.legend(fancybox=True, framealpha=0., loc=2, prop={'size': 26})

            ax.grid(alpha=0.4)
            os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "main"))
            
            pyplot.savefig(
                os.path.join(FIGURES_FOLDER, "main", f"Ns.pdf"),
                bbox_inches='tight')
            pyplot.close()

def figure_shifts(args):

    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    prop_idxs = {
        "noop": 4, "fcrl": 0, "fnf": 3, "sipm": 2, 
        "fair-path": 6, "tree": 7, "laftr": 1,
        "cvib_supervised": 5
    }
    
    labels = {
        'tree_2015': 'FARE Shifted',
        'tree_2015_true': 'FARE',
    }


    interactive = args.i 
    pareto = args.p 
   
    data = "shifts2"
    method_array = [ "tree_2015", "tree_2015_true" ]
    suffix = ''

    cuts = {
               "shifts2": [ 0.78, 0.79, 0.80, 0.81 ],
           }
    dfs = {} 
    for method in method_array:
        t = numpy.load(f"result/_eval/{data}{suffix}/{method}.npy", allow_pickle=True).item()
        dfs[method] = get_dataframe_from_results(t, has_ub=(method=='tree'),metrics=["acc", "dp"])
    keys_downstream = ['nn_1_layer_normalized']
    assert len( keys_downstream ) >  0
    print(keys_downstream)

    for idx, key in enumerate(keys_downstream): 
        unf_metrics = ['dp']
        for unfairness_metric in unf_metrics:
            unf_name = unfairness_metric 
            unf_key = f'{key}_{unf_name}'
            unf_ub_key = f'{key}_{unf_name}_ub'

            figure = pyplot.figure(figsize=(10, 10))
            ax = figure.add_subplot(1, 1, 1)

            XL = 0.70
            ax.set_xticks([0.72, 0.75, 0.78, 0.81])
            ax.set_xlim(0.695, 0.825)

            # also filter points under XL 
            for method in method_array:
                n_before = dfs[method].shape[0]
                dfs[method] = dfs[method][dfs[method][f'{key}_acc'] > XL]
                n_after = dfs[method].shape[0]
                if n_after < n_before:
                    print(f'[{method}] {n_before} -> {n_after} pruned because of {XL}')

            
            tabledata = {}
            for cut in cuts[data]:
                tabledata[cut] = {}

            for s, method in enumerate(method_array):
                if method == 'tree_2015_true':
                    s = 7
                else:
                    s = 6
                
                if not 'tree' in method:
                    continue
                tree_has_curr_ub = unf_ub_key in dfs[method]

                if tree_has_curr_ub:
                    tmp = dfs[method][[f'{key}_acc', unf_key, unf_ub_key]].to_numpy()
                    tree_acc, tree_unf, tree_unf_ub = tmp[:,0], tmp[:, 1], tmp[:, 2]
                    tree_acc1 = tree_acc
                    tree_acc2 = tree_acc
                else:
                    tmp = dfs[method][[f'{key}_acc', unf_key]].to_numpy()
                    tree_acc, tree_unf = tmp[:,0], tmp[:, 1]
                    tree_acc1 = tree_acc
                    tree_acc2 = tree_acc

                if tree_has_curr_ub:
                    tree_acc1, tree_unf_ub = get_pareto(tree_acc, tree_unf_ub)
                    ax.plot(tree_acc1, tree_unf_ub, c=COLOR[s], linewidth=2, alpha=1, zorder=-20, label=labels[method]+' (Upper Bound)', linestyle='solid')

                    xs = tree_acc1
                    ys = tree_unf_ub
                    xs, ys = zip(*sorted(list(zip(xs,ys))))

                    label = f'{method}-ub'
                    for cut in tabledata:
                        # look for cut?
                        if xs[0] > cut:
                            tabledata[cut][label] = 'it is right'
                        elif xs[-1] < cut:
                            tabledata[cut][label] = 'it is left'
                        else:
                            for i in range(len(xs)-1):
                                if xs[i] < cut and xs[i+1] >= cut:
                                    unf = ((cut - xs[i]) * ys[i+1] + (xs[i+1] - cut) * ys[i]) /  (xs[i+1] - xs[i])
                                    tabledata[cut][label] = unf
                                    break
                tree_acc2, tree_unf = get_pareto(tree_acc, tree_unf)

                xs = tree_acc2
                ys = tree_unf
                xs, ys = zip(*sorted(list(zip(xs,ys))))

                label = f'{method}-emp'
                for cut in tabledata:
                    # look for cut?
                    if xs[0] > cut:
                        tabledata[cut][label] = 'it is right'
                    elif xs[-1] < cut:
                        tabledata[cut][label] = 'it is left'
                    else:
                        for i in range(len(xs)-1):
                            if xs[i] < cut and xs[i+1] >= cut:
                                unf = ((cut - xs[i]) * ys[i+1] + (xs[i+1] - cut) * ys[i]) /  (xs[i+1] - xs[i])
                                tabledata[cut][label] = unf
                                break

                ax.plot(tree_acc2, tree_unf, c=COLOR[s], linewidth=2, alpha=1, zorder=-20, label=labels[method]+' (Empirical)', linestyle='--')

            #ax.set_ylim(bottom=-0.1)
            ax.set_xlabel(f"Accuracy")

            unf_plot_label = {'dp': "Demographic Parity Distance", 'eo': "Equalized Odds Distance", 'eopp': "Equal Opportunity Distance"}[unf_name]
            ax.set_ylabel(unf_plot_label)


            ax.set_ylim(bottom=-0.002)

            ax.legend(fancybox=True, framealpha=0., loc=2)

            ax.grid(alpha=0.4)
            os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "main"))

            tabledata = pd.DataFrame.from_dict(tabledata) 
            if interactive:
                pyplot.show()
            else:
                main_suffix = '_main'
                pyplot.savefig(
                    os.path.join(FIGURES_FOLDER, "main", f"all_methods_{data}_{key}_{unfairness_metric}{main_suffix}.{FORMAT}"),
                    bbox_inches='tight')
                pyplot.close()


# balance
def figure_c(args):

    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    prop_idxs = {
        "noop": 4, "fcrl": 0, "fnf": 3, "sipm": 2, 
        "fair-path": 6, "tree": 7, "laftr": 1,
        "cvib_supervised": 5
    }

    interactive = args.i 
    pareto = args.p 
    
    data = "ACSIncome-CA-2014-balance"
    method_array = ["tree_pc=0.9", "tree_pc=0.8", "tree_pc=0.7", "tree_pc=0.6", "tree_pc=0.5"]
    suffix = ''

    dfs = {} 
    for method in method_array:
        t = numpy.load(f"result/_eval/{data}{suffix}/{method}.npy", allow_pickle=True).item()
        dfs[method] = get_dataframe_from_results(t, has_ub=(method=='tree'),metrics=["acc", "dp"])
        
    keys_downstream = ['nn_1_layer_normalized']
    assert len( keys_downstream ) >  0
    print(keys_downstream)

    for idx, key in enumerate(keys_downstream): 
        unf_metrics = ['dp']
        for unfairness_metric in unf_metrics:
            unf_name = unfairness_metric
            unf_key = f'{key}_{unf_name}'
            unf_ub_key = f'{key}_{unf_name}_ub'

            figure = pyplot.figure(figsize=(10, 10))
            ax = figure.add_subplot(1, 1, 1)

            XL = 0.70
            ax.set_xticks([0.72, 0.75, 0.78, 0.81])
            ax.set_xlim(0.695, 0.815)

            # also filter points under XL 
            for method in method_array:
                n_before = dfs[method].shape[0]
                dfs[method] = dfs[method][dfs[method][f'{key}_acc'] > XL]
                n_after = dfs[method].shape[0]
                if n_after < n_before:
                    print(f'[{method}] {n_before} -> {n_after} pruned because of {XL}')

            for s, method in enumerate(method_array):
                if not 'tree' in method:
                    continue
                tree_has_curr_ub = unf_ub_key in dfs[method]

                if tree_has_curr_ub:
                    tmp = dfs[method][[f'{key}_acc', unf_key, unf_ub_key]].to_numpy()
                    tree_acc, tree_unf, tree_unf_ub = tmp[:,0], tmp[:, 1], tmp[:, 2]
                    tree_acc1 = tree_acc
                    tree_acc2 = tree_acc
                else:
                    tmp = dfs[method][[f'{key}_acc', unf_key]].to_numpy()
                    tree_acc, tree_unf = tmp[:,0], tmp[:, 1]
                    tree_acc1 = tree_acc
                    tree_acc2 = tree_acc

                if tree_has_curr_ub:
                    tree_acc1, tree_unf_ub = get_pareto(tree_acc, tree_unf_ub)
                    ax.plot(tree_acc1, tree_unf_ub, c=COLOR[s], linewidth=2, alpha=1, zorder=-20, label='FARE ' + method[6:] + ' (Upper Bound)', linestyle='solid')

                tree_acc2, tree_unf = get_pareto(tree_acc, tree_unf)
                ax.plot(tree_acc2, tree_unf, c=COLOR[s], linewidth=2, alpha=1, zorder=-20, label='FARE ' + method[6:] + ' (Empirical)', linestyle='--')

            ax.set_ylim(bottom=-0.002)
            ax.set_xlabel(f"Accuracy")

            unf_plot_label = {'dp': "Demographic Parity Distance", 'eo': "Equalized Odds Distance", 'eopp': "Equal Opportunity Distance"}[unf_name]
            ax.set_ylabel(unf_plot_label)

            ax.legend(fancybox=True, framealpha=0., loc=2, prop={'size': 16})

            ax.grid(alpha=0.4)
            os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "main"))
            
            if interactive:
                pyplot.show()
            else:
                pyplot.savefig(
                    os.path.join(FIGURES_FOLDER, "main", f"sensimbalance_{data}_{key}_{unfairness_metric}.{FORMAT}"),
                    bbox_inches='tight')
                pyplot.close()


def figure_ablation(path): # path to "tree.npy"
    # TODO fix to enable table 5 
    import numpy
    a = numpy.load(path, allow_pickle=True).item()

    fair = {'k':3,'ni':1000,'a':0.999,'s':0.5 }
    balanced = {'k':8,'ni':100,'a':0.7,'s':0.3}
    acc = {'k':50,'ni':10,'a':0.3,'s':0.1}
    settings = {'fair': fair, 'acc': acc, 'balanced': balanced}

    import pdb; pdb.set_trace()
    a_s = {'k':{3:{}, 5:{}, 8:{}, 20:{}, 50:{}}, 'ni':{10:{}, 50:{}, 100:{}, 500:{}, 1000:{}}, 'a':{0.3:{}, 0.5:{}, 0.7:{}, 0.85:{}, 0.999:{}}, 's':{0.1:{},0.2:{},0.3:{},0.4:{},0.5:{}}}

    def tostr(d):
        return f"k={d['k']},ni={d['ni']},a={d['a']},s={d['s']}"

    for p in a_s.keys():
        keys = a_s[p].keys()
        for k in keys:
            for k_s in settings.keys():
                setting = settings[k_s].copy()
                setting[p] = k
                s = tostr(setting)
                a_s[p][k][s] = a[s]

    import pdb; pdb.set_trace()
    for p in a_s.keys():
        for k in a_s[p].keys():
            a_s[p][k]['checksums'] = a['checksums']
            numpy.save(f'tree_{p}={k}.npy',a_s[p][k])


def figure_impute(args):

    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    prop_idxs = {
        "noop": 4, "fcrl": 0, "fnf": 3, "sipm": 2, 
        "fair-path": 6, "tree": 7, "laftr": 1,
        "cvib_supervised": 5
    }
    
    labels = {
        '0': 'Impuded 0%',
        '1': 'Impuded 1%',
        '5': 'Impuded 5%',
        '10': 'Impuded 10%',
    }


    interactive = args.i 
    pareto = args.p 
    
    data = "ACSIncome-ALL-2014-L"
    method_array = list(reversed(["tree_i=0.0", "tree_i=0.01", "tree_i=0.05", "tree_i=0.1"]))
    suffix = ''

    dfs = {} 
    for method in method_array:
        t = numpy.load(f"result/_eval/{data}{suffix}/{method}.npy", allow_pickle=True).item()
        dfs[method] = get_dataframe_from_results(t, has_ub=(method=='tree'),metrics=["acc", "dp"])
        
    keys_downstream = ['nn_1_layer_normalized']
    assert len( keys_downstream ) >  0
    print(keys_downstream)

    for idx, key in enumerate(keys_downstream):
        unf_metrics = ['dp']
        for unfairness_metric in unf_metrics:
            unf_name = unfairness_metric
            unf_key = f'{key}_{unf_name}'
            unf_ub_key = f'{key}_{unf_name}_ub'

            figure = pyplot.figure(figsize=(10, 10))
            ax = figure.add_subplot(1, 1, 1)

            XL = 0.70
            ax.set_xticks([0.72, 0.75, 0.78, 0.81])
            ax.set_xlim(0.695, 0.815)

            # also filter points under XL 
            for method in method_array:
                n_before = dfs[method].shape[0]
                dfs[method] = dfs[method][dfs[method][f'{key}_acc'] > XL]
                n_after = dfs[method].shape[0]
                if n_after < n_before:
                    print(f'[{method}] {n_before} -> {n_after} pruned because of {XL}')

            for s, method in enumerate(method_array):
                if not 'tree' in method:
                    continue
                tree_has_curr_ub = unf_ub_key in dfs[method]

                if tree_has_curr_ub:
                    tmp = dfs[method][[f'{key}_acc', unf_key, unf_ub_key]].to_numpy()
                    tree_acc, tree_unf, tree_unf_ub = tmp[:,0], tmp[:, 1], tmp[:, 2]
                    tree_acc1 = tree_acc
                    tree_acc2 = tree_acc
                else:
                    tmp = dfs[method][[f'{key}_acc', unf_key]].to_numpy()
                    tree_acc, tree_unf = tmp[:,0], tmp[:, 1]
                    tree_acc1 = tree_acc
                    tree_acc2 = tree_acc
                
                if method == 'tree_i=0.0':
                    s=7
                else:
                    s = s+1

                spec = f' i{method[6:]} ' if method != 'tree_i=0.0' else ' '

                if tree_has_curr_ub:
                    tree_acc1, tree_unf_ub = get_pareto(tree_acc, tree_unf_ub)
                    ax.plot(tree_acc1, tree_unf_ub, c=COLOR[s], linewidth=2, alpha=1, zorder=-20, label=f'FARE{spec}(Upper Bound)', linestyle='solid')

                tree_acc2, tree_unf = get_pareto(tree_acc, tree_unf)
                ax.plot(tree_acc2, tree_unf, c=COLOR[s], linewidth=2, alpha=1, zorder=-20, label=f'FARE{spec}(Empirical)', linestyle='--')

            ax.set_ylim(bottom=-0.002)
            ax.set_xlabel(f"Accuracy")

            unf_plot_label = {'dp': "Demographic Parity Distance", 'eo': "Equalized Odds Distance", 'eopp': "Equal Opportunity Distance"}[unf_name]
            ax.set_ylabel(unf_plot_label)

            ax.legend(fancybox=True, framealpha=0., loc=2, prop={'size': 16})

            ax.grid(alpha=0.4)
            os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "main"))
            
            if interactive:
                pyplot.show()
            else:
                pyplot.savefig(
                    os.path.join(FIGURES_FOLDER, "main", f"imputation_{data}_{key}_{unfairness_metric}.{FORMAT}"),
                    bbox_inches='tight')
                pyplot.close()



# main plots
def figure_main(args):

    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    prop_idxs = {
        "noop": 4, "fcrl": 0, "fnf": 3, "sipm": 2, 
        "fair-path": 6, "tree": 7, "laftr": 1,
        "cvib_supervised": 5, 'tree_1y' : 7, 'tree_2y' : 6, 
		"noop_1y": 4, "noop_2y": 6, 'kmeans': 4, 
        "tree-eo": 7, "tree-eopp": 7,
        "fnf-eo": 3, "fnf-eopp": 3
    }

    baserates = {
        'health_transfer_PCG_MISCHRT': 78.60,
        'health_transfer_PCG_NEUMENT': 71.43,
        'health_transfer_PCG_ARTHSPIN': 67.89,
        'health_transfer_PCG_METAB3': 65.11,
        'health_transfer_PCG_MSC2a3': 61.93
    }
    dp_noops = {
        'health_transfer_PCG_MISCHRT': -1,
        'health_transfer_PCG_NEUMENT': -1,
        'health_transfer_PCG_ARTHSPIN': -1,
        'health_transfer_PCG_METAB3': -1,
        'health_transfer_PCG_MSC2a3': -1
    }


    transfer_data = {
        'health_transfer_PCG_MISCHRT': {},
        'health_transfer_PCG_NEUMENT': {},
        'health_transfer_PCG_ARTHSPIN': {},
        'health_transfer_PCG_METAB3': {},
        'health_transfer_PCG_MSC2a3': {}
    }
    transfer_threshes = [0.3, 0.2, 0.15, 0.1, 0.05, -0.05, -0.1, -0.15, -0.2, -0.3]
    for k in transfer_data:
        for t in transfer_threshes:
            transfer_data[k][t] = {}

    transfer_methods = ['tree', 'fcrl', 'fnf', 'sipm']


    interactive = args.i 
    pareto = args.p

    rebuttal_cuts = {
        "ACSIncome-CA-2014": [0.65, 0.70, 0.75, 0.80],
        "health_notransfer": [0.74, 0.76, 0.78, 0.80],
        "ACSIncomeMulti-CA-2014": [0.35, 0.40, 0.45, 0.50]
    } # only used to present plots as tables in the rebuttal, don't use generally

    def add_rebuttal_cuts(k, xs, ys, tabledata):
        xs, ys = zip(*sorted(list(zip(xs,ys))))
        for cut in tabledata:
            # look for cut? 
            if xs[0] > cut:
                tabledata[cut][k] = 'it is right'
            elif xs[-1] < cut:
                tabledata[cut][k] = 'it is left'
            else:
                for i in range(len(xs)-1):
                    if xs[i] < cut and xs[i+1] >= cut:
                        unf = ((cut - xs[i]) * ys[i+1] + (xs[i+1] - cut) * ys[i]) /  (xs[i+1] - xs[i])
                        tabledata[cut][k] = unf
                        break
    use_rebuttal_cuts = False



    for data in ["health_notransfer", "health_notransfer-FULL", "health_transfer_PCG_MSC2a3", "health_transfer_PCG_METAB3", 
    "health_transfer_PCG_ARTHSPIN", "health_transfer_PCG_NEUMENT", 
    "health_transfer_PCG_MISCHRT",
    "ACSIncome-CA-2014", "ACSIncome-CA-2014-KMEANS", "ACSIncome-ALL-2014", "health_notransfer", "ACSIncome-CA-2014-FULL", "ACSIncome-ALL-2014-FULL", 
    "health_notransfer-FULL", "ACSIncome-ALL-,2014,2015", "ACSIncome-CA-2014-logreg", "ACSIncome-CA-2014-nn2", 
    "ACSIncome-CA-2014-forest", "ACSIncome-CA-2014-tree", "ACSIncomeMulti-CA-2014"]:
        suffix = ''
        if data[-5:] == '-FULL':
            suffix = '_APP'
            data = data[:-5]
            include_appendix_methods = True # laftr, fairpath, kmeans?
        else:
            include_appendix_methods = False

        if data[-7:] == '-KMEANS':
            suffix = '_KMEANS'
            data = data[:-7]
            kmeans = True 
        else:
            kmeans = False

        #################### edit this to change what's plotting
        if ',' in data:
            method_array = ["noop_2y", "noop_1y", "tree_2y", "tree_1y"]
        elif data == 'ACSIncomeMulti-CA-2014':
            method_array = ["noop", "tree"] # just FARE 
        elif '-logreg' in data or '-nn2' in data or '-forest' in data or '-tree' in data:
            method_array = ["noop", "cvib_supervised", "fcrl", "fnf", "sipm", "tree"]
        elif 'CA' in data:
            method_array = ["noop", "cvib_supervised", "fcrl", "fnf", "sipm", "fair-path", "laftr", "tree"] # kmeans separate
        elif 'ALL' in data:
            method_array = ["noop", "cvib_supervised", "fcrl", "fnf", "sipm", "fair-path", "laftr",  "tree"]
        elif data == 'health_notransfer':
            method_array = ["noop", "cvib_supervised", "fcrl", "fnf", "sipm", "fair-path", "laftr", "tree", "tree-eo", "tree-eopp", "fnf-eo", "fnf-eopp"]
        elif data == 'health_transfer':
            method_array = ["noop", "cvib_supervised", "fcrl", "fnf", "sipm", "fair-path", "laftr",  "tree"]
        else:
            method_array = ["noop", "fcrl", "fnf", "sipm", "fair-path", "tree"] # cvib and laftr dropped
        ####################

        if not include_appendix_methods:
            method_array = [k for k in method_array if k not in ["laftr", "fair-path", "kmeans"]]

        if kmeans:
            method_array = ['noop', 'tree', 'kmeans']

        labels = {
            'cvib_supervised': 'CVIB',
            'fcrl': 'FCRL',
            'fnf': 'FNF',
            'sipm': 'sIPM-LFR',
            'fair-path': 'FairPath',
            'laftr': 'LAFTR',
            'noop': 'Unfair Baseline',
            'noop_1y': 'Unfair Baseline (2014)',
            'noop_2y': 'Unfair Baseline (2014 + 2015)',
            'tree': 'FARE',
            'tree_1y': 'FARE (2014)',
            'tree_2y': 'FARE (2014 + 2015)',
            'kmeans': 'K-Means',
            'tree-eopp': 'FARE-EOpp',
            'tree-eo': 'FARE-EO',
            'fnf-eo': 'FNF-EO',
            'fnf-eopp': 'FNF-EOpp'
        }

        ds_labels = {
             "health_transfer_PCG_GIBLEED": 'Heritage Health (Reduced) => GIBLEED',
             "health_transfer_PCG_INFEC4": 'Heritage Health (Reduced) => INFEC4',
             "health_transfer_PCG_MISCHRT": 'Heritage Health (Reduced) => MISCHRT',
             "health_transfer_PCG_SKNAUT": 'Heritage Health (Reduced) => SKNAUT',
             "health_transfer_PCG_TRAUMA": 'Heritage Health (Reduced) => TRAUMA',

            'health_transfer_PCG_MSC2a3': 'Heritage Health (Reduced) => MSC2a3',
            'health_transfer_PCG_METAB3': 'Heritage Health (Reduced) => METAB3',
            'health_transfer_PCG_ARTHSPIN': 'Heritage Health (Reduced) => ARTHSPIN',
            'health_transfer_PCG_NEUMENT': 'Heritage Health (Reduced) => NEUMENT',
            'health_transfer_PCG_RESPR4': 'Heritage Health (Reduced) => RESPR4',

            'health_transfer': 'Heritage Health (Reduced)',
            'health_notransfer': 'Heritage Health',
            'ACSIncome-CA-2014': 'ACSIncome-CA',
            'ACSIncome-ALL-2014': 'ACSIncome-ALL'
        }

        dfs = {} 

        for method in method_array: 
            npypath = f"result/_eval/{data}/{method}.npy"
            # TODO refactor: we load npys from h_t_a sibling folder
            if 'transfer_' in data and method == 'tree':
                labindices = {'MSC2a3': 1, 'METAB3': 2, 'ARTHSPIN': 3, 'NEUMENT': 4, 'MISCHRT': 6}
                labidx = labindices[data.split('_')[-1]]
                npypath = f"result/_eval/health_transfer_all/tree-transfer-{labidx}.npy"
            t = numpy.load(npypath, allow_pickle=True).item()
            if data in ["ACSIncome-ALL-,2014,2015", "ACSIncome-CA-2014-logreg", "ACSIncome-CA-2014-nn2", "ACSIncome-CA-2014-forest", "ACSIncome-CA-2014-tree"] or "health_transfer_" in data:
                dfs[method] = get_dataframe_from_results(t, has_ub=(method in ['kmeans','tree']),metrics=["acc", "dp"])
            else:
                dfs[method] = get_dataframe_from_results(t, has_ub=(method in ['kmeans','tree']))
        
        keys_downstream = []
        for k in ["nn_1_layer_normalized", "nn_2_layer_normalized", "random_forest", "logistic_regression_normalized", "decision_tree"]:
            keys = dfs[list(dfs.keys())[0]].keys()
            if f'{k}_acc' in keys:
                keys_downstream.append(k)
        assert len( keys_downstream ) >  0
        print(keys_downstream)

        for idx, key in enumerate(keys_downstream):
            unf_metrics = ['dp']
            if data in ['health_notransfer']:
                unf_metrics = ['dp', 'eo', 'eopp'] # enable to plot eo and eopp
            for unfairness_metric in unf_metrics:
                unf_name = unfairness_metric 
                unf_key = f'{key}_{unf_name}'
                unf_ub_key = f'{key}_{unf_name}_ub'

                figure = pyplot.figure(figsize=(10, 10))
                ax = figure.add_subplot(1, 1, 1)

                # limits 
                XL = -1
                if data == "adult":
                    XL = 0.75
                if data == "health_notransfer":
                    XL = 0.72
                    ax.set_xticks([0.74, 0.76, 0.78, 0.80, 0.82])
                    ax.set_yticks([0, 0.10, 0.20, 0.30, 0.40, 0.50])
                if data == 'ACSIncome-CA-2014':
                    XL = 0.635
                if data == 'ACSIncome-ALL-2014':
                    XL = 0.675
                    ax.set_xticks([0.70, 0.73, 0.76, 0.79, 0.82])
                
                # also filter points under XL 
                for method in method_array:
                    n_before = dfs[method].shape[0]
                    dfs[method] = dfs[method][dfs[method][f'{key}_acc'] > XL]
                    n_after = dfs[method].shape[0]
                    if n_after < n_before:
                        print(f'[{method}] {n_before} -> {n_after} pruned because of {XL}')

                # plot all else
                if data in transfer_data:
                    best_at_thresh = {} 
                    for t in transfer_threshes:
                        best_at_thresh[t] = -1

                ####### transfer
                if data in transfer_data:
                    unf_noop = dfs['noop'][[unf_key]].to_numpy().item()
                    if unf_name == 'dp':
                        dp_noops[data] = unf_noop

                    for method in transfer_methods:
                        if method == 'tree':
                            tmp = dfs[method][[f'{key}_acc', unf_key, unf_ub_key]].to_numpy() 
                            ac, unf, proof = tmp[:,0], tmp[:, 1], tmp[:, 2]
                        else:
                            tmp = dfs[method][[f'{key}_acc', unf_key]].to_numpy() 
                            ac, unf = tmp[:,0], tmp[:, 1]

                        for thresh in transfer_threshes:
                            t = thresh if thresh>0 else unf_noop+thresh
                            mask = (unf <= t)
                            acmask = ac[mask]
                            if acmask.size == 0:
                                val = -1
                                if method == 'tree':
                                    val = (-1, -1)
                            else:
                                val = acmask.max().item()
                                if method == 'tree':
                                    val = (val, proof[mask][acmask.argmax()])
                            transfer_data[data][thresh][method] = val 


                MARKER_LW = 2 
                SCATTER_MARKERSIZE = 55

                if use_rebuttal_cuts:
                    tabledata = {}
                    for cut in rebuttal_cuts[data]:
                        tabledata[cut] = {}


                # actual plot
                for i, method in enumerate(method_array):
                    print(f'\n====================== METHOD: {method}\n')
                    print(dfs[method])

                    if 'tree' in method or 'kmeans' in method:
                        continue 

                    if unfairness_metric == 'dp' and method in ['fnf-eo', 'fnf-eopp']:
                        continue
                    if unfairness_metric == 'eo' and method in ['fnf', 'fnf-eopp']:
                        continue
                    if unfairness_metric == 'eopp' and method in ['fnf-eo', 'fnf']:
                        continue

                    dataframe = dfs[method]
                    color_idx = prop_idxs[method]

                    zo = -20 
                    if method == 'noop':
                        zo = 2000
                    c = "none" if method != 'noop' else COLOR[color_idx]

                    tmp = dataframe[[f'{key}_acc', unf_key]].to_numpy() 

                    if pareto:
                        xs, ys = get_pareto(tmp[:,0], tmp[:,1])
                        ax.plot(xs, ys, c=COLOR[color_idx], linewidth=2, alpha=1, zorder=zo)
                    else:
                        xs, ys = tmp[:,0], tmp[:,1]

                    if use_rebuttal_cuts:
                        if method == 'noop':
                            print(f'noop: {xs[0]:.2f} {ys[0]:.2f}')
                        add_rebuttal_cuts(method, xs, ys, tabledata)

                    sc = ax.scatter(x=xs, y=ys, label=labels[method], c=c, marker=MARKER[color_idx], edgecolors=COLOR[color_idx], s=SCATTER_MARKERSIZE, linewidth=MARKER_LW, alpha=1, zorder=zo)

                if XL > 0:
                    ax.set_xlim(left=XL, )
                

                # Plot tree separately
                for method in method_array:
                    if not 'tree' in method and not 'kmeans' in method:
                        continue

                    if unfairness_metric == 'dp' and method in ['tree-eo', 'tree-eopp']:
                        continue
                    if unfairness_metric == 'eo' and method in ['tree', 'tree-eopp']:
                        continue
                    if unfairness_metric == 'eopp' and method in ['tree-eo', 'tree']:
                        continue

                    tree_has_curr_ub = unf_ub_key in dfs[method]
                    color_idx = prop_idxs[method]

                    if tree_has_curr_ub:
                        tmp = dfs[method][[f'{key}_acc', unf_key, unf_ub_key]].to_numpy()
                        tree_acc, tree_unf, tree_unf_ub = tmp[:,0], tmp[:, 1], tmp[:, 2]
                        tree_acc1 = tree_acc
                        tree_acc2 = tree_acc
                    else:
                        tmp = dfs[method][[f'{key}_acc', unf_key]].to_numpy()
                        tree_acc, tree_unf = tmp[:,0], tmp[:, 1]
                        tree_acc1 = tree_acc
                        tree_acc2 = tree_acc

                    if tree_has_curr_ub:
                        if pareto:
                            tree_acc1, tree_unf_ub = get_pareto(tree_acc, tree_unf_ub)
                            ax.plot(tree_acc1, tree_unf_ub, c=COLOR[color_idx], linewidth=2, alpha=1, zorder=-20)

                            if use_rebuttal_cuts:
                                add_rebuttal_cuts(f'{method}-ub', tree_acc1, tree_unf_ub, tabledata)

                        sc1 = ax.scatter(x=tree_acc1, y=tree_unf_ub, label=labels[method] + ' (Upper Bound)', c="none", marker="v", edgecolors=COLOR[color_idx], s=SCATTER_MARKERSIZE, linewidth=MARKER_LW, alpha=1, zorder=-20)
                    if pareto:
                        tree_acc2, tree_unf = get_pareto(tree_acc, tree_unf)
                        ax.plot(tree_acc2, tree_unf, c=COLOR[color_idx], linewidth=2, alpha=1, zorder=-20)

                        if use_rebuttal_cuts:
                            add_rebuttal_cuts(f'{method}', tree_acc2, tree_unf, tabledata)                                

                    sc2 = ax.scatter(x=tree_acc2, y=tree_unf, label=labels[method] + ' (Empirical)', c="none", marker="^", edgecolors=COLOR[color_idx], s=SCATTER_MARKERSIZE, linewidth=MARKER_LW, alpha=1, zorder=-20)
                    if tree_has_curr_ub:
                        if not pareto:
                            ax.vlines(x=tree_acc, ymin=tree_unf+0.0015, ymax=tree_unf_ub-0.0015, color=COLOR[color_idx], linewidth=1.2, linestyle=(0,(2,4)), alpha=0.9, zorder=-20)

                    if pareto and tree_has_curr_ub and data != "ACSIncome-ALL-,2014,2015":
                        shadedxs = []
                        shadedys = []

                        shadedxs.extend(tree_acc1)
                        shadedys.extend(tree_unf_ub)
                        if tree_acc1[-1] < tree_acc2[-1]:
                            shadedxs.append(tree_acc1[-1])
                            shadedys.append(tree_unf[-1])
                        else:
                            shadedxs.append(tree_acc2[-1])
                            shadedys.append(tree_unf_ub[-1])
                        shadedxs.extend(list(reversed(tree_acc2)))
                        shadedys.extend(list(reversed(tree_unf)))
                        if tree_acc1[0] < tree_acc2[0]:
                            shadedxs.append(tree_acc2[0])
                            shadedys.append(tree_unf_ub[0])
                        else:
                            shadedxs.append(tree_acc1[0])
                            shadedys.append(tree_unf[0])
                        shadedxs.append(shadedxs[0])
                        shadedys.append(shadedys[0])
                        ax.fill(shadedxs, shadedys, c=COLOR[color_idx], alpha=0.1)

                if use_rebuttal_cuts:
                    print(f'doing {data} with {unfairness_metric}')
                    for k, dic in tabledata.items():
                        print(f'---------------------- TRESHOLD: {k}')
                        for method, val in dic.items():
                            if type(val) == str:
                                print(f'{method}: {val}')
                            else:
                                print(f'{method}: {val:.2f}')

                ax.set_ylim(bottom=-0.002)
                ax.set_xlabel(f"Accuracy")

                unf_plot_label = {'dp': "Demographic Parity Distance", 'eo': "Equalized Odds Distance", 'eopp': "Equal Opportunity Distance"}[unf_name]
                ax.set_ylabel(unf_plot_label)

                if kmeans:
                    ax.legend(fancybox=True, framealpha=0.) #, prop={'size': 16})
                elif not include_appendix_methods:
                    ax.legend(fancybox=True, framealpha=0., loc=2) #, prop={'size': 16})
                else:
                    ax.legend(fancybox=True, framealpha=0., loc=2, prop={'size': 16})

                ax.grid(alpha=0.4)
                os_utils.safe_makedirs(os.path.join(FIGURES_FOLDER, "main"))
                
                if interactive:
                    pyplot.show()
                else:
                    pyplot.savefig(
                        os.path.join(FIGURES_FOLDER, "main", f"main_{data}_{key}_{unfairness_metric}{suffix}.{FORMAT}"),
                        bbox_inches='tight')
                    pyplot.close()

    # Done 
    if "health_transfer_" in data:
        print('Transfer tables:')
        csv = []
        for k in transfer_data:
            print(f'\n{k:30} (baserate = {baserates[k]:.2f}, dp_noop = {dp_noops[k]:.3f}):')
            csv.append(f'{k},{baserates[k]:.2f},{dp_noops[k]:.3f}')
            
            for thresh in transfer_threshes:
                if thresh < 0: 
                    continue # we didnt present these
                if thresh == -0.05:
                    print()
                s = ''
                cs = ''
                real_t = thresh if thresh > 0 else dp_noops[k] + thresh 
                if real_t < 0:
                    continue
                if thresh > 0:
                    s += f'[@DP={thresh:.2f}]: '
                    cs += f'{thresh:.2f},'
                else:
                    s += f'[@DP=noop{thresh:.3f} ({real_t:.3f})]: '
                    cs += f'{thresh:.3f},{real_t:.3f},'
                for method in transfer_methods:
                    val = transfer_data[k][thresh][method]
                    if method == 'tree':
                        if val[0] == -1:
                            s += f'{method}=///// '
                            cs += f'/,'
                        else:
                            s += f'{method}={val[0]*100:.1f} (DP_UB={val[1]:.3f}) '
                            cs += f'{val[1]:.2f},{val[0]*100:.1f},'
                    else:
                        if val == -1:
                            s += f'{method}=///// '
                            cs += f'/,'
                        else:
                            s += f'{method}={val*100:.1f} '
                            cs += f'{val*100:.1f},'
                print(s)
                csv.append(cs)
        print("\nCSV:")
        for line in csv:
            print(line)
    return # end of main 


if __name__ == "__main__":
    os_utils.safe_makedirs(FIGURES_FOLDER)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action='store_true', help="interactive")
    parser.add_argument("-p", action='store_true', help="pareto")
    args = parser.parse_args()

    # plot all using `-p`

    # Figure 3 (main results, 3 datasets)  + Figure 10 (extended results)
    # + Figure 7 (kmeans in CA)
    # + Figure 8 (ACSIncomeMulti-CA-2014 (with DP and tree vs noop))
    # + Figure 9 (health_notransfer EO (with tree-eo and fnf-eo) and EOPP plots (with tree-eopp and fnf-eopp))
    # + Figure 12 (different downstream classifiers)
    # + Figure 13 (two years from ACS vs one)
    # + Table 1/3 (transfer)
    figure_main(args) 

    # Figure 5 (scaling) -> Ns.pdf
    #figureNs(args)

    # Figure 6 (downstream classifiers) -> diff_models.pdf
    #figure_diff_models(args) 

    # Figure 14 (sensitive attribute imbalance) -> ...-balance_....pdf
    #figure_c(args)

    # Figure 11 right (data imputation)
    #figure_impute(args)

    # Figure 11 left (data shifts)
    #figure_shifts(args)
