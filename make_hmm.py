
import glob     # Import glob to easily loop over files
import pympi    # Import pympi to work with elan files
import string   # Import string to get the punctuation data
from collections import defaultdict 
import numpy as np


import seaborn as sns
from matplotlib import cm

from hmmlearn import hmm

np.random.seed(56)

stepsize = 250
secondsize = 1000
num_components = 7
colors_pretty = ['#006cff', '#3689fc', '#51a4f7', '#6bbfef', '#89d9e4', '#b2f0d1', '#ffffaa']
colors_viridis = cm.get_cmap('viridis', num_components)
# colors_viridis = cm.get_cmap('spring', num_components)

filename = "viridis.png"


# Define some variables for later use
corpus_root = '.'
output_file = '{}/word_frequencies.txt'.format(corpus_root)
ort_tier_names = ['Person_A', 'Person_B']
staff_names = ['W_1', 'W_2']
all_tiers = staff_names + ort_tier_names

waiter_visits = []


timestamp_first = -1
timestamp_last = -1

valid_labels = []
valid_labels.append("null")

valid_labels.append("eating")
valid_labels.append("drinking")
valid_labels.append("ordering")
valid_labels.append("conversation")
valid_labels.append("standing")
valid_labels.append("bored")

valid_labels.append("away-from-table")

valid_labels.append("read:menu")
valid_labels.append("look:window")
valid_labels.append("look:waiter")
valid_labels.append("pay:check")
valid_labels.append("use:phone")
valid_labels.append("use:napkin")
valid_labels.append("use:wallet")
valid_labels.append("use:purse")
valid_labels.append("use:glasses")

valid_labels.append("conversation-and-eating")

waiter_labels_valid = []
waiter_labels_valid.append('bring:to-seat')
waiter_labels_valid.append('bring:menus')
# waiter_labels_valid.append('bring:silverware')
waiter_labels_valid.append('take:order')
waiter_labels_valid.append('bring:drinks')
waiter_labels_valid.append('bring:food')
waiter_labels_valid.append('take:dishes')
waiter_labels_valid.append('bring:check')
waiter_labels_valid.append('take:check')

 # 'bring:menus': 1, 'bring:silverware': 1,
 # 'answer-question': 1, 'arrive': 2, 
 # 'take:order': 2, '': 1, 'bring:drinks': 1, 
 # 'leaving': 1, 'take:checkin': 4, 'take:dishes': 1, 
 # 'arriving': 1, 'bring:check': 1, 'approach': 1, 'take:check': 2, 'bring:food': 2}

def getTimelineAnnotation(personLabel, stepsize, timestamp_first, timestamp_last):
    nSteps = int((timestamp_last - timestamp_first) / stepsize) + 1
    nLabels = len(valid_labels)
    personData = np.zeros((nSteps, 1), dtype=np.int8)

    for second in range(timestamp_first, timestamp_last, stepsize):

        annotations_here = eafob.get_annotation_data_at_time(personLabel, second)
        
        for annotation in annotations_here:
                if annotation is []:
                    annotation = [second, second, "null"]

                start, end, utterance = annotation[0], annotation[1], annotation[2]
                duration = end - start

                stepIndex = int((second - timestamp_first) / stepsize)
                # print(stepIndex)

                # Split, by default, splits on whitespace thus separating words
                words = utterance.split()

                # TODO verify that 
                for word in words:
                    if "conversation" in words and "eating" in words:
                        hmm_state = "conversation-and-eating"
                    elif word in valid_labels:
                        tagIndex = valid_labels.index(word)
                        personData[stepIndex] = int(tagIndex)
                        # print("labeled data")



                # if "conversation" in words and "eating" in words:
                #     hmm_state = "conversation-and-eating"
                # else:
                #     for word in words:
                #         if word in valid_labels:
                #             hmm_state = word

                # (n_samples, n_features)

    return personData




# Loop over all elan files the corpusroot subdirectory called eaf
for file_path in glob.glob('{}/*.eaf'.format(corpus_root)):
    # Initialize the elan file
    eafob = pympi.Elan.Eaf(file_path)
    print("Opening " + file_path)

    # Look at waiter-specific tasks
    for ort_tier in all_tiers:
        for annotation in eafob.get_annotation_data_for_tier(ort_tier):
            start, end = annotation[0], annotation[1]

            if timestamp_first is -1 or start < timestamp_first:
                timestamp_first = start
            if timestamp_last is -1 or end > timestamp_last:
                timestamp_last = start


    waiter_actions = {}
    for ort_tier in staff_names:
        for annotation in eafob.get_annotation_data_for_tier(ort_tier):
            start, end = annotation[0], annotation[1]
            start = int(start / stepsize)
            end = int(end / stepsize)

            for action in annotation[2].split(' '):
                if action in waiter_actions.keys():
                    waiter_actions[action] = waiter_actions[action] + 1
                else:
                    waiter_actions[action] = 1

                if action in waiter_labels_valid:
                    waiter_visits.append((start, end, action))
            # print("waiter visit at " + str(start))

    # print(waiter_actions)

    # for every second in the time range (timestamp_first to timestamp_last)
    # get_annotation_data_at_time(id_tier, time)
    # or
    # get_annotation_data_between_times(id_tier, start, end) 
    # add that to the array of observations

    label_person_A, label_person_B = ort_tier_names

    listA = []
    listB = []
    
    listA = getTimelineAnnotation(label_person_A, stepsize, timestamp_first, timestamp_last)
    listB = getTimelineAnnotation(label_person_B, stepsize, timestamp_first, timestamp_last)

    # # Loop over all the defined tiers that contain orthography
    # for ort_tier in ort_tier_names:
    #     # If the tier is not present in the elan file spew an error and
    #     # continue. This is done to avoid possible KeyErrors
    #     if ort_tier not in eafob.get_tier_names():
    #         print('WARNING!!!')
    #         print('One of the ortography tiers is not present in the elan file')
    #         print('namely: {}. skipping this one...'.format(ort_tier))
    #     # If the tier is present we can loop through the annotation data
    #     else:
    #         for annotation in eafob.get_annotation_data_for_tier(ort_tier):
    #             start, end, utterance = annotation[0], annotation[1], annotation[2]
    #             duration = end - start

    #             # Split, by default, splits on whitespace thus separating words
    #             words = utterance.split()

def getPrettyTransitionNames(trans_array):
    # majority_labels = []
    # for getPrettyTransitionNames
    return trans_arrays

def getPrettyEmissionNames(emission_array):
    print("Get Pretty Emission Names")
    majority_labels = []
    # print(len(valid_labels))
    # print(len(emission_array[0]))
    # print(len(emission_array))

    em_helper = {}

    for row in emission_array:
        em_helper = {}

        best_value = -1
        best_index = -1
        # TODO fix
        for c in range(len(valid_labels)):
            # em_helper[valid_labels[c]] = row[c]

            if row[c] > best_value:
                best_value = row[c]
                best_index = c
                # print(best_value)

        # bestVals = list(sorted(em_helper.values()))[-2]
        # print(bestVals)
        # overallLabel = bestVals[0] + " and \n" + bestVals[1]
        # majority_labels.append()

        # print(best_index)
        majority_labels.append(valid_labels[best_index])
        # majority_labels.append(valid_labels[max_index])
    return majority_labels



label_person_A, label_person_B = ort_tier_names

listA = []
listB = []

listA = getTimelineAnnotation(label_person_A, stepsize, timestamp_first, timestamp_last)
listB = getTimelineAnnotation(label_person_B, stepsize, timestamp_first, timestamp_last)

# X = sequence from actors
# X[actor] = sequence for each

# X = listA + listB
# lengths_list = [len(listA), len(listB)]


X = listA + listB
lengths_list = [len(listA), len(listB)]

# make a new HMM
hmm_all = hmm.MultinomialHMM(n_components=num_components)
model_all = hmm_all.fit(X)
prediction_all = hmm_all.predict(X)
decode_all = hmm_all.decode(X)

hmm_a = hmm.MultinomialHMM(n_components=num_components)
model_a = hmm_a.fit(listA)
prediction_a = hmm_a.predict(listA)

hmm_b = hmm.MultinomialHMM(n_components=num_components)
model_b = hmm_b.fit(X)
prediction_b = hmm_b.predict(X)

customers_same = []
all_same = []
overall_graph = []
decode_predict = []

print("Overall HMM")
print(hmm_a)

print("Transition probabilities")
print(hmm_all.transmat_)

print("Emission of feature probabilities")
print(hmm_all.emissionprob_)

human_labels = getPrettyEmissionNames(hmm_all.emissionprob_)
print(human_labels)

decode_all = decode_all[1]
print(decode_all)


for pred_a, pred_b, pred_all, decode_a in zip(prediction_a, prediction_b, prediction_all, decode_all):
    both_same = (pred_a == pred_b)
    customers_same.append(both_same)

    overall_graph.append(abs(pred_a - pred_b))

    pred_all = (pred_a == pred_b == pred_all)
    all_same.append(pred_all)

    decode_predict.append(decode_a == pred_a)



from matplotlib import pyplot as plt
# plt.plot(prediction_all)


s = prediction_a
# print(prediction_a)
step_to_ms = float(stepsize / secondsize)
s_range = (len(s) * stepsize) / secondsize
# s_range = len(s)

# print(step_to_ms)
# print(s_range)

t = np.arange(0, s_range, step_to_ms)
# for i in len(t):
#     t[i] = float(t[i] / (stepsize * secondsize))



# print(len(t))
# print(len(prediction_a))

import matplotlib

colors = ['red','yellow','green','blue','purple']

hl = human_labels
# hl = ['look:waiter', 'read:menu', 'conversation', 'eating', 
# 'conversation-and-eating', 'bored', 'look:window']
# hl = hl[::-1]

y = np.zeros(len(t))

fig, ax = plt.subplots()
plt.scatter(t, prediction_a, c=prediction_a, cmap=colors_viridis)
plt.yticks(np.arange(num_components), hl)
ax.set_xlabel('Time (s)')
ax.set_ylabel('HMM State by majority observation')
ax.set_title('HMM State Prediction for Customer A')

for wv in waiter_visits:
    (ws,we, wlabel) = wv
    ws = ws * step_to_ms
    # print(ws)
    plt.axvline(ws, c='black', linewidth='.5', label=wlabel)
    ax.text(x=ws, y=(-1.5), s=wlabel, alpha=0.7, color='#334f8d', rotation=90)

# plt.legend()
plt.tight_layout()
plt.savefig("pair2_hmm_" + filename)
# plt.show()

# plt.plot(prediction_a, 'bo')


# plt.plot(prediction_b)
# plt.show()

# plt.title("Do the customers have the same underlying state?")
# plt.plot(customers_same, 'bo')
# plt.show()

# plt.title("Do all three of the predictions (a, b and table) agree with eachother?")
# plt.plot(all_same, 'bs')
# plt.show()

# plt.title("How does the decode differ from the predict?")
# plt.plot(decode_predict)
# plt.show()


model = hmm_a
hidden_states = prediction_a

# print("Means and vars of each hidden state")
# for i in range(model.n_components):
#     print("{0}th hidden state".format(i))
#     print("mean = ", model.means_[i])
#     print("var = ", np.diag(model.covariances_[i]))
#     print()


# Create a new annotation layer
# add_tier(tier_id, ling='default-lt', parent=None, locale=None, part=None, ann=None, language=None, tier_dict=None)
# At each timestep, generate the predicted hidden state
# Consolidate those tags into blocks of state
# for each block, add that annotation to the tier
# 

pred_tier = "prediction"
eafob.add_tier(pred_tier)

nSteps = int((timestamp_last - timestamp_first) / stepsize) + 1

for slot in range(nSteps):
    start = timestamp_first + stepsize*slot
    end = timestamp_first + stepsize*slot + stepsize
    classIndex = prediction_a[slot]
    classMajorityLabel = human_labels[classIndex]

    # eafob.add_annotation(pred_tier, start, end, value=str(classIndex + " : " + classMajorityLabel))

# eafob.to_file("with_predictions.eaf")


# import pygraphviz as pgv
# G=pgv.AGraph()

# for node_name in human_labels:
#     G.add_node(node_name)

# for a in human_labels:
#     for b in human_labels:
#         G.add_edge(a, b)

# G.draw('output_graph')
# print("Output graph")

import pydot
from pydot import Dot, Edge,Node
g = Dot(human_labels[0])
g.set_node_defaults(color='lightgray',
                    style='filled',
                    shape='box',
                    fontname='Courier',
                    fontsize='10',
                    fontcolor='white')
    

transmats = hmm_all.transmat_

num_labels = range(len(human_labels))
# print("Num labels")
# print(num_labels)
# cm.rainbow(np.linspace(0, 1, num_components))

# human labels are the majority labels for things
hl = human_labels
hl = [s.replace(':', '-') for s in hl]
hl = [s.replace('null', 'no-observation') for s in hl]


print(hl)

for i in num_labels:
    new_node = Node(str(hl[i]))
    col = colors_viridis(i)
    col = matplotlib.colors.rgb2hex(col)
    # print(matplotlib.colors.rgb2hex(col))
    new_node.set_color(col)
    g.add_node(new_node)
    print("Add node " + str(hl[i]))

for i_a in num_labels:
    best = sorted(zip(transmats[i_a], hl), reverse=True)[:3]

    for j_b in num_labels:
        a = hl[i_a]
        b = hl[j_b]
        edge = pydot.Edge(str(a), str(b))
        # edge.set_color('black')

        print("edge from " + a + " to " + b)
        prob_label = transmats[i_a][j_b]
        if (prob_label,b) in best:
            # print("found a best")

            prob_label = float('%.3g' % prob_label)
            prob_label = str(prob_label)
            # print(prob_label)
            edge.set_label(prob_label)
            g.add_edge(edge)

# G.draw('output_graph')
print("Output graph")

g.write_png("pair2_" + filename)



# sns.set(font_scale=1.25)
# style_kwds = {'xtick.major.size': 3, 'ytick.major.size': 3,
#               'font.family':u'courier prime code', 'legend.frameon': True}
# sns.set_style('white', style_kwds)

# fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True, figsize=(12,9))
# colors = cm.rainbow(np.linspace(0, 1, model.n_components))

# for i, (ax, color) in enumerate(zip(axs, colors)):
#     # Use fancy indexing to plot data in each state.
#     mask = hidden_states == i
#     ax.plot_date(select.index.values[mask],
#                  select[col].values[mask],
#                  ".-", c=color)
#     ax.set_title("{0}th hidden state".format(i), fontsize=16, fontweight='demi')

#     # Format the ticks.
#     ax.xaxis.set_major_locator(YearLocator())
#     ax.xaxis.set_minor_locator(MonthLocator())
#     sns.despine(offset=10)

# plt.tight_layout()







# print(overall_HMM)

# predictA = overall_HMM.predict(X)


# predictB = overall_HMM.predict(listB)

# fit(X, lengths=None)
# lengths = how long each experimental input sequence is


# decode(X, lengths=None, algorithm=None)

# Create a new annotation layer
# add_tier(tier_id, ling='default-lt', parent=None, locale=None, part=None, ann=None, language=None, tier_dict=None)
# At each timestep, generate the predicted hidden state
# Consolidate those tags into blocks of state
# for each block, add that annotation to the tier
# 

# Open an output file to write the data to
# with open(output_file, 'w') as output_file:
    # Loop throught the words with their frequencies, we do this sorted because
    # the file will then be more easily searchable
    # for word, frequency in sorted(frequency_dict.items()):
    #     # We write the output separated by tabs
    #     output_file.write('{}\t{}\n'.format(word, frequency))
    # pass


# import pickle
# with open("filename.pkl", "wb") as file: pickle.dump(remodel, file)
# with open("filename.pkl", "rb") as file: pickle.load(file)
