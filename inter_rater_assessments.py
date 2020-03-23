import glob     # Import glob to easily loop over files
import pympi    # Import pympi to work with elan files
import string   # Import string to get the punctuation data
from collections import defaultdict 
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt     

import seaborn as sns
from matplotlib import cm
import matplotlib

from matplotlib import collections  as mc

from hmmlearn import hmm
from pathlib import Path
import time

import altair as alt
import datetime as dt
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection

import random
random.seed(9001)

# File interpretation variables
corpus_root = '.'
subfolder = "Annotations"
corpus_root = corpus_root + "/" + subfolder
output_file = '{}/word_frequencies.txt'.format(corpus_root)


# FLAGS
FLAG_ADD_waiter_actions = True
FLAG_ADD_customer_actions = False
FLAG_ALL_FILES = False
FLAG_which_file = 0

FLAG_DISPLAY_agreement_matrices = True
FLAG_DISPLAY_timeline_matches = True


#CONSTANTS
# Note if this is lower there's a chance of multiple matchings
CONST_percent_cutoff = .51


# File location information
# prefixes = ["8-21-18", "8-13-18", "8-9-18"]
prefixes = ["8-17-18"]
# 8-21 is the one with full labels

annotators = ['ada', 'michael', 'jake']
file_set = {}


event_merge_threshold = 1000

blacklist = []
# blacklist.append("arriving")
blacklist.append("leaving")
blacklist.append("arriving")

# blacklist.append("ready-for-food")
# blacklist.append("paying-final-check")
# blacklist.append("reviewing-bill")


valid_labels = []
valid_labels.append("null")

valid_labels.append("eating")
valid_labels.append("drinking")
valid_labels.append("ordering")
valid_labels.append("conversation")
valid_labels.append("standing")
valid_labels.append("idle")

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


customerTierKey = "customerTierKey"
waiterTierKey = "waiterTierKey"

tier_names = {}
tier_names[annotators[0]] = {}
tier_names[annotators[0]][customerTierKey] = ['PersonA', 'PersonB']
tier_names[annotators[0]][waiterTierKey] = ['Waiter', 'CustomerTransitions']

tier_names[annotators[1]] = {}
tier_names[annotators[1]][customerTierKey] = ['PersonA', 'PersonB']
tier_names[annotators[1]][waiterTierKey] = ['Waiter', 'CustomerTransitions']

conversions = {}
conversions["Standing"] = "standing"
conversions["Read"] = "read:menu"
conversions["bored"] = "idle"
conversions["Pay"] = "pay:check"
# conversions[""] = "null"
# ADA errors
conversions["return:check"] = "bring:check"
conversions["take:menu"] = "take:order"
conversions["take:checkin"] = "info"
conversions["take:info"] = "info"
conversions["use:menu"] = "read:menu"

# conversions["use:salt"] = "use:object"

#MICHAEL TO ADA
# conversions["answer-question"] = "info"
# conversions["menu"] = "bring:menu"
# # conversions["conversation"] = "info"
# conversions["bring:silverware"] = "bring:menu"
# conversions["silverware"] = "bring:menu"
# conversions["arrive"] = "arrival"
# conversions["approach"] = "arrival"
# conversions["reading-menu"] = "read:menu"
# conversions["eat"] = "eating"
# conversions["eat:food"] = "eating"
# conversions["bored"] = "idle"

# blacklist = ["", "leaving", "salt", "impatient", "tapping", "gesture", "dance", "move:food", "use:salt", "move:utensil", "move:utensils", "dancing"]
# blacklist.append("share:food")
# blacklist.append("gesture:mouth")
# blacklist.append("down")
# blacklist.append("wave")
# blacklist.append("laughing")
# blacklist.append("silent")
# blacklist.append("paking-up")
# blacklist.append("lean-back")
# blacklist.append("thumbs-up")
# blacklist.append("pickup-salt")
# blacklist.append("wiping")
# blacklist.append("wipe:hands")
# blacklist.append("cash")
# blacklist.append("lean-in")
# blacklist.append("look:watch")
# blacklist.append("move:dishes")

fileset = {}
dataset = {}
label_set = {}
# Look up subsets of title
for prefix in prefixes:
    fileset[prefix] = {}
    dataset[prefix] = {}
    label_set[prefix] = {}
    for annotator in annotators:
        dataset[prefix][annotator] = []
        label_set[prefix][annotator] = {}


#         fn = prefix + annotator + ".eaf"
#         file_recog[fn] = (prefix, annotator)

# overall_annotations = {}


num_tiers = 1

class AnnotationSegment:
    start = None
    end = None
    label = None
    match_index = None
    is_match = None
    color = None
    annotator = None
    tier = None
    prefix = None
    match_label = None

    def __init__(self, s, e, l, an, tier, prefix):
        self.start = s
        self.end = e
        self.label = l
        self.annotator = an
        self.tier = tier
        self.prefix = prefix

    # def __init__(self, s, e, l, mi, im, ci, an, tier):
    #     self.start = s
    #     self.end = e
    #     self.label = l
    #     self.match_index = mi
    #     self.is_match = im
    #     self.color = ci
    #     self.annnotator = am
    #     self.tier = tier

    def __str__(self):
        label_nice = ""

        s = self.getStart()
        e = self.getEnd()
        l = self.getLabel()
    
        s = dt.datetime.fromtimestamp(s / 1000)
        e = dt.datetime.fromtimestamp(e / 1000)

        s = s.strftime('%H:%M:%S')
        e = e.strftime('%H:%M:%S')

        label_nice += "Label: " + l + " by " + self.getAnnotator() + "\n" 
        label_nice += "From: " + str(s) + " to " + str(e)
        # label_nice += "\n color = " + str(ci)

        if self.getMatchIndex() == -1:
            label_nice += "\nNO MATCH FOUND"
        elif not self.getIsMatch():
            label_nice += "\n MISMATCH"
        label_nice += "\n"
        return label_nice[:-1]
    
    def getRawSeg(self):
        return (self.start, self.end, self.label)

    def getDuration(self):
        return self.end - self.start

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end

    def getLabel(self):
        return self.label

    def getMatchIndex(self):
        return self.match_index

    def getMatchLabel(self):
        return self.match_label

    def getIsMatch(self):
        return self.is_match

    def getColor(self):
        return self.color

    def getAnnotator(self):
        return self.annotator

    def getTier(self):
        return self.tier

    def updateWithMatchAndColor(self, mi, im, ml, ci):
        self.match_index = mi
        self.is_match = im
        self.match_label = ml
        self.color = ci
        return self

    def extend_to_cover(self, annotation2):
        self.start = min(self.start, annotation2.getStart())
        self.end = max(self.end, annotation2.getEnd())

        return self


def getKappaScore(agreement_array):

    # find proportionate agreement (p_o)
    # (agreements / total)
    # find how many would randomly agree (p_e)
    # expect them to randomly align 1/n of the time

    # k = (Po – pe) / (1 – pe)
    w, h = agreement_array.shape

    agree = 0
    total = sum(sum(agreement_array))
    for i in range(w):
        agree = agree + agreement_array[i][i]

    print(total)

    p_o = agree / total
    p_e = 1.0 / w

    kappa = (p_o - p_e) / (1.0 - p_e)
    return kappa



def getPrettyPopup(datapoints):
    label_nice = ""

    for datapoint in datapoints:
        s = datapoint.getStart()
        e = datapoint.getEnd()
    
        s = dt.datetime.fromtimestamp(s / 1000)
        e = dt.datetime.fromtimestamp(e / 1000)

        s = s.strftime('%H:%M:%S')
        e = e.strftime('%H:%M:%S')

        l = datapoint.getLabel()

        label_nice += "Label: " + l + " by " + datapoint.getAnnotator() + "\n" 
        label_nice += "From: " + str(s) + " to " + str(e)
        # label_nice += "\n color = " + str(ci)

        mi = datapoint.getMatchIndex()
        im = datapoint.getIsMatch()

        if mi == -1:
            label_nice += "\nNO MATCH FOUND"
        elif not im:
            label_nice += "\n MISMATCH (" + datapoint.getMatchLabel() + " vs " + l + ")" 


        label_nice += "\n"

    return label_nice[:-1]


# Import the annotations from files
for file_path in glob.glob('{}/*.eaf'.format(corpus_root)):
    # Each prefix represents a file that has been annotated by multiple people
    overall_annotations = {}
    file_label = file_path.split('/', 1)[-1]
    file_label = file_label.replace(".eaf", "")
    file_label = file_label.replace(subfolder + "/", "")
    # print(file_label)
    print(file_label)

    file_label = file_label.replace(subfolder + '\\', "")
    prefix, annotator = file_label.split("_")

    if prefix not in prefixes:
        print("File name not recognized, please add to repo")
        print("File found is " + prefix)
        continue
    if not FLAG_ALL_FILES:
        if prefix != prefixes[FLAG_which_file]:
            print("Ignoring this file to only look at the flagged file")
            continue

    # print(prefix)
    # print("Annotator " + annotator)

    eafob = pympi.Elan.Eaf(file_path)
    valid_tier_names = eafob.get_tier_names()

    fileset[prefix][annotator] = eafob

    customer_tiers = []
    waiter_tiers = []
    if FLAG_ADD_waiter_actions:
        waiter_tiers = tier_names[annotator][waiterTierKey]

    if FLAG_ADD_customer_actions:
        customer_tiers = tier_names[annotator][customerTierKey]



    all_tiers = customer_tiers + waiter_tiers

    existing_tiers = []
    for ti in all_tiers:
        if ti in eafob.get_tier_names():
            existing_tiers.append(ti)

    all_tiers = existing_tiers


    num_tiers = len(all_tiers)

    # import and clean all annotations
    for ort_tier in all_tiers:
        for annotation in eafob.get_annotation_data_for_tier(ort_tier):
            start, end, verbs = annotation[0], annotation[1], annotation[2]
            verbs = verbs.lower()
            # print(annotation)
            # print(verbs)
            verbs = verbs.strip()
            verbs = verbs.replace(",", " ")
            verbs = verbs.replace(";", " ")
            verb_list = verbs.split(" ")
            for verb in verb_list:
                blacklisted = False
                if verb in conversions.keys():
                    print("Converted " + verb + " to " + conversions[verb])
                    verb = conversions[verb]
                if verb in blacklist:
                    # print("Found __ in action set ~" + verbs + "~")
                    # print(verb_list)
                    blacklisted = True

                if not blacklisted:
                    noted = (start, end, verb)
                    new_annotation = AnnotationSegment(start, end, verb, annotator, ort_tier, prefix)

                    dataset[prefix][annotator].append(new_annotation)

                    if verb in label_set[prefix][annotator]:
                        label_set[prefix][annotator][verb] += 1
                    else:
                        label_set[prefix][annotator][verb] = 1


    # TODO: remove data cleaning because the data will be CLEAN yay
    dirty_data = dataset[prefix][annotator]
    # clean_data = []
    # clean_data.append(dirty_data[0])
    # for di in range(1, len(dirty_data)):
    #     a1 = dirty_data[di]
    #     a2 = dirty_data[di - 1]

    #     ap = a1.getLabel()
    #     a = a2.getLabel()

    #     # (s, e, a) = dirty_data[di].getRawSeg()
    #     # (sp, ep, ap) = dirty_data[di - 1].getRawSeg()

    #     # merge "arrival" class with the next one
    #     # as well as null labels with the next one

    #     #Merge contiguous events
    #     if a1.getLabel() == a2.getLabel():
    #         if a1.getEnd() - a2.getStart() < event_merge_threshold:
    #             merged = a1.extend_to_cover(a2)
    #             clean_data.append(merged)
    #     else:
    #         clean_data.append(a)


        # # merge 
        # if ap == "arriving":
        #     merged = a2.extend_to_cover(a1)
        #     clean_data.append(merged)
        # elif ap == "null" and s - e < event_merge_threshold:
        #     merged = a2.extend_to_cover(a1)
        #     clean_data.append(merged)
        # elif a != "arriving" and a != "null":
        #     clean_data.append(a1)


    # print("Cleaned data")
    # dataset[prefix][annotator] = clean_data
    print("Skipped cleaning data")
    # print(clean_data)


# Compare annotations
tolerance = 500 #ms
counters = {}

print("Overall labels used")
# print(label_set)

print("Dataset")
# print(dataset)


def getVerticalPos(annotator, tier):

    # There are (tier) x (# annotators)

    num_annotators = len(annotator_indices)
    tier_index = all_tiers.index(tier)

    total_y = tier_index*num_annotators + annotator_indices[annotator]
    
    return total_y

def findDataMatches(event):
    x = event.xdata
    y = event.ydata

    num_annotators = len(annotators)

    y = int(round(y))
    
    all_matches = []
    # print("y = " + str(y))

    tier_index = int(y / num_annotators)
    # print("Tier index: " + str(tier_index))

    annotator_index = y % num_annotators

    if annotator_index > len(annotators) or tier_index > len(all_tiers):
        return None

    if annotator_index < 0 or tier_index < 0:
        return None

    a_click = annotators[annotator_index]
    t_click = all_tiers[tier_index]

    # print("Annotator click -> " + a_click)
    # print("Tier click -> " + t_click)

    for t in timeline_data:
        s = t.getStart()
        e = t.getEnd()
        an = t.getAnnotator()
        tr = t.getTier()

        if an == a_click and tr == t_click:
            if s < x < e:
                all_matches.append(t)


    if len(all_matches) > 0:
        return all_matches

    return None




# ORGANIZATION OF DATA IN BETWEEN REVIEWS

# Dataset organization in between reviews
annotator_indices = {}
for ann_order in range(len(annotators)):
    annotator_indices[annotators[ann_order]] = ann_order



# For each set of matching files
for prefix in prefixes:
    print("REVIEWING DATA FOR " + prefix)
    counters[prefix] = 0
    compare_set = dataset[prefix]
    # print("Compare set")
    # print(compare_set)
    # TODO allow it to compare multiple annotators

    agreement_table = {}
    total_labels = []

    annotator_one = annotators[0]
    annotator_two = annotators[1]

    if len(compare_set) > 1 and len(compare_set[annotators[0]]) > 0:
        set1 = compare_set[annotator_one]
        set2 = compare_set[annotator_two]

        for annotator in annotators:
            print(label_set[prefix][annotator].keys())
            total_labels.extend(list(label_set[prefix][annotator].keys()))

        print("All labels are:")
        print(total_labels)
        total_labels = list(set().union(total_labels))

        omission_label = "_OMISSION"
        total_labels.append(omission_label)
        print(total_labels)

        for ti in total_labels:
            for tj in total_labels:
                agreement_table[(ti, tj)] = 0

        seg_checklist_1 = {}
        seg_checklist_2 = {}
        matches_log = []

        match_counter = 0

        for seg1 in set1:
            for seg2 in set2:
                seg1_start, seg1_end, seg1_label = seg1.getStart(), seg1.getEnd(), seg1.getLabel()
                seg2_start, seg2_end, seg2_label = seg2.getStart(), seg2.getEnd(), seg2.getLabel()


                same_tier = (seg1.getTier() == seg2.getTier())

                if same_tier and not (seg1_start >= seg2_end + (tolerance) or
                        seg2_start >= seg1_end + (tolerance)):
                    #then overlap

                    # Find percent overlap relative to the longer annotation
                    percent_overlap_top = min(seg1_end, seg2_end) - max(seg1_start, seg2_start)
                    percent_overlap_bottom = max((seg1_end - seg1_start), (seg2_end - seg2_start))

                    percent_overlap = float(percent_overlap_top) / percent_overlap_bottom
                    # print("percent_overlap")

                    print("Checking overlap for match " + str((seg1_label, seg2_label)) + " : " + str(percent_overlap))
                    # if this is our match
                    if percent_overlap > CONST_percent_cutoff:
                        is_match = (seg1_label == seg2_label)

                        # print("FOUND MATCH")
                        seg_checklist_1[seg1] = (match_counter, is_match, seg2_label)
                        seg_checklist_2[seg2] = (match_counter, is_match, seg1_label)

                        matches_log.append((annotator_one, seg1, annotator_two, seg2))
                        match_counter = match_counter + 1

                        agreement_table[(seg1_label, seg2_label)] += 1

                        if not is_match:                        
                            print("MISMATCHED LABELS")


        # print(agreement_table)
        # Catch the leftovers
        # print("Checklist")
        # print(seg_checklist_1)
        for seg1 in set1:
            if seg1 not in seg_checklist_1.keys():
                label1 = seg1.getLabel()
                print("No match for " + str(seg1) + " by ada")
                agreement_table[(label1, omission_label)] += 1
                # print("missing")
                # print((label1, omission_label))
                

        for seg2 in set2:
            if seg2 not in seg_checklist_2.keys():
                label2 = seg2.getLabel()
                print("No match for " + str(seg2) + " by michael")
                agreement_table[(omission_label, label2)] += 1
        
        agreement_array = np.array([[agreement_table[(label1,label2)] for label1 in total_labels] for label2 in total_labels])

        print("Kappa score is " + str(getKappaScore(agreement_array)))

        # print(agreement_array)

        from heapq import nlargest 
        top_values = nlargest(30, agreement_table, key = agreement_table.get) 
  
        print("Dictionary with 10 highest values:") 
        print("Keys: Values") 
          
        for val in top_values: 
            print(val, ":", agreement_table.get(val)) 



        if FLAG_DISPLAY_agreement_matrices:
            ax= plt.subplot()
            sns.heatmap(agreement_array, annot=True, ax = ax); #annot=True to annotate cells

            # labels, title and ticks
            ax.set_ylabel('Labeller 2: ' + annotator_two);
            ax.set_xlabel('Labeller 1: ' + annotator_one);

            graph_title = 'Agreement Matrix for ' + prefix
            graph_title += "\n"
            graph_title += "Kappa Score of " + str(getKappaScore(agreement_array))

            ax.set_title(graph_title); 
            ax.xaxis.set_ticklabels(total_labels, rotation=90)
            ax.yaxis.set_ticklabels(total_labels, rotation=0)
            plt.tight_layout()
            plt.show()

        # Within a given prefix
        if FLAG_DISPLAY_timeline_matches:
            # SETUP INFO
            bar_width = .8
            half_bar = bar_width / 2.0
            verts = []
            colors = []
            mi_vals = []
            
            # Make consistent color scheme
            num_labels = len(total_labels)
            label_order = total_labels.copy()
            random.shuffle(label_order)
            colorset = cm.get_cmap('viridis', num_labels)
            

            # LINE CONSTANTS
            LINECOLOR_match = 'black'
            LINECOLOR_mismatch = 'red'
            BOXCOLOR_UNMATCHED = 'red'


            # data = pd.DataFrame()
            timeline_data = []
            t_starts = []
            t_ends = []
            t_labels = []
            t_match_ids = []


            timeline_num_colors = 2

            for annotator in annotators:
                annotator_log = dataset[prefix][annotator]
                for timechunk in annotator_log:
                    # t_starts.append(s)
                    # t_ends.append(e)
                    # t_labels.append(a)

                    # Unmatched segments will have a value of -1
                    
                    # Does a match exist = mi
                    # Is it a correct match = im
                    mi =  -1
                    im = False
                    ml = ""
                    if timechunk in seg_checklist_1:
                        mi, im, ml = seg_checklist_1[timechunk]
                    if timechunk in seg_checklist_2:
                        mi, im, ml = seg_checklist_2[timechunk]

                    if mi != -1:
                        color_index = label_order.index(timechunk.getLabel())
                        # ci = 'blue'
                        ci = colorset(color_index)
                    else:
                        ci = BOXCOLOR_UNMATCHED

                    # t_match_ids.append(mi)


                    new_data = timechunk.updateWithMatchAndColor(mi, im, ml, ci)
                    timeline_data.append(new_data)

            print("Alignment graphing")
            
           
            # DRAW RECTANGLES FOR EACH TIME SEGMENT
            for d in timeline_data:

                tier = None
                y_offset = getVerticalPos(d.getAnnotator(), d.getTier())

                s = d.getStart()
                e = d.getEnd()

                ci = d.getColor()
                mi = d.getMatchIndex()

                v =  [(s, y_offset - half_bar),
                      (s, y_offset + half_bar),
                      (e, y_offset + half_bar),
                      (e, y_offset - half_bar),
                      (s, y_offset - half_bar)]

                verts.append(v)
                colors.append(ci)
                mi_vals.append(mi)


            # draw them in an order that will highlight the problem annotations
            verts = [x for _,x in sorted(zip(mi_vals,verts), reverse=True)]
            colors = [x for _,x in sorted(zip(mi_vals,colors), reverse=True)]


            # For each match that exists:
            lines = []
            line_colors = []
            all_matches_for_lines = seg_checklist_1.keys()

            for a1, val1, a2, val2 in matches_log:
                s1, e1, l1 = val1.getStart(), val1.getEnd(), val1.getLabel()
                s2, e2, l2 = val2.getStart(), val2.getEnd(), val2.getLabel()


                x1 = ((e1 + s1) / 2)
                x2 = ((e2 + s2) / 2)
                y1 = getVerticalPos(a1, val1.getTier())
                # annotator_indices[a1]
                y2 = getVerticalPos(a2, val2.getTier())
                # annotator_indices[a2]

                if y1 > y2:
                    y1 += -1 * half_bar
                    y2 += half_bar
                else:
                    y1 += half_bar
                    y2 += -1 * half_bar

                line = [(x1, y1), (x2, y2)]
                
                # If a correct match, use black
                c = LINECOLOR_match

                # if mismatch, red
                if l1 != l2:
                    c = LINECOLOR_mismatch

                # make sure if a mismatch was found, 
                # it is always painted last/on top
                if line in lines:
                    prev_index = lines.index(line)
                    prev_color = colors[prev_index]

                    if prev_color == LINECOLOR_mismatch:
                        c = LINECOLOR_mismatch


                lines.append(line)
                line_colors.append(c)


            bars = PolyCollection(verts, facecolors=colors)
            fig, ax = plt.subplots()
            ax.add_collection(bars)

            lc = mc.LineCollection(lines, colors=line_colors, linewidths=1)
            ax.add_collection(lc)

            ax.autoscale()


            # loc = mdates.MinuteLocator(byminute=[0,10,20,30,40,50,60,70,80])
            # ax.xaxis.set_major_locator(loc)
            # ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))

            # majorFormatter = dt.DateFormatter('%H:%M:%S')
            # ax.xaxis.set_major_formatter(majorFormatter)

            annotation_labels = []
            for tier in all_tiers:   
                for a in annotators:
                    new_label = a + ":\n" + tier
                    annotation_labels.append(new_label)


            ax.set_yticks(range(len(annotation_labels)))
            ax.set_yticklabels(annotation_labels)




            annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                bbox=dict(boxstyle="round", fc="white", ec="b", lw=2),
                arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)
            
            def update_annot(datapoints, event):
                #todo: verify this is a good rep
                d = datapoints[0]
                s = d.getStart()
                e = d.getEnd()
                an = d.getAnnotator()

                x = int((s + e) / 2.0)
                y = getVerticalPos(an, d.getTier())

                # y = annotator_indices[an] - .5*half_bar
                annot.xy = (x,y)
                text = getPrettyPopup(datapoints)

                annot.set_text(text)
                annot.set_visible(True)
                annot.get_bbox_patch().set_alpha(.8)


            def hover(event):
                    
                # print('%s hover: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % ('double' if event.dblclick else 'single', event.button, event.x, event.y, event.xdata, event.ydata))

                if event.xdata != None and event.inaxes == ax:
                    matching_event_data = findDataMatches(event)
                    if matching_event_data != None:
                        update_annot(matching_event_data, event)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        
                else:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", hover)

            ax.set_ylabel('Labellers');
            ax.set_xlabel('Time');


            graph_title = 'Annotations and matches over time for ' + prefix
            graph_title += "\n"
            graph_title += "Kappa Score of " + str(getKappaScore(agreement_array))

            ax.set_title(graph_title) 


            plt.show()



            print("Done with chart")








