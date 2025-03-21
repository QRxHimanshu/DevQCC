import argparse, pickle, itertools
import numpy as np
import os
from helper_functions.non_ibmq_functions import find_process_jobs, scrambled


def merge_prob_vector(unmerged_prob_vector, qubit_states):
    num_active = qubit_states.count("active")
    num_merged = qubit_states.count("merged")
    merged_prob_vector = np.zeros(2**num_active, dtype="float32")
    # print('merging with qubit states {}. {:d}-->{:d}'.format(
    #     qubit_states,
    #     len(unmerged_prob_vector),len(merged_prob_vector)))
    for active_qubit_states in itertools.product(["0", "1"], repeat=num_active):
        if len(active_qubit_states) > 0:
            merged_bin_id = int("".join(active_qubit_states), 2)
        else:
            merged_bin_id = 0
        for merged_qubit_states in itertools.product(["0", "1"], repeat=num_merged):
            active_ptr = 0
            merged_ptr = 0
            binary_state_id = ""
            for qubit_state in qubit_states:
                if qubit_state == "active":
                    binary_state_id += active_qubit_states[active_ptr]
                    active_ptr += 1
                elif qubit_state == "merged":
                    binary_state_id += merged_qubit_states[merged_ptr]
                    merged_ptr += 1
                else:
                    binary_state_id += "%s" % qubit_state
            # Debugging statements
            # print(f"Debug: qubit_states={qubit_states}")
            # print(f"Debug: active_qubit_states={active_qubit_states}, merged_qubit_states={merged_qubit_states}")
            # print(f"Debug: binary_state_id={binary_state_id}")
            if binary_state_id == "":
                # print(f"Debug: Empty binary_state_id for qubit_states: {qubit_states}, active_qubit_states: {active_qubit_states}, merged_qubit_states: {merged_qubit_states}")
                continue  # Skip this iteration
            state_id = int(binary_state_id, 2)
            merged_prob_vector[merged_bin_id] += unmerged_prob_vector[state_id]
    return merged_prob_vector


# def dump_frozen_status(frozen_status, data_folder, rank, subcircuit_idx):
#     frozen_file_path = "%s/rank_%d_subcircuit_%d_frozen_status.pckl" % (data_folder, rank, subcircuit_idx)
#     pickle.dump(frozen_status, open(frozen_file_path, "wb"))

#     # For debugging purposes: read and print after saving
#     frozen_status_read = pickle.load(open(frozen_file_path, "rb"))
#     print("Saved and read frozen status:", frozen_status_read)


def save_frozen_status_for_subcircuit(data_folder, rank, subcircuit_idx, frozen_status):
    """
    Save the frozen status for a particular subcircuit and rank
    """
    file_path = "%s/rank_%d_subcircuit_%d_frozen_status.pckl" % (data_folder, rank, subcircuit_idx)
    pickle.dump(frozen_status, open(file_path, "wb"))


def load_frozen_status_for_subcircuit(data_folder, rank, subcircuit_idx):
    """
    Load the frozen status for a particular subcircuit and rank
    """
    file_path = "%s/rank_%d_subcircuit_%d_frozen_status.pckl" % (data_folder, rank, subcircuit_idx)
    if os.path.exists(file_path):
        frozen_status = pickle.load(open(file_path, "rb"))
        return frozen_status
    else:
        return {}


    # frozen_status = pickle.load(open(frozen_file_path, "rb"))
    # print("read F_stat: ",frozen_status)

def save_merged_probs_for_subcircuit(data_folder, rank, subcircuit_idx, merged_probs):
    file_path = "%s/rank_%s_subcircuit_%s_merged_probs.pckl" % (data_folder, rank, subcircuit_idx)

    pickle.dump(
        merged_probs,
        open(file_path, "wb"),
    )

def load_previous_merged_probs_for_subcircuit(data_folder, rank, subcircuit_idx):
    file_path = "%s/rank_%s_subcircuit_%s_merged_probs.pckl" % (data_folder, rank, subcircuit_idx)
    # file_path = f"{data_folder}/rank_{rank}_subcircuit_{subcircuit_idx}_merged_probs.pckl"
    # print("FILE: ",file_path)

    if os.path.exists(file_path):
        #  print("loaded: ",file_path)
        merged_probs = pickle.load(open(file_path,"rb",))
        return merged_probs
    else:
        return {}
def is_close_enough(vector1, vector2, tolerance=1e-6):
    # print("checking closeness!!!",np.allclose(vector1, vector2, atol=tolerance))
    return np.allclose(vector1, vector2, atol=tolerance)
# if __name__ == "__main__":
#     """
#     The first merge of subcircuit probs using the target number of bins
#     Saves the overhead of writing many states in the first SM recursion
#     """
#     parser = argparse.ArgumentParser(description="Merge probs rank.")
#     parser.add_argument("--data_folder", metavar="S", type=str)
#     parser.add_argument("--rank", metavar="N", type=int)
#     parser.add_argument("--num_workers", metavar="N", type=int)
#     args = parser.parse_args()
#     fold_path = "devqcc/prob_val"
    
#     meta_info = pickle.load(open("%s/meta_info.pckl" % (args.data_folder), "rb"))
#     dd_schedule = pickle.load(open("%s/dd_schedule.pckl" % (args.data_folder), "rb"))

    
#     merged_subcircuit_entry_probs = {}

#     # print("rank: ", args.rank)
#     for subcircuit_idx in meta_info["entry_init_meas_ids"]:
#         frozen_status = load_frozen_status_for_subcircuit(fold_path,args.rank,subcircuit_idx)
#         previous_merged_probs = load_previous_merged_probs_for_subcircuit(
#             data_folder=fold_path, rank=args.rank, subcircuit_idx=subcircuit_idx
#         )
#         # print(previous_merged_probs)
#         # print("#######keyssss########", previous_merged_probs.keys())
#         rank_jobs = find_process_jobs(
#             jobs=list(meta_info["entry_init_meas_ids"][subcircuit_idx].keys()),
#             rank=args.rank,
#             num_workers=args.num_workers,
#         )
#         merged_subcircuit_entry_probs[subcircuit_idx] = {}

#         # Check if qubit_states is empty for the subcircuit
#         qubit_states = dd_schedule["subcircuit_state"][subcircuit_idx]
#         # if not qubit_states:
#         #     print(f"Warning: Subcircuit {subcircuit_idx} has empty qubit_states")

#         for subcircuit_entry_init_meas in rank_jobs:
#             # print("subcircuit idx:", subcircuit_idx)
#             # print("subcircuit entry init meas:", subcircuit_entry_init_meas)
#             subcircuit_entry_id = meta_info["entry_init_meas_ids"][subcircuit_idx][subcircuit_entry_init_meas]
#             previous_prob = None
#             if previous_merged_probs is not None:

#                 previous_prob_subcirc = previous_merged_probs.get(subcircuit_idx)
#                 if previous_prob_subcirc is not None:    
#                     previous_prob = previous_prob_subcirc.get(subcircuit_entry_init_meas)
                    
#                     is_frozen = frozen_status.get(subcircuit_entry_init_meas)
#                     # print("FF: ",ff)
#                     if is_frozen is not None:
#                         # print("ff subidx:",subcircuit_idx)
                        
#                         # print("########## FROZEN ##############", {subcircuit_idx},{subcircuit_entry_init_meas})
#                         merged_subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_init_meas] = previous_prob
#                         continue


#             file_path = "%s/subcircuit_%d_entry_%d.pckl" % (args.data_folder, subcircuit_idx, subcircuit_entry_id)
#             if os.path.exists(file_path):
#                 unmerged_prob_vector = pickle.load(
#                     open(
#                         file_path,
#                         "rb",
#                     )
#                 )
#             else:
#                 # print("HEre!")
#                 continue
#             # merged_subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_init_meas] = merge_prob_vector(
#             #     unmerged_prob_vector=unmerged_prob_vector,
#             #     qubit_states=qubit_states,
#             # )
#             merged_prob = merge_prob_vector(
#                 unmerged_prob_vector=unmerged_prob_vector,
#                 qubit_states=qubit_states,
#             )
#             if previous_prob is not None and is_close_enough(merged_prob, previous_prob):

#                 # print("previous merged prob bla bla bla ",previous_merged_probs[subcircuit_idx].get(subcircuit_entry_init_meas))
                
#                 # If close enough, mark as frozen and use the previous prob
#                 # print("Close enough! subidx", subcircuit_idx)
#                 # if subcircuit_entry_init_meas not in frozen_status:
#                 #     frozen_status[subcircuit_entry_init_meas] = {}
#                 frozen_status[subcircuit_entry_init_meas] = True
#                 save_frozen_status_for_subcircuit(fold_path,args.rank,subcircuit_idx, frozen_status)
#                 merged_subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_init_meas] = previous_prob
#             else:
#                 # If not, use the new merged prob
#                 # print("CALLED merge")
#                 merged_subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_init_meas] = merged_prob
            
#             # print("####CURRENT PROB###################:",merged_subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_init_meas])

#         save_merged_probs_for_subcircuit(
#             data_folder=fold_path, rank=args.rank, subcircuit_idx=subcircuit_idx, merged_probs=merged_subcircuit_entry_probs
#         )
#     pickle.dump(
#         merged_subcircuit_entry_probs,
#         open("%s/rank_%d_merged_entries.pckl" % (args.data_folder, args.rank), "wb"),
#     )
if __name__ == "__main__":
    """
    The first merge of subcircuit probs using the target number of bins
    Saves the overhead of writing many states in the first SM recursion
    """
    parser = argparse.ArgumentParser(description="Merge probs rank.")
    parser.add_argument("--data_folder", metavar="S", type=str)
    parser.add_argument("--rank", metavar="N", type=int)
    parser.add_argument("--num_workers", metavar="N", type=int)
    args = parser.parse_args()

    meta_info = pickle.load(open("%s/meta_info.pckl" % (args.data_folder), "rb"))
    dd_schedule = pickle.load(open("%s/dd_schedule.pckl" % (args.data_folder), "rb"))

    merged_subcircuit_entry_probs = {}
    for subcircuit_idx in meta_info["entry_init_meas_ids"]:
        rank_jobs = find_process_jobs(
            jobs=list(meta_info["entry_init_meas_ids"][subcircuit_idx].keys()),
            rank=args.rank,
            num_workers=args.num_workers,
        )
        merged_subcircuit_entry_probs[subcircuit_idx] = {}

        # Check if qubit_states is empty for the subcircuit
        qubit_states = dd_schedule["subcircuit_state"][subcircuit_idx]
        # if not qubit_states:
        #     print(f"Warning: Subcircuit {subcircuit_idx} has empty qubit_states")

        for subcircuit_entry_init_meas in rank_jobs:
            subcircuit_entry_id = meta_info["entry_init_meas_ids"][subcircuit_idx][subcircuit_entry_init_meas]
            unmerged_prob_vector = pickle.load(
                open("%s/subcircuit_%d_entry_%d.pckl" % (args.data_folder, subcircuit_idx, subcircuit_entry_id), "rb")
            )
            merged_subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_init_meas] = merge_prob_vector(
                unmerged_prob_vector=unmerged_prob_vector,
                qubit_states=qubit_states,
            )
    pickle.dump(
        merged_subcircuit_entry_probs,
        open("%s/rank_%d_merged_entries.pckl" % (args.data_folder, args.rank), "wb"),
    )