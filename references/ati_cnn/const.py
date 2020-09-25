"""
"""

SEED = 42

freq = 500
cell_len_t = 6
model_input_len = freq * model_input_len

batch_size = 128

all_labels = ['N', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
nb_classes = len(all_labels)

nb_leads = 12
