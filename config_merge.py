import os

# front-end parameter settings
win_size = 320
fft_num = 320
win_shift = 160
chunk_length = 4*16000
feat_type = 'sqrt' #the compression on magnitude  # normal, sqrt, cubic, log_1x
is_conti = False   #False
conti_path = r"C:\Users\Admin\Downloads\codes\DB_CPT\CP_dir\XXX\checkpoint_early_exit_26th.pth.tar"
is_pesq =  False  # True  #use pesq criterion for validate or not
# server parameter settings
json_dir = r"C:\Users\Admin\Downloads\codes\DB_CPT\Json"
file_path = r"Y:\dataset_low_snr_10_15"
loss_dir = r"C:\Users\Admin\Downloads\codes\DB_CPT\LOSS\XXX.mat"
batch_size = 2
epochs = 100
lr = 5e-4
model_best_path = r"C:\Users\Admin\Downloads\codes\DB_CPT\BEST_MODEL\XXX.pth.tar"
check_point_path = r"C:\Users\Admin\Downloads\codes\DB_CPT\CP_dir\XXX"

#os.makedirs(r"C:\Users\Admin\Downloads\codes\DB_CPT\BEST_MODEL", exist_ok=True)
#os.makedirsr"'C:\Users\Admin\Downloads\codes\DB_CPT\LOSS", exist_ok=True)
#os.makedirs(check_point_path, exist_ok=True)