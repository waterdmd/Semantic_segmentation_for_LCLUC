#%%
import os
import glob as glob 
import random
# %%
input_dir = '/Users/sebrah13/Downloads/SEVNIR/merged/Processed/tile/2023_merged'

list_files = glob.glob(input_dir + '/*.tif')
# %%
print(list_files)
print(len(list_files))

val_list= []
for _ in range(len(list_files)//10):
    
    a = random.choice(list_files)
    val_list.append(a)
print(len(val_list))
# %%
new_list = [x for x in list_files if x not in val_list]
print(len(new_list))
test_list = []
for _ in range(len(list_files)//10):
  
    a = random.choice(new_list)
    test_list.append(a)
    
print(len(test_list))
# %%
train_list =[x for x in new_list if x not in test_list]
print(len(train_list))
# %%
os.mkdir(os.path.join(input_dir, 'train')) if not os.path.exists(os.path.join(input_dir, 'train')) else print('Directory already exists')
os.mkdir(os.path.join(input_dir, 'val')) if not os.path.exists(os.path.join(input_dir, 'val')) else print('Directory already exists')
os.mkdir(os.path.join(input_dir, 'test')) if not os.path.exists(os.path.join(input_dir, 'test')) else print('Directory already exists')

# %%
for i in val_list:
    os.system('mv ' + i + ' ' + input_dir + '/val')
    
for i in test_list:
    os.system('mv ' + i + ' ' + input_dir + '/test')
for i in train_list:
    os.system('mv ' + i + ' ' + input_dir + '/train')

# %%
