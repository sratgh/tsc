import os
from glob import glob, iglob
from tqdm import tqdm

def convert_recursive(filepath=".", convert_from='ppm', convert_to='png', test_set=False, delete_old=False):
	#print(filepath+"*."+convert_from)
	if not test_set:
		folder_names_classes = sorted(os.listdir(filepath))
	else:
		folder_names_classes=['']
		
	for name in tqdm(folder_names_classes):
	#files = iglob(filepath+"**/*."+convert_from, recursive=True)
	
		#print(filepath+name)

		files = glob(os.getcwd()+'/'+filepath+name+"/*."+convert_from)
		for file in files:
			#print(file)
			convert_files_in_folder = "convert "+file +" "+file[:-4]+"."+convert_to
			#print(convert_files_in_folder)
			os.system(convert_files_in_folder)
			os.remove(file)

convert_recursive(filepath="test/", test_set=True, delete_old=True)