install-dependencies:
	conda install -y -c anaconda click scipy numpy pymongo scikit-learn pandas jupyter
	conda install -y -c conda-forge tqdm matplotlib
	pip install sacred tables vdom
	conda install -y -c pytorch pytorch torchvision
	conda install -y h5py
