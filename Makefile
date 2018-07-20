cython:
	#python setup.py build_ext --inplace
	python3 setup.py build_ext --inplace

test: cython
	python3 -m pytest --nomatlab tests

