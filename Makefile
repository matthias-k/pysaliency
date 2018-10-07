cython:
	#python setup.py build_ext --inplace
	python3 setup.py build_ext --inplace

test: cython
	python3 -m pytest --nomatlab tests

prepublish:
	./run-docker.sh rm -rf dist
	./run-docker.sh bash build.sh
	twine upload dist/pysaliency*.tar.gz -r testpypi


publish:
	./run-docker.sh rm -rf dist
	./run-docker.sh bash build.sh
	twine upload dist/pysaliency*.tar.gz
