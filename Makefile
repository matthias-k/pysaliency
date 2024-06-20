cython:
	#python setup.py build_ext --inplace
	python3 setup.py build_ext --inplace

test: cython
	python3 -m pytest --nomatlab tests

prepublish:
	./run-docker.sh rm -rf dist
	./run-docker.sh bash build.sh
	twine upload --repository=pysaliency-test dist/pysaliency*.tar.gz  # assumes that ~/.pypirc defines a pysaliency-test entry, see https://test.pypi.org/manage/account/token/
#	twine upload dist/pysaliency*.tar.gz -r testpypi


publish:
	./run-docker.sh rm -rf dist
	./run-docker.sh bash build.sh
	twine upload --repository=pysaliency dist/pysaliency*.tar.gz

