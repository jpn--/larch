#!/bin/bash

#conda env create jpn/seedling-3.6 --name seedling36

for MINOR in 5 6
do

	TEMPDIR=`mktemp -d /tmp/larch-from-git-3${MINOR}-XXXXXXXX`
	CURDIR=`pwd`
	ENV_TMP=3${MINOR}-$$-${RANDOM}
	SEEDLING=seedling-${ENV_TMP}
	SAPLING=sapling-${ENV_TMP}
	git clone https://github.com/jpn--/larch.git ${TEMPDIR}
	cd ${TEMPDIR}
	conda env create --file seedling-3${MINOR}.yml --name ${SEEDLING}
	source activate ${SEEDLING}
	python setup.py build
	python setup_test.py

	if [ $? -ne 0 ]; then
		echo 'abort on builder test fail';
		source deactivate
		conda env remove -n ${SEEDLING} --yes
		exit -1 ;
	else
		echo test on builder python3.${MINOR} success;
	fi

	python setup.py bdist_wheel --keep-temp


	conda env create --file seedling-3${MINOR}.yml --name ${SAPLING}
	source activate ${SAPLING}
	#pip uninstall larch --yes
	cd ..
	pip install ${TEMPDIR}/dist/larch-*.whl

	python -c"import larch.test; larch.test.run(exit=True,fail_fast=True)"

	if [ $? -ne 0 ]; then
		echo abort on installer test fail;
		source deactivate
		conda env remove -n ${SAPLING} --yes
		conda env remove -n ${SEEDLING} --yes
		exit -1 ;
	else
		echo test on installer python3.${MINOR} success;
	fi

	source deactivate
	conda env remove -n ${SAPLING} --yes
	conda env remove -n ${SEEDLING} --yes

	cd $CURDIR

done