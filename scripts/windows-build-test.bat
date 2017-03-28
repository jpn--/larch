@echo off 
setlocal enableDelayedExpansion 

set uniqid=%random%


for /l %%m in (5, 1, 6) do (

	set ENV_TMP=3%%m-!uniqid!
	echo ENV_TMP is !ENV_TMP!
	set SEEDLING=seedling-!ENV_TMP!
	echo SEEDLING is !SEEDLING!
	set SAPLING=sapling-!ENV_TMP!
	echo SAPLING is !SAPLING!

	set TEMP_BLD_DIR=%temp%\larch-from-git-3%%m-!uniqid!
	echo TEMP_BLD_DIR is !TEMP_BLD_DIR!
	mkdir !TEMP_BLD_DIR!
	git clone https://github.com/jpn--/larch.git !TEMP_BLD_DIR!
	
	cd /D !TEMP_BLD_DIR!

	conda env create --file seedling-3%%m.yml --name !SEEDLING!
	

	  call activate !SEEDLING!
	set OPENBLAS_NUM_THREADS=1


	  python setup.py build
	  EXIT /B 1
	if !ERRORLEVEL! NEQ 0 (
	  REM Install failed
	  echo python build larch returned error !ERRORLEVEL!
	  EXIT /B 1
	)
	
	python setup_test.py
	if !ERRORLEVEL! NEQ 0 (
	  REM Install failed
	  echo python test larch returned error !ERRORLEVEL!
	  EXIT /B 1
	)

)
