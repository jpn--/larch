@echo off 
> larch_build_log-%DATE:~-4%-%DATE:~4,2%-%DATE:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log (

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
		if !ERRORLEVEL! NEQ 0 (
		  REM Install failed
		  set LARCHBAT_ERR=!ERRORLEVEL!
		  echo python 3.%%m build larch returned error !LARCHBAT_ERR!
		  call deactivate
		  conda env remove -n !SEEDLING! --yes
		  echo python 3.%%m build larch returned error !LARCHBAT_ERR!
		  EXIT /B 1
		)
	
		python setup_test.py
		if !ERRORLEVEL! NEQ 0 (
		  REM Install failed
		  set LARCHBAT_ERR=!ERRORLEVEL!
		  echo python 3.%%m test larch returned error !LARCHBAT_ERR!
		  call deactivate
		  conda env remove -n !SEEDLING! --yes
		  echo python 3.%%m test larch returned error !LARCHBAT_ERR!
		  EXIT /B 1
		)

	
	
	
	
		python setup.py bdist_wheel --keep-temp


		conda env create --file seedling-3%%m.yml --name !SAPLING!
		call activate !SAPLING!
		cd ..
		pip install !TEMP_BLD_DIR!/dist/larch-*.whl

		python -c"import larch.test; larch.test.run(exit=True,fail_fast=True)"

		if !ERRORLEVEL! NEQ 0 (
			set LARCHBAT_ERR=!ERRORLEVEL!
			echo python 3.%%m abort on installer test fail error !LARCHBAT_ERR!
			call deactivate
			conda env remove -n !SAPLING! --yes
			conda env remove -n !SEEDLING! --yes
			echo python 3.%%m abort on installer test fail error !LARCHBAT_ERR!
			EXIT /B 1
		) else (
			echo test on installer python3%%m success;
		)

		call deactivate
		conda env remove -n !SAPLING! --yes
		conda env remove -n !SEEDLING! --yes

	
	
	
	)
)