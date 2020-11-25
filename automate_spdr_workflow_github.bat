@ECHO OFF 
TITLE Running SPDR ETF Gold Prices Forecasting Pipeline in Task Scheduler
ECHO Please Wait...
:: Section 1: Activate the environment.
ECHO ============================
ECHO Activating Conda Environment
ECHO ============================
@CALL "<full_path_to_conda_batch_file>\activate.bat" "<full_path_to_conda_venv_folder_if_not_in_usual_conda_location>\envs_spdr_gold"
:: Section 2: Execute python script.
ECHO ============================
ECHO Change cwd
ECHO ============================
cd /d "<full_path_to_project_root_folder>"
ECHO ============================
ECHO Run full pipeline script
ECHO ============================
python ".\<location_of_py_script_relative_to_root>\RUN_PIPELINE_V002.py"

ECHO ============================
ECHO End
ECHO ============================

EXIT
::PAUSE