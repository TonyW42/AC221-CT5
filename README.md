To install dependencies, run
```
pip install -r requirements.txt
```

To generate the template_db and probe_db directories from your reference_db, run
```
python make_datasets.py --data_dir {path_to_data} --max_people {max_people} --erase_old
```
- Replace `{path_to_data}` with the path to your dataset root directory (which contains reference_db).
- Replace `{max_people}` with the maximum number of distinct individuals to include in the databases.
- Use `--erase_old` if you want to clear and recreate the databases.

To launch the face recognition demo, run
```
streamlit run main.py
```
If not auto-redirected, copy the link that appears in the terminal (e.g., http://localhost:8501) into your browser (e.g., Chrome).
