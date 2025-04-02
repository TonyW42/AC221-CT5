To install dependencies, run
```
pip install -r requirements.txt
```

To generate the template_db and probe_db directories from your reference_db, run
```
python experiment-maker.py --data_dir {path_to_data} --max_people {max_people} --erase_old
```
- Replace `{path_to_data}` with the path to your dataset root directory (which contains reference_db).
- Replace `{max_people}` with the maximum number of distinct individuals to include in the databases.
- Use `--erase_old` if you want to clear and recreate the databases.

To launch the face recognition demo, run
```
streamlit run main.py
```

To run the experiment and evaluate the face recognition system, execute

```
python experiment_runner.py
```

This will:
1. Initialize the face recognition system
2. Test against all images in probe_db
3. Generate accuracy metrics and detailed results
4. Save results to experiment_results.csv




If not auto-redirected, copy the link that appears in the terminal (e.g., http://localhost:8501) into your browser (e.g., Chrome).
