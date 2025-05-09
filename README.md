## Prerequisites

- **Libraries:** The following libraries are required:
    - `datasets`
    - `transformers`
    - `torch`
    - `json`
    - `os`
    - `google.colab`
    - `spacy` (for POS tagging)


bash !pip install datasets transformers torch spacy

- **Google Drive:** The code uses Google Drive to save and load progress and results.
  Make sure to mount your Google Drive to Colab before running the code.
- **GPU:** Using a GPU is recommended for faster inference. 

## Notes: 
In each .py file, change file paths as neccessary to your GPU or CPU. Google Colabs is highly recommended, since
files are intermediately saved virtually to save computing power, but it is not mandatory. Code is currently set up for GPU usage. 

## Warning: 
This code has very slow runtimes and so it is reccomended to use a validation split of under 1000 for quicker results.
This will need to be manually adjusted at the input file pipelines. 



## Steps:

1. **initial_predictions.py:** Imports necessary modules, mounts Google Drive, defines file paths for saving progress,
   loads the SQuAD dataset, sets up the QA pipeline, and performs initial inference on the SQuAD dataset.
   It saves the progress and the final predictions to JSON files on Google Drive. 
2. **paraphraser.py:** Imports modules, mounts Google Drive, loads the SQuAD dataset, initializes a paraphrasing model,
   generates paraphrases for each question, and saves them to a JSON file on Google Drive.
3. **remove_duplicates.py:** Removes duplicate paraphrases from the generated JSON file by comparing 'id' values.
   Important to check intermediate status of results for clarity. 
4. **score_variants.py:** Mounts Google Drive, loads paraphrased questions and QA data, runs the QA pipeline on each paraphrase,
   stores predictions (including original question, paraphrased question, answer, and score), and saves them to a file on Google Drive.
5. **variant_evaluation.py:** Loads the predictions generated from the previous cell, groups them by 'id', and then evaluates the score difference between the original questions and the         
paraphrased questions, saving the results to a file.
6. **trigram_generation.py:** Imports libraries, loads the predicted data with paraphrases, extracts POS trigrams for each paraphrase and the
   original question, and stores the annotated data in a JSON file.
8. **predictions_with_trigrams.py:** Groups the data by question ID, tracks trigram performance, calculates score differences,
   updates trigram statistics, computes intermediate metrics, and saves the results in a JSON file.
9. **avg_model_scores.py:** Creates a bar chart comparing the average score of the original questions and the paraphrased questions.
10. **contribution_scores.py:** Loads the prediction data, groups it by question ID, tracks trigram delta scores, aggregates average
    contribution per trigram, and saves the results to a JSON file. 

