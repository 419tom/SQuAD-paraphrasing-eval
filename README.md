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

## Purpose of Each Cell

1. **Install `datasets`:** Installs the `datasets` library, which is used to load the SQuAD dataset.
2. **QA pipeline setup and initial run:** Imports necessary modules, mounts Google Drive, defines file paths for saving progress,
   loads the SQuAD dataset, sets up the QA pipeline, and performs initial inference on the SQuAD dataset.
   It saves the progress and the final predictions to JSON files on Google Drive.
4. **Install `datasets` (repeated):** Installs the `datasets` library again. This seems redundant and might be removed.
5. **Paraphrasing setup and execution:** Imports modules, mounts Google Drive, loads the SQuAD dataset, initializes a paraphrasing model,
   generates paraphrases for each question, and saves them to a JSON file on Google Drive.
7. **Resuming paraphrasing and saving:** Initializes paraphrasing, checks for existing progress, loads the model and data, defines the
   paraphrasing function, paraphrases questions incrementally, and saves the paraphrased data and progress index.
9. **Remove duplicates:** Removes duplicate paraphrases from the generated JSON file by comparing 'id' values.
10. **QA on paraphrases:** Mounts Google Drive, loads paraphrased questions and QA data, runs the QA pipeline on each paraphrase, stores predictions (including original question, paraphrased question, answer, and score), and saves them to a file on Google Drive.
11. **Paraphrase evaluation:** Loads the predictions generated from the previous cell, groups them by 'id', and then evaluates the score difference between the original questions and the paraphrased questions, saving the results to a file.
12. **POS trigram extraction:** Imports libraries, loads the predicted data with paraphrases, extracts POS trigrams for each paraphrase and the original question, and stores the annotated data in a JSON file.
