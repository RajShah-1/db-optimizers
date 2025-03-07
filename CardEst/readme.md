# Cardinality Estimator Benchmark

This project benchmarks different cardinality estimators using the IMDB dataset.

## Prerequisites

* Python 3.6 or higher
* pip (Python package installer)

## Setup

1.  **Clone the Repository (if applicable):**

    ```bash
    git clone <your_repository_url>
    cd <your_repository_directory>
    ```

2.  **Create a Virtual Environment:**

    ```bash
    python3 -m venv venv
    ```

    This command creates a virtual environment named `venv` in your project directory.

3.  **Activate the Virtual Environment:**

    * **On macOS and Linux:**

        ```bash
        source venv/bin/activate
        ```

    * **On Windows:**

        ```bash
        venv\Scripts\activate
        ```

    Your terminal prompt should now indicate that the virtual environment is active (e.g., `(venv) $`).

4.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    This command installs the required Python packages (pandas, numpy, matplotlib, seaborn) into your virtual environment.

5.  **Download the IMDB Dataset (if needed):**

    * Ensure you have the IMDB dataset in a SQLite database file named `imdb.db` and place it in the `./data/` directory.
    * Also ensure that the `queries.json` file is in the `./data/` directory.
    * If you don't have the dataset, you'll need to obtain it and populate the database.
    * This script assumes the database exists with the tables used by the queries in `queries.json`.

6.  **Run the Script:**

    ```bash
    python main.py
    ```

    This command executes the `main.py` script, which runs the benchmark and generates the output.

## Output

The script will produce the following output:

* **Overall Analysis:** A summary of the performance of each cardinality estimator, including median q-error, mean q-error, max q-error, percentiles, and mean execution time.
* **Analysis by Number of Tables:** A breakdown of the performance of each estimator grouped by the number of tables involved in the queries.
* **Plots:**
    * `q_error_cdf.png`: A cumulative distribution function (CDF) plot of the q-errors for each estimator.
    * `q_error_by_tables.png`: A bar plot showing the median q-error for each estimator grouped by the number of joined tables.

## Deactivate the Virtual Environment (when finished):**

```bash
deactivate
```

Notes
Dataset Setup: The script assumes that the IMDB dataset is correctly formatted and populated in the SQLite database. Ensure that the database schema matches the queries in queries.json.
Estimator Implementation: The HistogramEstimator, SampleBasedEstimator and HyperLogLogEstimator classes contain placeholder implementations. You'll need to fill in the estimation logic based on the algorithms you want to implement.
Query Parsing: The script uses a simple regular expression-based parser for SQL queries. For more complex queries, you might need to use a proper SQL parser.
Error Metrics: The benchmark uses q-error as the primary error metric. You can add other error metrics as needed.
Visualization: The script uses matplotlib and seaborn for visualization. You can customize the plots to suit your needs.
Example Directory Structure
your_project_directory/
├── venv/
├── data/
│   ├── imdb.db
│   └── queries.json
├── main.py
├── requirements.txt
└── README.md
This README should guide you through setting up the environment and running the benchmark successfully.