import sqlite3
from benchmark import QErrorBenchmark
from estimators.PostgresEstimator import PostgresEstimator
from estimators.HistogramEstimator import HistogramEstimator
from estimators.SampleBasedEstimator import SampleBasedEstimator
from estimators.HyperLogLogEstimator import HyperLogLogEstimator
from estimators.acah.main_estimator import ACAHv3Estimator

def main():
    """Main function to run the benchmark"""
    # Paths to database and queries
    db_path = "./data/imdb.db"
    query_file = "./data/queries.json"
    
    # Create the benchmark
    benchmark = QErrorBenchmark(db_path, query_file)
    
    # Create database connection for estimators
    conn = sqlite3.connect(db_path)
    
    # Add estimators
    # benchmark.add_estimator(PostgresEstimator(conn))
    benchmark.add_estimator(ACAHv3Estimator(conn))
    benchmark.add_estimator(HistogramEstimator(conn))
    # benchmark.add_estimator(SampleBasedEstimator(conn))
    # benchmark.add_estimator(HyperLogLogEstimator(conn))
    
    # Run the benchmark
    benchmark.run_benchmark()
    
    # Analyze the results
    analysis = benchmark.analyze_results()
    print("Overall Analysis:")
    for estimator, metrics in analysis.items():
        print(f"\n{estimator}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # Analyze by number of tables
    analysis_by_tables = benchmark.analyze_by_num_tables()
    print("\nAnalysis by Number of Tables:")
    for estimator, table_metrics in analysis_by_tables.items():
        print(f"\n{estimator}:")
        for num_tables, metrics in sorted(table_metrics.items()):
            print(f"  {num_tables} tables (count: {metrics['count']}):")
            print(f"    Median Q-Error: {metrics['median_q_error']}")
            print(f"    Mean Q-Error: {metrics['mean_q_error']}")
    
    # Plot results
    benchmark.plot_q_error_cdf("q_error_cdf.png")
    benchmark.plot_q_error_by_tables("q_error_by_tables.png")

if __name__ == "__main__":
    main()