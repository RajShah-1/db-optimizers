import sqlite3
import numpy as np
import matplotlib.pyplot as plt
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
    # benchmark.add_estimator(HistogramEstimator(conn))
    # benchmark.add_estimator(SampleBasedEstimator(conn))
    # benchmark.add_estimator(HyperLogLogEstimator(conn))
    
    # Run the benchmark
    # benchmark.run_benchmark()

    # Run comparative feedback benchmark for estimators that support it
    for estimator in benchmark.estimators:
        if hasattr(estimator, "apply_feedback"):
            print(f"\n[Feedback Benchmark] Running feedback-aware benchmark for {estimator.name}...")
            results = benchmark.run_comparative_feedback_benchmark(estimator)

            print("Mean Q-error (before feedback):", np.mean(results['before']))
            print("Mean Q-error (after feedback):", np.mean(results['after']))

            plt.plot(results['before'], label="Before Feedback")
            plt.plot(results['after'], label="After Feedback")
            plt.yscale("log")
            plt.xlabel("Query Index")
            plt.ylabel("Q-error")
            plt.title(f"Q-error Before vs. After Feedback ({estimator.name})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"qerror_feedback_comparison_{estimator.name}.png")
            plt.show()

            # Save manually into benchmark.results for compatibility
            # benchmark.results[estimator.name] = {
            #     'q_errors': results['after'],  # So post-feedback Q-errors go into standard analysis
            # }
            benchmark.results[estimator.name] = [
                {"q_error": q} for q in results["after"]
            ]

        else:
            print(f"\n[Standard Benchmark] Running standard benchmark for {estimator.name}...")
            benchmark.results[estimator.name] = benchmark._evaluate_estimator(estimator)

    
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