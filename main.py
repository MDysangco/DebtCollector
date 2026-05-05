from pipeline import run_pipeline

def main():
    print("Running WF tuning pipeline...\n")

    result = run_pipeline(use_tuning=True)

    print("\n=== WF TUNING COMPLETE ===\n")

    print("WF Stats (first 5 folds):")
    print(result["wf_stats"][:5])

    print("\nWF Trades (first 10):")
    print(result["wf_trades"][:10])

if __name__ == "__main__":
    main()
