def main(analyzer: SentimentAnalyzer):
    model = analyzer.train()
    model = analyzer.load_from_file()

    # read in inference dataset

    y_pred = model.predict(dataset)

    print(...) # metrics
