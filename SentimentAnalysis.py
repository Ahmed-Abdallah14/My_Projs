from textblob import TextBlob

print("--- AI Sentiment Analyzer System ---")
print("Enter 'exit' to close the program\n")

while True:

    user_input = input("Enter a sentence to analyze: ")

    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    print(f"Analysis Score: {round(polarity, 2)}")
    print(f"Result: {sentiment}")
    print("\n")
