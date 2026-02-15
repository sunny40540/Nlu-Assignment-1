"""
Assignment 1 - Problem 4: Sports vs Politics Classifier
Name: Sunny Kumar
Roll Number: B23CS1071
Date: January 19, 2026

Description:
This program classifies text into Sports or Politics categories.
It compares three machine learning algorithms:
1. Naive Bayes
2. Support Vector Machine
3. Logistic Regression

Feature Extraction:
TF-IDF representation is used for converting text to numerical features.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ---------------- DATASET ----------------
def get_data():

    sports_texts = [
        "The team won the cricket match in the final over",
        "The football player scored a hat trick",
        "Olympic athletes prepared for the competition",
        "The tennis champion won the grand slam title",
        "The basketball match went into overtime",
        "The coach praised the team performance",
        "The player suffered a knee injury during match",
        "Fans celebrated victory in the stadium",
        "The bowler took five wickets in the match",
        "The striker scored the winning goal",
        "The marathon runner finished first",
        "The goalkeeper made a brilliant save",
        "The hockey team entered the semifinals",
        "The swimming competition broke records",
        "The boxing champion defended his title",
        "The referee stopped the match",
        "The league tournament begins tomorrow",
        "The player was awarded man of the match",
        "The team practiced hard before the game",
        "The championship final attracted huge crowd"
    ]

    politics_texts = [
        "The government passed a new law in parliament",
        "The election campaign started today",
        "The prime minister addressed the nation",
        "The president met foreign diplomats",
        "The parliament discussed economic policy",
        "The minister resigned after corruption charges",
        "The opposition party criticized the government",
        "The senate approved the budget proposal",
        "The political debate focused on development",
        "The leader promised new reforms",
        "The election results were announced",
        "The government introduced tax reforms",
        "The cabinet meeting discussed national security",
        "The party gained majority in election",
        "The bill was rejected in assembly",
        "The constitution guarantees citizen rights",
        "The mayor launched welfare scheme",
        "The administration faced public protests",
        "The political rally gathered supporters",
        "The policy will affect public services"
    ]

    texts = sports_texts + politics_texts
    labels = ["Sports"] * len(sports_texts) + ["Politics"] * len(politics_texts)

    return texts, labels


# ---------------- MAIN ----------------
def main():

    print("\nLoading dataset...")
    texts, labels = get_data()
    print("Total samples:", len(texts))

    # Feature extraction
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=42
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "Support Vector Machine": SVC(kernel="linear"),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    print("\n---- Model Comparison ----")
    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        results[name] = acc

        print(f"\n{name} Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, predictions))

    # summary table
    print("\nFinal Accuracy Comparison")
    print("---------------------------")
    for model, acc in results.items():
        print(f"{model:25s} : {acc*100:.2f}%")


if __name__ == "__main__":
    main()
