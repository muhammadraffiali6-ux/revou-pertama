# Requirements Document

## Introduction

This feature is a web-based prototype application for detecting whether Indonesian-language news text is factual ("FAKTA") or a hoax ("HOAX"). The system accepts raw news text pasted by the user, preprocesses it using Indonesian NLP techniques, classifies it using a machine learning model (TF-IDF with Naive Bayes or Logistic Regression), and presents the result with a confidence score through a Streamlit user interface. The application is intended as a starting point for Indonesian news fact-checking efforts.

## Glossary

- **App**: The Streamlit-based web application being built.
- **User**: A person who interacts with the App via the browser to check news text.
- **News_Text**: Raw Indonesian-language news article or snippet pasted by the User.
- **Preprocessor**: The module responsible for cleaning and normalizing News_Text.
- **Stopword_Remover**: The component that removes common Indonesian stopwords using the Sastrawi or NLTK library.
- **Stemmer**: The component that reduces Indonesian words to their root form using the Sastrawi library.
- **Classifier**: The trained machine learning model (TF-IDF vectorizer combined with Naive Bayes or Logistic Regression) that predicts whether a given text is FAKTA or HOAX.
- **Label**: The predicted classification output: either "FAKTA" or "HOAX".
- **Confidence_Score**: A numeric percentage (0–100%) representing the Classifier's certainty in the predicted Label.
- **Dataset**: The labeled sample training data consisting of Indonesian news text examples annotated as FAKTA or HOAX.
- **requirements.txt**: The file listing all Python package dependencies required to run the App.

---

## Requirements

### Requirement 1: Text Input

**User Story:** As a User, I want to paste Indonesian news text into the App, so that I can submit it for hoax detection analysis.

#### Acceptance Criteria

1. THE App SHALL render a multi-line text input area on the main page where the User can paste News_Text of up to 10,000 characters.
2. THE App SHALL render an "Analyze" button that the User can click to trigger classification.
3. IF the User clicks the "Analyze" button and the text input is empty, THEN THE App SHALL display an error message prompting the User to enter News_Text before proceeding.
4. IF the User clicks the "Analyze" button and the News_Text contains fewer than 10 characters, THEN THE App SHALL display an error message indicating that the input is too short for analysis.
5. IF the User pastes or enters News_Text exceeding 10,000 characters, THEN THE App SHALL display an error message indicating the character limit has been exceeded and SHALL NOT submit the text for analysis.
6. WHEN the User modifies the News_Text such that it meets the minimum length requirement, THE App SHALL clear any previously displayed input validation error message.

---

### Requirement 2: Text Preprocessing

**User Story:** As a User, I want the App to clean and normalize my input text, so that the Classifier receives well-formed Indonesian language data.

#### Acceptance Criteria

1. WHEN News_Text is submitted, THE Preprocessor SHALL convert all characters in the News_Text to lowercase.
2. WHEN News_Text is submitted, THE Preprocessor SHALL remove URLs, punctuation, numbers, and non-alphabetic non-space characters from the News_Text, and SHALL collapse any resulting multiple consecutive spaces into a single space and trim leading and trailing whitespace.
3. WHEN News_Text is submitted, THE Stopword_Remover SHALL remove common Indonesian stopwords from the normalized text using the Sastrawi stopword list.
4. WHEN News_Text is submitted, THE Stemmer SHALL reduce remaining tokens to their Indonesian root forms using the Sastrawi PySastrawi stemmer.
5. THE Preprocessor SHALL apply all preprocessing steps in the following order: lowercase conversion, then character removal and whitespace normalization, then stopword removal, then stemming.
6. IF the Preprocessor produces an empty string after all preprocessing steps, THEN THE App SHALL display an error message indicating that the text contains no analyzable content.

---

### Requirement 3: Model Training and Classification

**User Story:** As a developer, I want the App to train a machine learning model on a sample dataset, so that the Classifier can predict whether news is FAKTA or HOAX.

#### Acceptance Criteria

1. THE App SHALL include a built-in Dataset of at least 20 labeled Indonesian news text samples, where each class (FAKTA and HOAX) represents between 40% and 60% of the total samples.
2. THE Classifier SHALL be trained using TF-IDF vectorization combined with either a Multinomial Naive Bayes or Logistic Regression algorithm.
3. WHEN the App starts, THE Classifier SHALL be trained on the Dataset using at least a 70/30 train-test split and held in memory for the duration of the session.
4. WHEN preprocessed News_Text is passed to the Classifier, THE Classifier SHALL return a Label ("FAKTA" or "HOAX") and a Confidence_Score expressed as a percentage rounded to two decimal places between 0.00% and 100.00%.
5. IF the App fails to train the Classifier on startup, THEN THE App SHALL display an error message indicating that training failed and halt further execution.
6. IF the News_Text passed to the Classifier is empty or blank, THEN THE Classifier SHALL reject the input and return an error message indicating that the input is invalid without returning a Label or Confidence_Score.

---

### Requirement 4: Classification Output Display

**User Story:** As a User, I want to see a clear and visually distinct result after analysis, so that I can immediately understand whether the news is FAKTA or HOAX.

#### Acceptance Criteria

1. WHEN the Classifier returns a Label of "FAKTA", THE App SHALL display the label "FAKTA" in green-colored text.
2. WHEN the Classifier returns a Label of "HOAX", THE App SHALL display the label "HOAX" in red-colored text.
3. WHEN the Classifier returns a result, THE App SHALL display the Confidence_Score as a whole-number percentage value between 0 and 100 alongside the Label.
4. WHEN the Classifier returns a result, THE App SHALL display a visual confidence bar representing the Confidence_Score, where the bar fill corresponds proportionally to the Confidence_Score value.
5. IF the Classifier returns a Confidence_Score of less than 60%, THEN THE App SHALL display a notice indicating that the confidence level is low and the result may be unreliable.
6. IF the Classifier returns a Label that is neither "FAKTA" nor "HOAX", THEN THE App SHALL display an error message indicating that the classification result is invalid and the analysis could not be completed.

---

### Requirement 5: User Interface and Usability

**User Story:** As a User, I want a clean and intuitive web interface, so that I can use the App without technical knowledge.

#### Acceptance Criteria

1. THE App SHALL display a title and a description of no more than 280 characters summarizing the App's purpose on the main page.
2. THE App SHALL render all primary UI elements (text input, button, and results) on a single page without requiring navigation.
3. WHEN the User submits News_Text for classification, THE App SHALL display a loading indicator and continue displaying it until the Classifier returns a result or an error.
4. WHEN a classification result is displayed, THE App SHALL provide a "Reset" or "Clear" mechanism that clears the News_Text input field, removes the displayed result, and returns the interface to its initial state without refreshing the page.

---

### Requirement 6: Dependency and Setup Documentation

**User Story:** As a developer, I want clear setup instructions and a complete requirements.txt, so that I can install and run the App in a new environment.

#### Acceptance Criteria

1. THE App SHALL include a requirements.txt file listing all direct Python package dependencies, each pinned to an exact version using the `==` specifier.
2. THE App SHALL include a README.md file containing step-by-step setup instructions that cover virtual environment creation, dependency installation via requirements.txt, and the command to launch the App.
3. WHEN a developer runs `streamlit run app.py` in an environment where all dependencies from requirements.txt are installed, THE App SHALL start successfully and present the application UI in the browser without errors.
4. THE App SHALL include inline code comments on every function definition and every logically distinct code section explaining the purpose and logic of that code.
