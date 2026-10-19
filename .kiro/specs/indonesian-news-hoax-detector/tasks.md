# Implementation Plan: Indonesian News Hoax Detector

## Overview

Implement a single-page Streamlit web application that classifies Indonesian-language news text as "FAKTA" or "HOAX". The implementation follows a three-module structure (`preprocessor.py`, `model.py`, `app.py`) with pinned dependencies in `requirements.txt` and setup instructions in `README.md`.

## Tasks

- [ ] 1. Set up project structure and dependencies
  - [-] 1.1 Create `requirements.txt` with all pinned dependencies
    - Include `streamlit`, `scikit-learn`, `PySastrawi`, `hypothesis`, `pytest`, `numpy` each pinned to an exact version using `==`
    - _Requirements: 6.1_
  - [-] 1.2 Create `README.md` with setup and launch instructions
    - Cover virtual environment creation, `pip install -r requirements.txt`, and `streamlit run app.py`
    - _Requirements: 6.2, 6.3_

- [ ] 2. Implement `preprocessor.py` — text preprocessing pipeline
  - [~] 2.1 Implement internal helper functions and the public `preprocess()` function
    - Implement `_lowercase(text)`, `_remove_noise(text)`, `_remove_stopwords(text)` using `StopWordRemoverFactory`, and `_stem(text)` using `StemmerFactory`
    - Implement `preprocess(text: str) -> str` that applies all four steps in order: lowercase → remove noise → remove stopwords → stem
    - Raise `ValueError` if `text` is `None`
    - Return empty string (not an error) if preprocessing reduces the text to nothing
    - Add inline comments on every function and every logically distinct code section
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 6.4_
  - [ ]* 2.2 Write property test for lowercase invariant
    - **Property 3: Lowercase invariant**
    - **Validates: Requirements 2.1**
    - Generate arbitrary Unicode strings; assert `preprocess(text)` contains no uppercase characters
    - Tag: `# Feature: indonesian-news-hoax-detector, Property 3: Lowercase invariant`
  - [ ]* 2.3 Write property test for noise character removal
    - **Property 4: Preprocessing removes all noise characters**
    - **Validates: Requirements 2.2**
    - Generate strings with URLs, digits, and punctuation; assert output contains none, has no leading/trailing whitespace, and no consecutive spaces
    - Tag: `# Feature: indonesian-news-hoax-detector, Property 4: Preprocessing removes all noise characters`
  - [ ]* 2.4 Write property test for stopword removal idempotency
    - **Property 5: Stopword removal is idempotent**
    - **Validates: Requirements 2.3**
    - Generate arbitrary strings; assert `_remove_stopwords(_remove_stopwords(t)) == _remove_stopwords(t)`
    - Tag: `# Feature: indonesian-news-hoax-detector, Property 5: Stopword removal is idempotent`
  - [ ]* 2.5 Write property test for stemming idempotency
    - **Property 6: Stemming is idempotent**
    - **Validates: Requirements 2.4**
    - Generate arbitrary lowercase alphabetic strings; assert `_stem(_stem(t)) == _stem(t)`
    - Tag: `# Feature: indonesian-news-hoax-detector, Property 6: Stemming is idempotent`
  - [ ]* 2.6 Write unit tests for `preprocessor.py`
    - Test lowercase conversion, URL removal, punctuation removal, number removal, whitespace normalization, stopword removal, stemming, pipeline order, and empty-after-preprocessing edge case
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [~] 3. Checkpoint — Ensure preprocessor tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Implement `model.py` — dataset, training, and classification
  - [~] 4.1 Define the built-in `DATASET` and `DatasetEntry` dataclass
    - Create at least 20 labeled samples where each class ("FAKTA" and "HOAX") represents 40–60% of total entries
    - Define `DatasetEntry` dataclass with `text: str` and `label: str` fields
    - Add inline comments on every logically distinct code section
    - _Requirements: 3.1, 6.4_
  - [~] 4.2 Implement `load_model()` with `@st.cache_resource`
    - Build a `sklearn.pipeline.Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])` trained on `DATASET` with at least a 70/30 train-test split
    - Raise `RuntimeError` if training fails for any reason
    - Return `(pipeline, test_accuracy)` tuple
    - Add inline comments on every function and logically distinct section
    - _Requirements: 3.2, 3.3, 3.5, 6.4_
  - [~] 4.3 Implement `classify()` function
    - Accept `pipeline` and `preprocessed_text: str`; raise `ValueError` if text is empty or blank
    - Return `{"label": str, "confidence": float}` where label ∈ {"FAKTA", "HOAX"} and confidence ∈ [0.00, 100.00] rounded to two decimal places
    - Add inline comments on every function and logically distinct section
    - _Requirements: 3.4, 3.6, 6.4_
  - [ ]* 4.4 Write property test for valid classifier output
    - **Property 7: Classifier always returns a valid label and bounded confidence**
    - **Validates: Requirements 3.4**
    - Generate non-empty preprocessed text strings; assert label ∈ {"FAKTA", "HOAX"} and confidence ∈ [0.00, 100.00]
    - Tag: `# Feature: indonesian-news-hoax-detector, Property 7: Classifier always returns a valid label and bounded confidence`
  - [ ]* 4.5 Write unit tests for `model.py`
    - Test dataset balance (40–60% per class), dataset size (≥ 20), `load_model()` returns a fitted pipeline, `classify()` returns valid label and confidence, `classify()` raises `ValueError` on empty and whitespace-only input
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6_

- [~] 5. Checkpoint — Ensure model tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement `app.py` — Streamlit UI layer
  - [~] 6.1 Implement session state initialization and header
    - Initialize `st.session_state` keys: `result`, `error`, `input_key` (per `AppState` schema)
    - Render title and a ≤280-character description of the app's purpose
    - Add inline comments on every logically distinct section
    - _Requirements: 5.1, 5.2, 6.4_
  - [~] 6.2 Implement input validation logic as a standalone function
    - Implement `validate_input(text: str) -> str | None` that returns an error message string or `None`
    - Reject empty input (Req 1.3), input with fewer than 10 stripped characters (Req 1.4), and input exceeding 10,000 characters (Req 1.5)
    - Add inline comments
    - _Requirements: 1.3, 1.4, 1.5, 6.4_
  - [~] 6.3 Implement the text input area and Analyze button
    - Render `st.text_area` keyed by `input_key` with a max-chars note in `help` text
    - On "Analyze" click: run `validate_input()`, call `preprocess()`, check for empty preprocessed result, call `classify()`, store result or error in session state
    - Wrap classification in `st.spinner` loading indicator
    - Catch `RuntimeError` from model training failure and halt UI rendering with a fatal error message
    - Catch `ValueError` from `classify()` and store as error
    - Add inline comments
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.6, 3.5, 3.6, 5.3, 6.4_
  - [~] 6.4 Implement the results panel and Reset button
    - Display label in green ("FAKTA") or red ("HOAX") using `st.markdown` with inline color styling
    - Display confidence as a whole-number percentage and a `st.progress` bar with fill = `confidence / 100`
    - Display low-confidence notice if `confidence < 60.0`
    - Display error message for invalid label (neither "FAKTA" nor "HOAX")
    - Render a "Reset" / "Clear" button that clears `result`, `error`, and increments `input_key` without a page refresh
    - Add inline comments
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.4, 6.4_
  - [ ]* 6.5 Write property test for input validation (short and empty)
    - **Property 1: Input validation rejects short and empty inputs**
    - **Validates: Requirements 1.3, 1.4**
    - Generate strings with `len(text.strip()) < 10`; assert `validate_input()` returns a non-None error and `classify` is not called
    - Tag: `# Feature: indonesian-news-hoax-detector, Property 1: Input validation rejects short and empty inputs`
  - [ ]* 6.6 Write property test for input validation (over-length)
    - **Property 2: Input validation rejects over-length inputs**
    - **Validates: Requirements 1.5**
    - Generate strings with `len(text) > 10_000`; assert `validate_input()` returns a non-None error
    - Tag: `# Feature: indonesian-news-hoax-detector, Property 2: Input validation rejects over-length inputs`
  - [ ]* 6.7 Write property test for confidence display rendering
    - **Property 8: Confidence display renders correctly for any confidence value**
    - **Validates: Requirements 4.3, 4.4**
    - Generate float values in [0.0, 100.0]; assert display shows rounded integer and progress bar value equals `confidence / 100`
    - Tag: `# Feature: indonesian-news-hoax-detector, Property 8: Confidence display renders correctly for any confidence value`
  - [ ]* 6.8 Write property test for low-confidence notice threshold
    - **Property 9: Low-confidence notice appears if and only if confidence is below threshold**
    - **Validates: Requirements 4.5**
    - Generate arbitrary float confidence values; assert notice is shown iff `confidence < 60.0`
    - Tag: `# Feature: indonesian-news-hoax-detector, Property 9: Low-confidence notice appears if and only if confidence is below threshold`
  - [ ]* 6.9 Write property test for Reset restoring initial state
    - **Property 10: Reset returns the app to its initial state for any result state**
    - **Validates: Requirements 5.4**
    - Generate arbitrary classification result states; assert after Reset, `result=None`, `error=None`, and `input_key` is incremented
    - Tag: `# Feature: indonesian-news-hoax-detector, Property 10: Reset returns the app to its initial state for any result state`
  - [ ]* 6.10 Write unit tests for `app.py` validation logic and results rendering
    - Test all `validate_input()` boundary cases (empty, 9 chars, 10 chars, 10 000 chars, 10 001 chars) and low-confidence notice logic
    - _Requirements: 1.3, 1.4, 1.5, 4.5_

- [ ] 7. Implement integration tests
  - [~] 7.1 Write full pipeline integration test
    - Test: raw Indonesian text → `preprocess()` → `classify(pipeline, ...)` → valid `ClassificationResult` with label ∈ {"FAKTA", "HOAX"} and confidence ∈ [0.0, 100.0]
    - _Requirements: 3.4_
  - [~] 7.2 Write `load_model()` startup integration test
    - Test that `load_model()` completes without error and returns a fitted pipeline with non-zero test accuracy
    - _Requirements: 3.3, 3.5_

- [~] 8. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at module boundaries
- Property tests validate universal correctness properties using the Hypothesis library (min 100 iterations each)
- Unit tests validate specific examples, boundary values, and error conditions
- The `@st.cache_resource` decorator on `load_model()` ensures training runs only once per server process — tests that call `load_model()` directly should clear the cache between runs using `load_model.clear()`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "2.5", "2.6", "4.2"] },
    { "id": 3, "tasks": ["4.3", "6.1", "6.2"] },
    { "id": 4, "tasks": ["4.4", "4.5", "6.3"] },
    { "id": 5, "tasks": ["6.4", "6.5", "6.6", "7.1", "7.2"] },
    { "id": 6, "tasks": ["6.7", "6.8", "6.9", "6.10"] }
  ]
}
```
