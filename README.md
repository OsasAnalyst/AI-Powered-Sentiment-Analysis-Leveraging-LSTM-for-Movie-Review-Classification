# **AI-Powered Sentiment Analysis: Leveraging LSTM for Movie Review Classification**  

---

## **Executive Summary**  

As a **Lead Data Scientist at a Media Analytics Firm**, my role is to develop **AI-driven solutions** that enhance the understanding of public sentiment across various domains. In the modern digital age, millions of users post reviews, feedback, and opinions online, influencing consumer behavior and industry trends. One of the most critical areas where sentiment analysis can be applied is the **entertainment industry**, where movie reviews significantly impact box office revenue, streaming platform ratings, and audience engagement.  

Traditional sentiment analysis methods rely on **rule-based** or **shallow machine learning approaches** that may fail to capture the **context, tone, and nuance** of user-generated text. To address this challenge, we leverage **Deep Learning techniques, specifically Long Short-Term Memory (LSTM) networks**, which are highly effective in processing **sequential data** like text.  

This project focuses on building an **LSTM-based Sentiment Analysis model** trained on the **IMDB dataset**, which contains **50,000 movie reviews** labeled as either **positive** or **negative**. The objective is to create an AI-powered solution capable of accurately classifying sentiments in movie reviews, demonstrating the **power of Natural Language Processing (NLP) in business intelligence, marketing strategies, and customer satisfaction analysis**.  

### **Key Highlights:**  
- **Develop an end-to-end pipeline**, from data preprocessing to model evaluation.  
- **Perform Exploratory Data Analysis (EDA)** to uncover key insights in review sentiments.  
- **Implement Tokenization, Padding, and Word Embedding** to transform textual data into a format suitable for LSTM networks.  
- **Train and optimize an LSTM model** to achieve high accuracy in sentiment prediction.  
- **Evaluate model performance** using appropriate metrics such as accuracy and loss.  
- **Demonstrate the practical applications of AI-powered sentiment analysis** in media analytics, brand monitoring, and customer feedback analysis.  

This project highlights the **business impact of sentiment analysis**, showcasing how AI can enhance decision-making in industries driven by consumer sentiment. Future advancements in this area could involve **transfer learning with pre-trained models**, **real-time sentiment tracking**, and **multilingual sentiment analysis** to expand its applicability across different industries.  

---

## **Objectives**  

The primary goal of this project is to **develop a deep learning model using LSTM** for sentiment analysis on **movie reviews**, ensuring high accuracy and reliable predictions.  

### **Specific Objectives:**  
1. **Develop a robust LSTM-based model** to classify movie reviews as **positive** or **negative**.  
2. **Preprocess and clean textual data**, including **tokenization, padding, and word embeddings** to enhance model learning.  
3. **Perform Exploratory Data Analysis (EDA)** to understand the distribution of sentiments in the dataset.  
4. **Optimize hyperparameters** (e.g., LSTM units, dropout rate, embedding dimensions) for improved model performance.  
5. **Evaluate the model's accuracy and performance** using appropriate validation techniques.  
6. **Apply the trained model for real-world sentiment analysis tasks**, such as analyzing **customer reviews, social media feedback, and brand reputation tracking**.  

---

## **Data Collection**  

The dataset used for this project is the **IMDB Large Movie Review Dataset**, a widely used benchmark dataset for sentiment analysis.  

### **Dataset Overview:**  
- **Source:** Kaggle (IMDB Dataset of 50K Movie Reviews)  
- **Number of Records:** 50,000 movie reviews  
- **Labels:**  
  - `1 = Positive Review`  
  - `0 = Negative Review`  
- **Text Data:** Raw textual reviews from IMDB users  

### **Data Preprocessing Steps:**  
1. **Loading the dataset** into a Pandas DataFrame for analysis.  
2. **Cleaning and tokenizing text data** to remove unnecessary symbols, punctuation, and stopwords.  
3. **Converting sentiment labels into numerical format** (`1` for positive and `0` for negative).  
4. **Splitting the dataset into training and test sets** (80% training, 20% testing).  
5. **Applying tokenization and padding** to prepare text for input into the LSTM model.  
6. **Using Word Embedding** to represent words in vector space, capturing their semantic meaning.  

---


## **Exploratory Data Analysis (EDA)**  

### **Checked for Missing Values**  
Before analyzing the dataset, it is essential to check for missing values that might affect the model's performance.  

```python
missing_values = data.isnull().sum()
print("Missing Values in Each Column:\n", missing_values)
```
The result showed that there is no missing value in the data

### **Sentiment Distribution (Positive vs. Negative)**  
To understand the balance of sentiment classes (positive and negative), i visualized their distribution.  

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(x=data["sentiment"], palette="coolwarm")
plt.title("Distribution of Sentiments")
plt.xlabel("Sentiment (0 = Negative, 1 = Positive)")
plt.ylabel("Count")
plt.show()
```

![Distribution of Sentiment](https://github.com/user-attachments/assets/ea37b7e4-8a22-4b7e-a56d-9bada476837a)

#### **Interpretation of Sentiment Distribution Plot**  
- The **bar chart** shows an equal distribution of **positive and negative** reviews.  
- The balanced dataset ensures that the model does not favor one class over the other, improving prediction accuracy.  

---

### **3. Review Length vs. Sentiment**  
To analyze whether review length influences sentiment, we calculate the number of words per review and visualize it using a **box plot**.  

```python
data["word_count"] = data["review"].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(8, 5))
sns.boxplot(x=data["sentiment"], y=data["word_count"], palette="coolwarm")
plt.title("Review Length vs. Sentiment")
plt.xlabel("Sentiment (0 = Negative, 1 = Positive)")
plt.ylabel("Number of Words in Review")
plt.show()
```

![Review Length vs Sentiment](https://github.com/user-attachments/assets/f3a4e5c8-70eb-497e-ae81-d539b1006045)


#### **Interpretation of Review Length vs. Sentiment**  
- The **box plot**  shows that **positive and negative reviews have slightly similar length distributions**.  
- However, **some positive reviews tend to be longer**, possibly because users provide detailed explanations when they like a movie.  
 
---


## **Model Development & Evaluation**

### **1. Data Preprocessing**
Before building the model, i preprocessed the dataset to ensure proper data structure and format.

```python
# Encoded target labels: Convert 'positive' to 1 and 'negative' to 0
data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

# Displayed the updated dataset to verify changes
data.head()
```
- **Encoding transformation** converted categorical labels to numerical values (0 = Negative, 1 = Positive).

---

### **2. Splitting Data into Training & Testing Sets**
The dataset was splitted into training and testing sets in an **80:20 ratio**.

```python
# Splitted Data into Training and Testing Sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Displayed the size of the training and testing sets
print(f"Training Data Shape: {train_data.shape}")
print(f"Testing Data Shape: {test_data.shape}")
```

---

### **3. Text Tokenization & Padding**
The reviews were converted into sequences of numbers using **tokenization** and then padded to a uniform length of **200 words**.

```python
tokenizer = Tokenizer(num_words=5000)  # Limit vocabulary size to 5000 words
tokenizer.fit_on_texts(train_data["review"])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)

# Target labels (sentiments)
Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]
```

---

### **4. Building the LSTM Model**
A **Long Short-Term Memory (LSTM) neural network** was designed for sentiment classification.

```python
model = Sequential([
    Embedding(input_dim=5000, output_dim=120, input_length=200),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation="sigmoid")
])

# Built the model explicitly
model.build(input_shape=(None, 200))

# Displayed the model architecture
model.summary()
```

- **Embedding Layer**: Converts words into dense vector representations.
- **LSTM Layer**: Captures long-term dependencies in the text.
- **Dense Layer**: Uses a **sigmoid activation** to classify sentiment.

---

### **5. Compiling & Training the Model**
The model was compiled with the **Adam optimizer** and **binary cross-entropy loss**, suitable for binary classification tasks.

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Trained the model for 5 epochs with a batch size of 64
model.fit(
    X_train, Y_train, 
    epochs=5, 
    batch_size=64, 
    validation_split=0.2
)
```

- **Optimizer**: Adam, known for adaptive learning rates.
- **Loss Function**: Binary cross-entropy to measure the error in classification.
- **Epochs**: 5, to prevent overfitting while allowing sufficient learning.

---

## **Model Evaluation**
The trained model was evaluated on the test dataset.

```python
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
```
![Model Evaluation](https://github.com/user-attachments/assets/6b5fa253-8f1e-4e92-956d-ec228a2ca4fa)

### **Model Evaluation Result**  
The LSTM model achieved an **accuracy of 88.2%** and a **test loss of 0.317**, indicating strong performance in classifying sentiments with minimal error.  

---

## **Findings**
1. **The LSTM model achieved high accuracy**, demonstrating its effectiveness in sentiment classification.
2. **The dataset was well-balanced**, contributing to the model’s performance.
3. **Tokenization and sequence padding were crucial** in structuring the data for LSTM processing.

---

## **Recommendations Based on Findings**
1. **For better performance**, additional preprocessing steps like **removing stopwords, lemmatization, and handling negations** should be explored.  
2. **Increasing the vocabulary size** beyond **5,000 words** may enhance model understanding.  
3. **Experimenting with different deep learning architectures**, such as **bidirectional LSTMs or transformer models**, can further improve accuracy.  
4. **Hyperparameter tuning** (e.g., adjusting the **LSTM units, dropout rates, or batch sizes**) may optimize results.  

---

## **Limitations** 
1. **Handling sarcasm and complex sentiments**: The model might struggle with detecting sarcasm or ambiguous reviews.  
2. **Computational cost**: Training deep learning models requires high computational power.  

---

## **Future Work**
1. **Integrate transformer-based models like BERT**, which have shown superior performance in NLP tasks.  
2. **Expand the dataset** to include **more diverse and multilingual reviews** to test generalizability.  
3. **Fine-tune hyperparameters** to improve accuracy and reduce training time.
