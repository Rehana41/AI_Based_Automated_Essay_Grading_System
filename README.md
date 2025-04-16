# 📚 AI-Based Essay Grading System
## 📌 Overview
The AI-Based Essay Grading System is a Deep Learning-powered project that evaluates and predicts scores for student essays. This system leverages a BiLSTM (Bidirectional Long Short-Term Memory) model to assess essay quality based on content, grammar, and structure. It is trained on a curated academic essay dataset and can be easily extended for real-world automated scoring use cases.

## 🚀 Features
- **Automated Scoring**: Predicts essay scores on a scale from 0 to 10.

- **BiLSTM Architecture**: Uses a deep learning model capable of capturing contextual patterns in textual data.

- **Flexible Dataset Input**: Works with .csv essay files, with two columns: essay and score.

- **Modular Design**: Clear separation of data processing, model training, and evaluation for easy customization.

## 🛠️ Technologies Used
- **Python**: Core programming language.

- **PyTorch**: For building and training the BiLSTM model.

- **Pandas & NumPy**: For data manipulation.

- **Sklearn**: For evaluation (MSE, train-test splitting).

- **Jupyter Notebook / Colab**: For running and visualizing training progress.

## 🗂️ Dataset
Format: CSV file with two columns: essay (text) and score (float/int).
Dataset link(essays.csv):- https://drive.google.com/file/d/1v-iKbOcuJc4McFXWca5o-ntGCtQUbDe-/view?usp=drive_link

## 🧠 Model Architecture
- **Embedding Layer**: Converts ASCII-tokenized characters to vector embeddings.

- **BiLSTM Layer**: Captures both forward and backward textual dependencies.

- **Fully Connected Output Layer**: Outputs a continuous score between 0 and 10.

- **Activation**: Sigmoid scaled to 0–10 range.

## 🖼️ Workflow
**Preprocessing**:

Essay text is tokenized via a simple ASCII-based tokenizer.

Padding applied to maintain fixed length sequences.

**Training**:

BiLSTM model trained using MSELoss and Adam optimizer.

Validation set used to evaluate model performance.

**Evaluation**:

Mean Squared Error (MSE) and prediction samples are printed for monitoring.

Predicted scores are compared to actual scores for accuracy insights.

**Model Saving**:

Trained model is saved to models/bilstm_essay_model.pt.

## 📊 Model Performance
- **Loss Function**: Mean Squared Error (MSE)

- **Output Range**: Continuous score from 0 to 10

- **Typical Accuracy**: Depends on dataset quality and size

- **Training Time**: Varies by dataset size

## 🌐 How to Run

### ✅ Steps
```bash
Place your essays.csv file into data/processed/ as data/processed/essays 

Install requirements: pip install torch pandas sklearn

Run training:
python src/train_model.py

Predictions: After training, use the model for prediction
python webapp/app.py
```
### 🔧 Example Input (from CSV)

```bash
essay,score 
"Mobile phones can help students study more efficiently...",8.0
"Phone is good. Phone bad. School maybe yes or no phone...",3.5
```







