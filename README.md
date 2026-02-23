Spam Email Classifier using AI & ML

This project is a **Spam Email Classification System** built using **Python, Machine Learning, and Natural Language Processing (NLP)**.  
It classifies messages as **Spam** or **Ham (Not Spam)** using supervised learning algorithms.


Features
- Preprocesses text using NLP techniques  
- Converts text into numerical features using Bag-of-Words  
- Trains model using **Naive Bayes** and **Logistic Regression**  
- Predicts whether an email/message is spam or not  
- Displays accuracy and evaluation metrics  


Algorithms Used
- Naive Bayes Classifier  
- Logistic Regression  


Dataset
- SMS Spam Collection Dataset  
- Contains labeled messages as `spam` or `ham`  


Technologies Used
- Python  
- Pandas  
- Scikit-learn  
- NLP (CountVectorizer)  

Project Structure
SpamClassifier/
│
├── clsfier.py
├── spam.csv
└── README.md


How to Run the Project

1. Clone the repository
bash
git clone https://github.com/MuthuKarthik033/SpamClassifier.git
cd SpamClassifier

pip install pandas scikit-learn

python clsfier.py
