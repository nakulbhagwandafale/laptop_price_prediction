# 💻 Laptop Price Predictor

A Machine Learning project providing a sleek Streamlit web interface to accurately predict laptop prices (in INR) based on hardware specifications.

## ✨ Features
- ⚡ **Interactive Application**: Built entirely using Python and Streamlit.
- 📊 **Dynamic Input Handling**: Abstracted parameters like PPI (Pixels Per Inch) are calculated under the hood via the chosen Screen Resolution and Screen Size.
- 🌳 **RandomForest Regression**: Pipeline-based approach handling categorical variables natively and robustly preventing data leakage.

## 🗂️ Dataset
The model was trained on the `laptop_data.csv` dataset, learning from thousands of unique models spanning multiple form-factors and hardware components. Features included:
- **Company**: Apple, HP, Acer, Asus, Dell, Lenovo, etc.
- **Type**: Ultrabook, Gaming, Notebook, Netbook.
- **Processor**: Intel Core i3/i5/i7, AMD.
- **GPU & Storage**: Nvidia / AMD / Intel, combination of HDD & SSD.
- **RAM & Weight**

## 🚀 Quick Start
To run this application locally, you will need Python installed. We recommend using a virtual environment.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nakulbhagwandafale/laptop_price_prediction.git
   cd laptop_price_prediction
   ```
2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit server**:
   ```bash
   streamlit run app.py
   ```
This will spin up a local server and host the app simultaneously in your default web browser!

## 🤖 File Structure
- `app.py`: Contains the logic for the Streamlit Front-End Web Application.
- `train_model.py`: Training script converting dataset -> preprocessed columns -> RF estimator -> `pipe.pkl` pipeline target.
- `train.ipynb`: Jupyter notebook logging standard data exploration and dataset structuring methods.
- `df.pkl` & `pipe.pkl`: Binary pickles housing the clean DataFrame and the Scikit-learn Random Forest Estimator. 

## 🛠️ Built With...
- `scikit-learn`
- `pandas` & `numpy`
- `streamlit`
