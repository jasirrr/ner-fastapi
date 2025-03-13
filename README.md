# 🚀 NER FastAPI Project

This project implements a **Named Entity Recognition (NER)** model using **FastAPI**. It includes data preprocessing, model training, and deployment as a REST API using Docker.

Here is the deployed API -- https://ner-fastapi.onrender.com

---

## 📁 **Project Structure**

ner-fastapi/
├── app.py                  # FastAPI main application file
├── train.py                # Training script for NER model
├── preprocess.py           # Preprocessing script for NER data
├── ner_crf_model.pkl       # Trained NER model file
├── requirements.txt        # List of dependencies
├── Dockerfile              # Docker configuration file
├── .env                    # Environment variables
├── data/                   # Data files (training/testing)
├── __pycache__/            # Cached Python files


---

## 🏗️ **Setup & Installation**
### 1. **Clone the repository**  
```bash
git clone https://github.com/jasirrr/ner-fastapi.git
cd ner-fastapi

2. Set up a virtual environment (optional)

python -m venv venv
source venv/bin/activate   # Linux/macOS
.\venv\Scripts\activate    # Windows


3. Install dependencies

pip install -r requirements.txt


🐳 Docker Setup
1. Build Docker Image

docker build -t ner-fastapi .

2. Run Docker Container
docker run -p 8000:8000 ner-fastapi


🌐 API Endpoints

Method	Endpoint	Description
POST	/predict	Perform NER on input text
GET	/health	Check API health status


🧪 Example Request
POST /predict
Request:
{
  "text": "Elon Musk is the CEO of Tesla."
}

Response:

{
  "entities": [
    {
      "entity": "PERSON",
      "value": "Elon Musk"
    },
    {
      "entity": "ORG",
      "value": "Tesla"
    }
  ]
}


📖 How It Works
Preprocessing

Lowercasing
Stopword removal
Lemmatization
Tokenization
Training

Uses Conditional Random Fields (CRF) for NER
Spacy model integration
Prediction

Predict entities in the input text using the trained model

✅ Environment Variables
Create a .env file:
API_USERNAME=admin
API_PASSWORD=secret

🚀 Deploy to Render
Push the repo to GitHub
Create a new service on Render
Connect the repo and set the build command:
bash
Copy
Edit
docker build -t ner-fastapi .
Start the service 🚀
🎯 Future Improvements
Improve model accuracy using transformers
Add more entity types
Implement authentication
🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

💡 Author
Ahamed Jasir V

📜 License
This project is licensed under the MIT License.
