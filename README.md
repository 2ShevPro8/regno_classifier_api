# License Plate Classification Service

## **1. Functionality Overview**

This service is built with **FastAPI** to perform **classification of vehicle license plates**. It receives data in **Base64 JSON** format, converts it into features, feeds it to the model, and returns probabilistic predictions as **Base64 JSON**.

To avoid loading the model each time from `pick_regno.py`, model loading is handled in `model.py`. Additionally, the `r` prefix is used in regular expressions within `pick_regno.py` for correct functionality.

### **Key Implementation Details**

**Why FastAPI:**

* High performance
* Convenient asynchronous request handling
* Easy integration with Pydantic for data validation

**pick\_regno.py:**

* Uses regular expressions to classify license plate templates (civilian, diplomatic, military, etc.)
* Calculates additional features for the model: max/min character, number of foreign letters, plate template type

**Testing:**

* Uses **pytest** with fixtures for model mocking
* Endpoint tests: `/`, `/health_check`, `/predict_b64`

**How to run tests:**

* With mock model:

  ```bash
  pytest -m test_api
  ```

  Add the `-s` flag and uncomment `print` in `test_api.py` to see predictions in the terminal.

* With the real model:

  ```bash
  pytest -m test_model
  ```

* Run all tests:

  ```bash
  pytest -s
  ```

---

## **2. Project Structure**

```
project/
├── app.py             # FastAPI endpoints
├── handler.py         # Handler class with pipe method for preprocessing and model inference
├── pick_regno.py      # Data transformation, feature extraction, and regex-based template classification
├── model.py           # ML model loading and initialization
├── test_data.csv      # Test dataset
├── tests/
│   ├── conftest.py    # Common pytest fixtures, including mock model
│   ├── test_api.py    # FastAPI endpoint tests
│   └── test_handler.py # Handler class tests
└── requirements.txt   # Project dependencies
```

---

## **3. How to Run the Service**

### **1. Create environment and install dependencies**

```bash
conda create -n env_name python=3.11
conda activate env_name
pip install -r requirements.txt
```

### **2. Start the server**

```bash
uvicorn app:app --reload
```

After launch, the endpoints will be available at:
`http://127.0.0.1:8000`

---

## **4. Example Curl Request (with Base64)**

The project includes **test\_request\_b64.txt** (contains the request in base64).

To send a request and get predictions:

```bash
curl -X POST "http://127.0.0.1:8000/predict_b64" --data-binary @test_request_b64.txt
```

### **How to decode a Base64 response in terminal:**

If it is **text** (string), run:

```bash
echo "string_b64" | base64 -d
```

---

## **5. Code Style and Static Analysis**

Code is verified using:

* **flake8** – PEP8 style checks:

  ```bash
  flake8 .
  ```

* **mypy** – static type checking:

  ```bash
  mypy .
  ```

