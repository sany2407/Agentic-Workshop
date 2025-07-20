Here's a professional and comprehensive **README.md** file for your **Agentic EDA with Gemini + Autogen** Streamlit application:

---

# 🔍 Agentic EDA with Gemini + Autogen

A multi-agent data analysis system powered by [Google Gemini](https://ai.google) and [Autogen](https://github.com/microsoft/autogen), wrapped in an interactive [Streamlit](https://streamlit.io) interface. Upload a CSV dataset and let AI agents handle data cleaning, exploratory data analysis (EDA), reporting, and critique automatically.

---

## 🚀 Features

* 🤖 **Multi-agent Architecture** using Autogen
* 📊 **EDA Automation**: Summary stats, insights & chart suggestions
* 🧹 **Data Cleaning**: Handle missing values, fix types, remove duplicates
* 📄 **Report Generation**: AI-generated human-readable EDA reports
* 🧐 **Critic Review**: Automated feedback on the report quality
* ✅ **Code Validation**: Preprocessing code verification and improvement suggestions
* 🧠 **Gemini 1.5 Flash API** for fast, intelligent content generation
* 🌐 **Streamlit UI** for an intuitive and interactive experience

---

## 📁 Project Structure

```
├── agentic_eda_app.py       # Main Streamlit app
├── .env                     # Environment variables (e.g., Gemini API key)
├── requirements.txt         # Python dependencies
└── README.md                # You're reading it!
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/agentic-eda-gemini.git
cd agentic-eda-gemini
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add `.env` File

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

---

## ▶️ Run the App

```bash
streamlit run agentic_eda_app.py
```

Visit `http://localhost:8501` in your browser to access the app.

---

## 🤖 Agents Overview

| Agent Name               | Role Description                                                                          |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| **DataPrepAgent**        | Cleans the dataset by handling missing values, fixing data types, and removing duplicates |
| **EDAAgent**             | Extracts key insights, statistics, and visual suggestions                                 |
| **ReportGeneratorAgent** | Generates a detailed EDA report                                                           |
| **CriticAgent**          | Reviews the report and suggests improvements                                              |
| **ExecutorAgent**        | Validates the preprocessing code                                                          |
| **Admin (Proxy)**        | Orchestrates the conversation among agents                                                |

---

## 📝 Example Output

* 🧹 **Preprocessing Code**
* 📊 **Insights & Visual Suggestions**
* 📄 **EDA Report**
* 🧐 **Critique of Report**
* ✅ **Validation of Preprocessing Logic**

All of the above are shown in a user-friendly expandable Streamlit interface.

---

## 📌 Requirements

* Python 3.8+
* Google Gemini API Key
* Streamlit
* Pandas
* dotenv
* Autogen (Microsoft)

Install them via:

```bash
pip install -r requirements.txt
```

---

## ✅ To-Do / Enhancements

* [ ] Add real-time chart rendering (Matplotlib / Plotly)
* [ ] Enable execution of preprocessing code
* [ ] Store session reports
* [ ] Export EDA reports to PDF

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

---

