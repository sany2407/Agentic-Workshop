# 🏪 Clothing Store Competitor Analysis AI

A conversational AI assistant built with **LangGraph** and **Google Gemini** that helps clothing store owners analyze their competition and make data-driven business decisions.

## 🚀 Features

**Conversational Interface**: Natural language queries about competitors
**Real-time Analysis**: Instant competitor research and market insights
**Footfall Analytics**: Peak hours, customer traffic patterns, and trends
**Market Intelligence**: Competition level, saturation analysis, and opportunities
**Strategic Recommendations**: Actionable insights for business growth
**Location-based**: Customizable analysis for any area

## 🛠️ Technology Stack

**LangGraph**: State management and workflow orchestration
**Google Gemini**: Large language model for conversational AI
**Streamlit**: Web interface and chat functionality
**LangChain**: Tool integration and message handling

## 📦 Installation

> **Note:** It is strongly recommended to use a Python virtual environment (venv) to isolate project dependencies and avoid conflicts with other Python packages on your system.

1. **Create and Activate Virtual Environment**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

2. **Clone the repository**

```bash
git clone https://github.com/sany2407/Agentic-Workshop.git
cd ConversationalAi
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Get Gemini API Key**

   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for use in the app

5. **Run the application**

```bash
streamlit run app.py
```

## 🎯 Usage

1. **Start the app** and enter your Gemini API key
2. **Set your location** (e.g., "Koramangala, Bangalore")
3. **Ask natural questions** like:
   - "Who are my main competitors?"
   - "When do stores get the most customers?"
   - "What's the market saturation in my area?"
   - "Give me recommendations for my clothing store"

## 💬 Example Conversations

**User**: "Who are my competitors in Koramangala?"

**AI**: "I found 3 main competitors in Koramangala:

1. **Fashion Hub** (4.2⭐)

📍 80 Feet Road, Koramangala*
💰 Price range: ₹₹₹*
🏷️ Specialties: Casual wear, Ethnic wear\*

Would you like me to analyze their footfall patterns?"

**User**: "What are the peak hours for shopping?"

**AI**: "Based on competitor analysis:

📊 **Peak Hours:** 11:00-13:00, 18:00-20:00
📈 **Busiest Days:** Saturday, Sunday
💹 **Total Area Footfall:** 1,450 customers/day

This suggests lunch hours and evening times are optimal for promotions!"

## 🏗️ Project Structure

├── app.py # Main application file
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## 🧠 How It Works

### **LangGraph Workflow**

1. **Intent Analysis**: Determines what the user is asking about
2. **Data Retrieval**: Fetches competitor information when needed
3. **Conversational Response**: Generates natural, contextual replies

### **Agent State Management**

Maintains conversation history
Stores competitor data between queries
Tracks analysis completion status
Preserves user context and location

### **Tools Available**

search_competitor_stores(): Finds nearby clothing stores
analyze_footfall_trends(): Calculates traffic patterns and insights

## 👥 Target Users

**🏢 Business Owners \& Managers**: Market positioning and competitive analysis
**📈 Marketing Teams**: Campaign timing and audience insights
**🏗️ Real Estate Analysts**: Location assessment and site selection
**💼 Investors**: Market potential evaluation and risk analysis

## 📊 Data Sources

Currently uses mock data for demonstration. In production, integrate with:

Google Places API for real store data
Foot traffic analytics services (Placer.ai, SafeGraph)
Business intelligence platforms
Social media APIs for engagement metrics

## 🔧 Configuration

### **Environment Variables**

bash
export GEMINI_API_KEY="your-api-key-here"

### **Customization**

Modify competitor data in search_competitor_stores()
Adjust analysis algorithms in analyze_footfall_trends()
Update conversation templates in conversational_response()

## 📈 Future Enhancements

[ ] Real-time data integration with Google Places API
[ ] Advanced analytics with seasonal trends
[ ] Multi-language support
[ ] Export reports to PDF/Excel
[ ] Integration with business intelligence tools
[ ] Mobile-responsive design
[ ] User authentication and data persistence


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions or issues:

Create an issue in this repository

## 🙏 Acknowledgments

**Google AI** for Gemini API
**LangChain** for the framework
**Streamlit** for the web interface
**LangGraph** for state management

**Built with ❤️ for small business owners who want to compete smarter, not harder.**

## 🚀 Quick Start

bash
git clone <repo>
pip install -r requirements.txt
streamlit run app.py

Enter your Gemini API key → Set location → Start chatting!
