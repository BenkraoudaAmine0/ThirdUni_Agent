# 🌙 Beginner's Guide to LangChain & Google Gemini AI

Welcome! This project is a simple, hands-on introduction to using **LangChain** with **Google's Gemini AI**. 

Even if you aren't a software developer, this guide will walk you through setting up everything step-by-step so you can start experimenting with AI agents, customized prompts, and conversational AI.

---

## 📌 What Does This Project Do?

This project uses a Jupyter Notebook (`.ipynb` file) to demonstrate:
1. **Connecting to Google Gemini:** Asking simple questions (like "What is the capital of the Moon?") and receiving AI responses.
2. **Adjusting AI Creativity:** Modifying "temperature" settings to make the AI give more consistent or more creative answers.
3. **Creating an AI Agent:** Setting up an AI that can keep track of chat history and remember context.
4. **Streaming Responses:** Outputting AI responses word-by-word in real time.
5. **System Prompts & Personas:** Giving the AI a specific personality (for example, pretending to be a science fiction writer).

---

## 🚀 Quick Start Guide

Follow these steps to get the project running on your computer.

### Step 1: Get a Google Gemini API Key
To allow your code to talk to Google's AI models, you need a free API key:

1. Open your browser and go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click on the button that says **"Get API key"** (or **"Create API key in new project"**).
4. Copy the long string of text/letters—this is your **API Key**. Keep it private!

---

### Step 2: Set Up Your Environment File (`.env`)

API keys should never be typed directly into your code. Instead, we store them in a hidden configuration file named `.env`.

1. Open your terminal, command prompt, or file manager in the project folder.
2. Duplicate or copy the template file named `.env.example` and rename the copy to `.env`:
   - **On Mac/Linux (Terminal):**
     ```bash
     cp .env.example .env
     ```
   - **On Windows (Command Prompt):**
     ```cmd
     copy .env.example .env
     ```
   - *Or manually:* Copy `.env.example`, paste it in the same folder, and rename it to `.env`.
3. Open the newly created `.env` file in any text editor (like Notepad or VS Code).
4. Replace `your_api_key` with the actual key you copied from Google AI Studio:
   ```env
   GOOGLE_API_KEY=AIzaSyYourActualKeyHere...
   ```
5. Save and close the file.

---

### Step 3: Install Dependencies

Make sure you have Python installed on your system. Open your terminal or command prompt and run the following command to install the required libraries:

```bash
pip install langchain langchain-google-genai python-dotenv
```

---

### Step 4: Run the Notebook

1. Launch Jupyter Notebook or open the project folder in **VS Code**.
2. Open the notebook file (`.ipynb`).
3. Run the cells sequentially from top to bottom!

---

## 🛠️ Requirements

* **Python:** 3.10 or higher recommended.
* **Libraries:** `langchain`, `langchain-google-genai`, `python-dotenv`.
* **API Key:** A free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/).