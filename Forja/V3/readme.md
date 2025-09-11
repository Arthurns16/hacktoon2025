# Nexa AI PDF Data Extractor

This project allows you to upload a PDF and extract specific data points using an AI backend (Google Gemini). You can specify which fields to extract (e.g., "Client Name, Session Date, Chief Complaint") and receive the results in a structured format.

## Prerequisites

- Node.js (v18 or higher recommended)
- npm (comes with Node.js)

## 1. Install Dependencies

Open a terminal in the project root and run:

```
cd backend
npm install
cd ../frontend
npm install
```

## 2. Set Up Environment Variables

In the `backend` folder, create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_gemini_key_here
```

## 3. Start the Backend Server

In the `backend` folder, run:

```
npm start
```

The backend will start on [http://localhost:5000](http://localhost:5000).

## 4. Start the Frontend

In the `frontend` folder, run:

```
npm run dev
```

The frontend will start on [http://localhost:5173](http://localhost:5173) (or similar).

## 5. Usage

1. Open the frontend in your browser.
2. Upload a PDF file.
3. In the input field, specify the data points you want to extract (e.g., "Client Name, Session Date, Chief Complaint").
4. Click "Extract Data".
5. The extracted data will appear in the results section.

## Notes
- Make sure your API keys are valid and have sufficient quota.
- The backend uses Google Gemini for extraction. You can switch to OpenAI or Hugging Face by modifying the backend code.
- If you encounter errors, check the terminal output for details.

## Troubleshooting
- If you get CORS errors, ensure both frontend and backend are running on localhost and the backend has CORS enabled.
- If PDF extraction fails, make sure the PDF is not password-protected or image-only (no OCR support).

---

For further customization or issues, please refer to the code comments or open an issue.
