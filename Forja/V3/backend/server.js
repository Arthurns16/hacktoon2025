const express = require("express");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const cors = require("cors");
const dotenv = require("dotenv");
const OpenAI = require("openai");

const axios = require("axios");
const { GoogleGenAI } = require("@google/genai");

dotenv.config();
const app = express();
const port = 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Multer setup for file uploads (memory storage)
const upload = multer({ storage: multer.memoryStorage() });

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// (removed accidental top-level prompt declaration)
// Route: Extract text from multiple PDFs and process with Google Gemini (GoogleGenAI)
app.post("/extract-multi-file-data", upload.array("files"), async (req, res) => {
  try {
    if (!req.files || !Array.isArray(req.files) || req.files.length === 0) {
      return res.status(400).json({ error: "No files uploaded" });
    }
    if (!req.body.fields) {
      return res.status(400).json({ error: "All fields are important" });
    }

    const prompt = `Role: A meticulous and highly accurate data extraction assistant. you are designed to be a reliable tool for pulling specific, structured information from user-provided PDF documents data.
Core Objective: To accurately extract user-requested information from an uploaded PDF, format it into a csv format, and rigorously verify the output against the source document to eliminate hallucinations and ensure factual correctness.

2. Key Capabilities and Actions:

Information Scoping:  precisely understand what data points or fields need to be extracted from the document from user input and remove any comma from the value as it will ruin our csv structure.

Content Parsing: Analyze and parse the text, tables, and layout of the uploaded PDF to locate the relevant information.

Structured Data Generation: Organize and format the extracted information into a well-structured csv format.

Self-Correction and Verification: Before presenting the final output, perform a mandatory verification step. Re-scan the source PDF to cross-reference each entry in the generated csv formatted data, ensuring every piece of data directly corresponds to the content in the document.

Output Delivery: Present the final, verified csv data to the user.

I had attached the pdf content in last. 

3. Constraints and Limitations:

Boundaries:No Invention: The agent is strictly forbidden from inferring, predicting, or inventing information. If a requested piece of data cannot be found in the PDF, the corresponding csv cell must explicitly state 'NA'.
if there is any extra character like comma ',' in the single extracted value remove that character and provide only values without the comma ','.
Focus on Extraction: Do not provide analysis, interpretation, or summaries of the data unless explicitly requested by the user as a separate task after extraction.

One Document at a Time: The agent's context is limited to the single PDF document uploaded in the current session. It must not use information from previous documents or external sources.

Handle Unreadable Files: If a PDF is unreadable, password-protected, or image-based (without OCR capabilities), inform the user clearly that it cannot be processed.

Safety Protocols: Avoid processing documents that appear to contain highly sensitive personal information unless operating in a secure, compliant environment.

Escalation Protocol: If the user's request is ambiguous and cannot be clarified, or if the document structure is too complex to parse reliably, the agent should state its limitation clearly. E.g., 'The structure of this document is highly complex, and I cannot guarantee an accurate extraction for this request.'

5. Memory and Knowledge Management:

Short-Term Memory (Context Window): The agent must maintain the context of the user's specific data request throughout the extraction and verification process for a single document.

Long-Term Memory/Knowledge Base: The sole source of truth for any session is the uploaded PDF data. The agent's knowledge base is wiped clean upon the start of a new session or a new file upload.

6. Operational Guidelines:

Mandatory Process Flow: The agent must follow a strict operational sequence:Receive PDF from the user.

Clarify extraction requirements from the user input.

Perform initial data extraction into a draft csv.

Execute Verification Loop: Iterate through each cell of the draft csv and find the exact corresponding text in the source PDF. Flag any discrepancies.

Finalize the csv format data based on the verification step.

Present the final, verified csv data to the user.

Error Handling: If the verification step reveals significant discrepancies or low confidence in the extracted data, the agent should not present a potentially incorrect table. Instead, it should write NA in that cell?`;

    // Process all files in parallel
    const results = await Promise.all(req.files.map(async (file) => {
      try {
        const pdfData = await pdfParse(file.buffer);
        const rawText = pdfData.text;
        const response = await ai.models.generateContent({
          model: "gemini-2.5-flash",
          contents: `${prompt}. Here is the extracted PDF text:\n\n${rawText} please extract according to these specific data columns: ${req.body?.fields}`,
        });
        let result = response.text || JSON.stringify(response);
        return { filename: file.originalname, result };
      } catch (err) {
        return { filename: file.originalname, error: "Failed to extract PDF with Gemini" };
      }
    }));

    // Clean CSVs (remove code block markers)
    const cleanCsv = (csv) => {
      if (!csv) return '';
      if (csv.includes('```')) {
        const match = csv.match(/```(?:csv)?\s*([\s\S]*?)\s*```/i);
        if (match) return match[1].trim();
      }
      return csv.trim();
    };

    // Prepare all CSVs for the model
    const csvsWithFilenames = results.map(r => `Filename: ${r.filename}\n${cleanCsv(r.result)}`).join('\n\n');
    const combinePrompt = `You are a CSV data assistant. You will be given multiple CSVs, each extracted from a different PDF file. Each CSV is preceded by its filename. Combine all the CSVs into a single CSV table. Add a new column called 'filename' as the first column, and for each row, fill it with the corresponding filename. Ensure the header row appears only once at the top. If the columns differ, use all unique columns and leave missing values blank. Output only the final combined CSV, nothing else.\n\n${csvsWithFilenames}`;

    // Call the model to combine CSVs
    let combinedCsv = '';
    try {
      const combineResponse = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: combinePrompt,
      });
      combinedCsv = cleanCsv(combineResponse.text || JSON.stringify(combineResponse));
    } catch (combineErr) {
      console.error("Error combining CSVs with Gemini:", combineErr);
      // fallback: return empty string
      combinedCsv = '';
    }
    res.json({ results, combinedCsv });
  } catch (err) {
    console.error("Error extracting multiple PDFs with Gemini:", err);
    res.status(500).json({ error: "Failed to extract multiple PDFs with Gemini" });
  }
});


// Route: Extract text from PDF
app.post("/extract", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    if(!req.body.fields){
      return res.status(400).json({ error: "All fields are important" });
    }


    // Extract text from PDF
    const pdfData = await pdfParse(req.file.buffer);
    const rawText = pdfData.text;


    // Optional: Send extracted text to GPT for cleaning/analysis
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini", // or gpt-4o
      messages: [
        {
          role: "system",
          content: prompt,
        },
        {
          role: "user",
          content: `Here is the extracted PDF text:\n\n${rawText} please extract according to these specific data columns: ${req .body?.fields}`,
        },
      ],
    });

    const result = response.choices[0].message.content;

    res.json({ result });
  } catch (err) {
    console.error("Error extracting PDF:", err);
    res.status(500).json({ error: "Failed to extract PDF" });
  }
});

// Route: Extract text from PDF and process with Google Gemini (GoogleGenAI)
const ai = new GoogleGenAI({
    apiKey: "AIzaSyBmeVmC_utFE_va7pEij4-kqGKJQUbohwg"
});

// Start server
app.listen(port, () => {
  console.log(`✅ Server running at http://localhost:${port}`);
});
