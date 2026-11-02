GEN AI & OPEN AI PRACTICAL CODES (17 PROGRAMS)

Each experiment is saved in a separate .py file. These are based on your practical-record PDF.
I corrected obvious syntax/incomplete-code issues so the files can be run, and replaced the API key printed in the PDF with the GROQ_API_KEY environment variable.

1) Install Python 3.10+ and open a terminal in this folder.
2) For experiments 1-10: python -m pip install openai
3) Set your Groq key before running those experiments. Windows PowerShell: $env:GROQ_API_KEY="YOUR_NEW_GROQ_API_KEY"
   Windows CMD: set GROQ_API_KEY=YOUR_NEW_GROQ_API_KEY
   macOS/Linux: export GROQ_API_KEY="YOUR_NEW_GROQ_API_KEY"
4) Run one file at a time, for example: python 01_simple_chatbot.py
5) For experiments 11-14: python -m pip install tensorflow numpy matplotlib
6) For experiments 15-17: python -m pip install transformers datasets torch
7) GPT-2 downloads model files the first time. TensorFlow GAN experiments may take a long time and work best with a GPU.

IMPORTANT SECURITY NOTE: The PDF contains a Groq API key in plain text. Do not reuse it. Revoke/rotate it in your Groq account and create a new key. Never publish API keys inside code or PDFs.

Note: Experiments 12-14 are faithful, runnable implementations of the PDF's stated programs; some syntax in the PDF extraction was incomplete or malformed, so those sections were repaired.
