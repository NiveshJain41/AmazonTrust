import google.generativeai as genai

# Configure API key
genai.configure(api_key="AIzaSyB1O1HaUkHmXrakakKzU2CkQBtRi-H5kBQ")  # Use your key

# Initialize Gemini Pro model
model = genai.GenerativeModel("gemini-2.0-flash")

# Prompt (you can ask for explanations or code)
prompt = "Write a Python function to check if a number is prime."

# Generate response
response = model.generate_content(prompt)

# Print the response
print(response.text)
