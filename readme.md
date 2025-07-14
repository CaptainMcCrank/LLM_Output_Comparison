# LLM Comparison Tool

A Python script that simultaneously queries multiple Large Language Model (LLM) APIs with the same prompt and compares their responses. Currently supports ChatGPT (OpenAI), Claude (Anthropic), and Gemini (Google).

## Features

- **Multi-Provider Support**: Query ChatGPT, Claude, and Gemini with a single command
- **Flexible Input**: Accept prompts from command line, user input, or markdown files
- **Rich Output Options**: Save responses to timestamped markdown files or display in console
- **Automatic HTML Generation**: Convert markdown responses to HTML and open in browser
- **Detailed Metadata**: Track response times, token usage, model versions, and more
- **Visual Progress**: Animated status indicators and colored console output
- **Error Handling**: Graceful timeout handling and clear error reporting

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Git
- Pandoc (for HTML conversion)

### API Keys Required
You'll need API keys from:
- **OpenAI** (for ChatGPT)
- **Anthropic** (for Claude)
- **Google AI** (for Gemini)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/llm-comparison-tool.git
cd llm-comparison-tool
```

### 2. Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with:
```
openai>=1.0.0
anthropic>=0.3.0
google-generativeai>=0.3.0
```

### 4. Install Pandoc (for HTML conversion)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install pandoc
```

**macOS:**
```bash
brew install pandoc
```

**Windows:**
Download and install from [pandoc.org](https://pandoc.org/installing.html)

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` with your API keys:
```bash
# OpenAI API Key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-your-openai-key-here

# Anthropic API Key (get from https://console.anthropic.com/)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Google AI API Key (get from https://aistudio.google.com/app/apikey)
GOOGLE_API_KEY=your-google-ai-key-here
```

### 6. Load Environment Variables

Add this to your shell profile (`.bashrc`, `.zshrc`, etc.) or run before each session:

```bash
# Load environment variables
set -a
source .env
set +a
```

Or install python-dotenv and modify the script to load automatically:
```bash
pip install python-dotenv
```

## Getting API Keys

### OpenAI (ChatGPT)
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new secret key
5. Copy the key (starts with `sk-`)

### Anthropic (Claude)
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Go to API Keys section
4. Generate a new API key
5. Copy the key (starts with `sk-ant-`)

### Google AI (Gemini)
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with Google account
3. Create a new API key
4. Copy the generated key

## Usage

### Basic Commands

**Command line prompt:**
```bash
python llm_compare.py "What is artificial intelligence?"
```

**Interactive prompt:**
```bash
python llm_compare.py
# You'll be prompted to enter your question
```

**From markdown file:**
```bash
python llm_compare.py my-prompt.md
```

**Console output (no files):**
```bash
python llm_compare.py "Explain quantum computing" --console
# or
python llm_compare.py query.md -c
```

### Command Line Options

- `--console, -c`: Display responses in console instead of saving to files
- `--help, -h`: Show help message

### Input Methods

1. **Direct text**: `python llm_compare.py "Your question here"`
2. **Markdown file**: `python llm_compare.py prompt.md`
3. **Interactive**: `python llm_compare.py` (will prompt for input)

### Output Files

When not using `--console`, the script creates:

**Markdown files** (timestamped):
- `ChatGPT_Response_YYYYMMDD_HHMMSS.md`
- `Claude_Response_YYYYMMDD_HHMMSS.md`
- `Gemini_Response_YYYYMMDD_HHMMSS.md`

**HTML files** (auto-generated and opened):
- `ChatGPT_Response_YYYYMMDD_HHMMSS.html`
- `Claude_Response_YYYYMMDD_HHMMSS.html`
- `Gemini_Response_YYYYMMDD_HHMMSS.html`

Each file includes:
- Query metadata (model, timestamp, response time, tokens)
- Source information (file path, MD5 hash, modification date)
- Full response content

## Example Workflow

1. **Create a prompt file:**
```bash
echo "Compare Python and JavaScript for web development" > comparison.md
```

2. **Run the comparison:**
```bash
python llm_compare.py comparison.md
```

3. **View the results:**
The script will automatically:
- Query all three LLM providers
- Save responses to timestamped markdown files
- Convert to HTML using pandoc
- Open HTML files in your default browser

4. **Quick console test:**
```bash
python llm_compare.py "What's 2+2?" --console
```

## Troubleshooting

### Common Issues

**"ModuleNotFoundError":**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

**"API Key not found":**
- Check `.env` file exists and has correct keys
- Verify environment variables are loaded: `echo $OPENAI_API_KEY`

**"Pandoc not found":**
- Install pandoc using system package manager
- Script will still work but won't generate HTML

**Long response times:**
- Default timeout is 120 seconds
- Check your internet connection
- Some complex prompts may take longer

**"Model not found" errors:**
- Verify your API keys have access to the specified models
- Check API quotas and billing status

### Debug Mode

For verbose output, modify the script or add debug prints to troubleshoot API issues.

## Customization

### Adding New Providers
To add support for additional LLM providers:

1. Install the provider's Python SDK
2. Add configuration to the imports section
3. Create a new `query_provider()` function following the existing pattern
4. Add the provider to the `providers` list in `main()`

### Modifying Models
Edit the model names in each query function:
- ChatGPT: Change `model = "gpt-4o-mini"`
- Claude: Change `model = "claude-3-5-sonnet-20241022"`
- Gemini: Change `model_name = "gemini-2.0-flash-exp"`

### Adjusting Parameters
Modify temperature, max_tokens, or other parameters in each query function.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for ChatGPT API
- Anthropic for Claude API  
- Google for Gemini API
- Pandoc for markdown to HTML conversion