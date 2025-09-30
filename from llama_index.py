import os
OPENAI_API_KEY="AIzaSyAAtuLyKbQsK1sKu3_EOkoqqP76KP7R2X4"
OPENAI_API_BASE="https://gemini.google.com"

# laod environment variables from .env file
import dotenv
dotenv.load_dotenv()

def test_use_api_key():
    from gptpdf import parse_pdf
    pdf_path = '24_all.pdf'
    output_dir = '../output/'
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_API_BASE')
    # Manually provide OPENAI_API_KEY and OPEN_API_BASE
    content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, api_key=api_key, base_url=base_url, model='gemini-pro', gpt_worker=6)
    print(content)
    print(image_paths)
    # also output_dir/output.md is generated


def test_use_env():
    from gptpdf import parse_pdf
    pdf_path = '24_all.pdf'
    output_dir = '../output/'
    # Use OPENAI_API_KEY and OPENAI_API_BASE from environment variables
    content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, model='gemini-pro', verbose=True)
    print(content)
    print(image_paths)
    # also output_dir/output.md is generated

def test_azure():
    from gptpdf import parse_pdf
    api_key = "AIzaSyAAtuLyKbQsK1sKu3_EOkoqqP76KP7R2X4" # Azure API Key
    base_url = "https://gemini.google.com"
 # Azure API Base URL
    model = 'gemini-pro' # azure_ with deploy ID name (not open ai model name), e.g. azure_cpgpt4

    pdf_path = '24_all.pdf'
    output_dir = '../output/'
    # Use OPENAI_API_KEY and OPENAI_API_BASE from environment variables
    content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, api_key=api_key, base_url=base_url, model=model, verbose=True)
    print(content)
    print(image_paths)
    



if __name__ == '__main__':
    # test_use_api_key()
    # test_use_env()
    test_azure()