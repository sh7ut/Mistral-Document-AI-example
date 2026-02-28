## Problem Statement

Partner Request: "We have a Tier-1 client looking to transition from legacy OCR to a GenAI-native workflow. They need to process high volumes of complex PDFs—specifically multi-page financial statements containing nested tables and embedded images—for automated entity extraction and risk classification.
How can we leverage Mistral’s Document AI and AI Studio to build a production-ready pipeline that manages these visual complexities while maintaining data integrity? Please walk us through the target architecture and provide a code-level implementation of the extraction-to-classification flow. Additionally, explain how you would recommend we 'Judge' the output quality at scale.” 
Your solution must utilize the Mistral native stack.


## Check the latest Docs for Document AI - OCR Processor 
https://docs.mistral.ai/capabilities/document_ai/basic_ocr

## There is an example for opening files uploaded on Mistral Cloud

First, you will have to upload your PDF file to our cloud, this file will be stored and only accessible via an API key.

```python
from mistralai import Mistral
import os

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

uploaded_pdf = client.files.upload(
    file={
        "file_name": "2201.04234v3.pdf",
        "content": open("2201.04234v3.pdf", "rb"),
    },
    purpose="ocr"
)  
```

Once the file uploaded, you can retrieve it at any point.

```python
retrieved_file = client.files.retrieve(file_id=uploaded_pdf.id)
```

For OCR tasks, you can get a signed url to access the file. An optional expiry parameter allow you to automatically expire the signed url after n hours.

```python
signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
```

You can now query the OCR endpoint with the signed url.

```python
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": signed_url.url,
    },
    table_format="html", # default is None
    # extract_header=True, # default is False
    # extract_footer=True, # default is False
    include_image_base64=True
)
```

Once all OCR done, you can optionally delete the pdf file from our cloud unless you wish to reuse it later.

```python
client.files.delete(file_id=file.id)
```
The output will be a JSON object containing the extracted text content, images bboxes, metadata and other information about the document structure.

```json
{
  "pages": [ # The content of each page
    {
      "index": int, # The index of the corresponding page
      "markdown": str, # The main output and raw markdown content
      "images": list, # Image information when images are extracted
      "tables": list, # Table information when using `table_format=html` or `table_format=markdown`
      "hyperlinks": list, # Hyperlinks detected
      "header": str|null, # Header content when using `extract_header=True`
      "footer": str|null, # Footer content when using `extract_footer=True`
      "dimensions": dict # The dimensions of the page
    }
  ],
  "model": str, # The model used for the OCR
  "document_annotation": dict|null, # Document annotation information when used, visit the Annotations documentation for more information
  "usage_info": dict # Usage information
}
```

## Others
If you need the Mistral API keys, they are stored in my `~/.zshrc` file. 
Never use the keys in the code for security. 

Use this snippet in the code:

```python
# Initialize Mistral client
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set. Please set it before running.")

client = MistralClient(api_key=api_key)
print("✓ Mistral client initialized")
```



