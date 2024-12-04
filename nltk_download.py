# nltk_download.py
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Specify custom download directory
nltk.download('punkt', download_dir='/Users/jesseelorddushime/nltk_data')
nltk.download('stopwords', download_dir='/Users/jesseelorddushime/nltk_data')

print("NLTK data downloaded successfully!")