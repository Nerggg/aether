python3 -m venv venv
mkdir data/rules
cd data/rules
git clone https://github.com/downfallx/dnd-5e-srd-markdown
cd dnd-5e-srd-markdown
rm -rf .git
pip install sentence-transformers chromadb ollama pydantic
