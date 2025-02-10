# Marimo Viewer: Cirro
Visualization of data managed in the Cirro data platform


## Development

Set up your development environment:

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Launch the app in editable notebook format:

```
marimo edit app.py
```

Launch the app locally via HTML-WASM

```
rm -rf test_build;
marimo export html-wasm app.py -o test_build --mode run --show-code;
python -m http.server --directory test_build;
```
