# Language Agent for Games

## Usage
- `python main.py --log_root "log/main-pd/" --game "prisoners_dilemma" --test`
  - see `main.py` for more information

### Dependencies
- Use one of this
  - `conda create --name <env> --file requirements.txt`
  - `conda env create -f environment.yml -p <path-to-env>`

### API Key
- you should add a `.env` file under the repo root folder with the content `OPENAI_API_KEY=<your API KEY>`

## Implementation Detail
- Language Agent corresponds to an `AgentFactory` that has various prompting strategy
  - generate `Agent` codes to interact with the environments
- To add a new environment, simply add a new `EnvConfig`