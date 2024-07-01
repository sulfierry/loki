
# ChemBERTa Fine-Tuning with Spark

Este projeto realiza o fine-tuning do modelo pré-treinado `seyonec/ChemBERTa-zinc-base-v1` nos dados SMILES de kinases do ChEMBL utilizando Spark e PyTorch.

## Requisitos

- Python 3.8 ou superior
- `pip` (gerenciador de pacotes Python)

## Instalação

Siga as etapas abaixo para configurar um ambiente virtual Python e instalar as dependências do projeto.

### Passo 1: Clone o Repositório

Clone este repositório em sua máquina local usando o comando:

```bash
git clone https://github.com/username/repository.git
cd repository
```

### Passo 2: Crie um Ambiente Virtual

Crie um ambiente virtual para isolar as dependências do projeto:

```bash
python -m venv env
```

### Passo 3: Ative o Ambiente Virtual

Ative o ambiente virtual. O comando exato depende do seu sistema operacional:

#### Windows:

```bash
.\env\Scripts\activate
```

#### macOS e Linux:

```bash
source env/bin/activate
```

### Passo 4: Instale as Dependências

Instale as dependências do projeto listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt

sudo apt-get install graphviz

```

### Passo 5: Execute o Script

Depois de configurar o ambiente e instalar as dependências, você pode executar o script principal:

```bash
python chemberta_finetuner.py
```

## Estrutura do Projeto

- `chemberta_finetuner.py`: Script principal para realizar o fine-tuning do modelo.
- `requirements.txt`: Lista de pacotes necessários para executar o projeto.
- `README.md`: Instruções para configurar e executar o projeto.


