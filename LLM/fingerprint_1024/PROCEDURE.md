### Explicação das Mudanças

1. **Removemos o Autoencoder:** Simplificamos o código removendo o autoencoder e focamos apenas no fine-tuning do modelo `seyonec/ChemBERTa-zinc-base-v1`.
2. **Barra de Progresso com `tqdm`:** Adicionamos a barra de progresso `tqdm` nos loops de treinamento e avaliação para monitorar o progresso.
3. **Função de Perda:** Utilizamos `nn.CrossEntropyLoss` para calcular a perda durante o treinamento.

### `PROCEDURE.md` Atualizado

```markdown
# Procedimento de Fine-Tuning para ChemBERTa

Este documento descreve o procedimento detalhado para realizar o fine-tuning do modelo pré-treinado `seyonec/ChemBERTa-zinc-base-v1` utilizando dados SMILES de kinases do ChEMBL.

## Descrição Geral

O fine-tuning é realizado para adaptar o modelo pré-treinado ChemBERTa aos dados específicos de kinases, melhorando a capacidade do modelo de prever atividades específicas de compostos químicos. O procedimento envolve várias etapas, desde o carregamento e particionamento dos dados até o treinamento e avaliação do modelo.

## Carregamento e Preparação dos Dados

1. **Carregamento dos Dados:**
   - Os dados são carregados a partir de um arquivo Parquet (`train_data.parquet`) produzido pela classe `FormatFileML` usando Spark.

2. **Conversão para Pandas:**
   - Os dados carregados com Spark são convertidos para um DataFrame Pandas para facilitar o uso com PyTorch DataLoader.

3. **Particionamento dos Dados:**
   - Os dados são divididos em conjuntos de treino (80%) e teste (20%) usando `train_test_split` do scikit-learn.
   - A divisão garante que a avaliação do modelo seja feita em dados não vistos durante o treinamento.

## Estrutura do Modelo

1. **Tokenização:**
   - As SMILES são tokenizadas usando `RobertaTokenizer` do `transformers` para preparar os dados de entrada para o modelo BERT.

2. **Modelo BERT:**
   - O modelo `RobertaModel` pré-treinado é carregado do

 `transformers`.

## Procedimento de Treinamento

1. **Configuração:**
   - O treinamento é realizado em um dispositivo GPU se disponível, caso contrário, em CPU.
   - Otimizador AdamW é utilizado com uma taxa de aprendizado inicial de `5e-5`.
   - Agendador de taxa de aprendizado linear é usado para ajustar a taxa de aprendizado durante o treinamento.

2. **Treinamento:**
   - O modelo é treinado por 10 épocas, iterando sobre os lotes de dados de treino.
   - Para cada lote, a perda é calculada usando `cross_entropy` e o otimizador ajusta os pesos do modelo.
   - A barra de progresso `tqdm` é utilizada para monitorar o progresso do treinamento.

## Procedimento de Avaliação

1. **Avaliação no Conjunto de Teste:**
   - Após o treinamento, o modelo é avaliado no conjunto de teste.
   - As previsões são comparadas com os rótulos reais para calcular a acurácia.
   - A barra de progresso `tqdm` é utilizada para monitorar o progresso da avaliação.

2. **Cálculo da Acurácia:**
   - A acurácia é calculada como a razão entre o número de previsões corretas e o total de previsões realizadas.
   - Fórmula: `Acurácia = (Número de Previsões Corretas) / (Total de Previsões)`

## Salvamento do Modelo

- O modelo fine-tuned e o tokenizer são salvos em um diretório especificado (`./finetuned_chemberta`) para uso futuro.

## Método de Validação

- **Validação Simples:** Foi utilizada uma divisão simples dos dados em conjuntos de treino e teste (80/20) em vez de uma validação cruzada. Isso foi escolhido para simplificar o processo e focar no fine-tuning do modelo. No entanto, para uma análise mais robusta, a validação cruzada pode ser implementada em trabalhos futuros.

## Resumo do Script

1. **Carregar e preparar os dados:**
   ```python
   df = self.spark.read.parquet(self.data_path)
   df = df.select(col("canonical_smiles"), col("target"))
   df_pandas = df.toPandas()
   smiles = df_pandas['canonical_smiles'].tolist()
   labels = df_pandas['target'].astype('category').cat.codes.tolist()
   smiles_train, smiles_test, labels_train, labels_test = train_test_split(smiles, labels, test_size=0.2, random_state=42)
   ```

2. **Treinamento do modelo:**
   ```python
   for epoch in range(self.epochs):
       epoch_loss = 0
       for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}/{self.epochs}"):
           tokens, labels = batch
           tokens = {key: val.to(self.device) for key, val in tokens.items()}
           labels = labels.to(self.device)
           outputs = self.model(**tokens)
           embeddings = outputs.last_hidden_state.mean(dim=1)
           loss = criterion(embeddings, labels)
           loss.backward()
           optimizer.step()
           scheduler.step()
           optimizer.zero_grad()
           epoch_loss += loss.item()
       print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(self.train_loader)}")
   ```

3. **Avaliação do modelo:**
   ```python
   self.model.eval()
   correct_predictions = 0
   total_predictions = 0
   with torch.no_grad():
       for batch in tqdm(self.test_loader, desc="Evaluating"):
           tokens, labels = batch
           tokens = {key: val.to(self.device) for key, val in tokens.items()}
           labels = labels.to(self.device)
           outputs = self.model(**tokens)
           embeddings = outputs.last_hidden_state.mean(dim=1)
           predictions = torch.argmax(embeddings, dim=1)
           correct_predictions += (predictions == labels).sum().item()
           total_predictions += labels.size(0)
   accuracy = correct_predictions / total_predictions
   print(f"Test Accuracy: {accuracy * 100:.2f}%")
   ```

4. **Salvamento do modelo:**
   ```python
   self.model.save_pretrained(output_dir)
   self.tokenizer.save_pretrained(output_dir)
   ```

## Autor

Leon - PhD Candidate in Computational Modeling of Biological Systems

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
```

### Estrutura do Repositório

```plaintext
repository/
├── chemberta_finetuner.py
├── format_file_ml.py  # Script fornecido para gerar o arquivo Parquet
├── requirements.txt
├── README.md
├── PROCEDURE.md
└── LICENSE  # Se você tiver um arquivo de licença
```

