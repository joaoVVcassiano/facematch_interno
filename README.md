# facematch_interno
Aqui está um exemplo de como você pode estruturar o arquivo `README.md` para o seu projeto em Markdown e em português:

# FaceMatch Interno

O projeto FaceMatch Interno é destinado a desenvolver e manter um sistema de comparação facial para fins de segurança e verificação de identidade.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

```
facematch_interno/
├── base/
├── data/
├── figures/
├── notebooks/
├── tests/
├── trash/
├── utils/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
```

- `base/`: Contém as bases de brutos, utilizados inicialmente no projeto.
- `data/`: Diretório para armazenar datasets utilizados e gerados pelo sistema.
- `figures/`: Imagens e figuras geradas para documentação e relatórios.
- `notebooks/`: Jupyter notebooks para experimentação e prototipagem.
- `tests/`: Contém os testes automatizados do projeto.
- `trash/`: Arquivos temporários, legados ou descartáveis.
- `utils/`: Funções utilitárias para uso geral no projeto.

## Configuração do Ambiente

Para configurar o ambiente de desenvolvimento, siga os passos abaixo:

1. Clone o repositório do projeto:
```
git clone https://github.com/joaoVVcassiano/facematch_interno
```

2. Navegue até a pasta do projeto:
```
cd facematch_interno
```

3. Instale as dependências utilizando o `requirements.txt`:
```
pip install -r requirements.txt
```

## Uso

O projeto de facematch interno é composto por 3 módulos principais:
- etapa de pré-processamento dos dados (brilho, alinhamento, etc);
- landmarks das faces(68 pontos, 5 pontos, etc);
- comparação facial :
  - 1:1 -> através dos landmarks (distância euclidiana, etc);
  - 1:N -> salvar os landmakrs em um banco de dados e comparar com os landmarks de uma nova imagem. 

## Testes

Para executar os testes, use o seguinte comando:

```
python -m unittest discover tests
```

## Contribuição

Contribuições são bem-vindas. Por favor, leia o arquivo `LICENSE` para mais detalhes sobre os termos e condições de uso.

## Licença

Este projeto está licenciado sob a Licença SEM LICENÇA - veja o arquivo `LICENSE` para mais detalhes, rs.