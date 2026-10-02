# Chemical Dice Molecular Embeddings

Embeds a molecule as an 8192-dimensional vector from its SMILES alone. The authors fused six views of each molecule (descriptors, graph, image, bioactivity, quantum and language features) into a teacher embedding and trained a Mamba state-space model on about 2.2 million ChEMBL compounds to predict it. A classifier built on the embedding prioritised two compounds that reduced hydroxyurea-induced DNA damage in yeast. Ersilia canonicalises SMILES first; inputs beyond 126 tokens are truncated.



## Information
### Identifiers
- **Ersilia Identifier:** `eos88ir`
- **Slug:** `chemical-dice-embeddings`

### Domain
- **Task:** `Representation`
- **Subtask:** `Featurization`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `Embedding`, `Chemical language model`, `ChEMBL`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `8192`
- **Output Consistency:** `Fixed`
- **Interpretation:** 8192-dimensional molecular embedding distilled from six fused molecular views; individual dimensions have no direct meaning.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| feat_0000 | float |  | Chemical Dice embedding dimension 0 of 8192 predicted from the SMILES by the distilled Mamba model |
| feat_0001 | float |  | Chemical Dice embedding dimension 1 of 8192 predicted from the SMILES by the distilled Mamba model |
| feat_0002 | float |  | Chemical Dice embedding dimension 2 of 8192 predicted from the SMILES by the distilled Mamba model |
| feat_0003 | float |  | Chemical Dice embedding dimension 3 of 8192 predicted from the SMILES by the distilled Mamba model |
| feat_0004 | float |  | Chemical Dice embedding dimension 4 of 8192 predicted from the SMILES by the distilled Mamba model |
| feat_0005 | float |  | Chemical Dice embedding dimension 5 of 8192 predicted from the SMILES by the distilled Mamba model |
| feat_0006 | float |  | Chemical Dice embedding dimension 6 of 8192 predicted from the SMILES by the distilled Mamba model |
| feat_0007 | float |  | Chemical Dice embedding dimension 7 of 8192 predicted from the SMILES by the distilled Mamba model |
| feat_0008 | float |  | Chemical Dice embedding dimension 8 of 8192 predicted from the SMILES by the distilled Mamba model |
| feat_0009 | float |  | Chemical Dice embedding dimension 9 of 8192 predicted from the SMILES by the distilled Mamba model |

_10 of 8192 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`

### Resource Consumption


### References
- **Source Code**: [https://github.com/the-ahuja-lab/ChemicalDice](https://github.com/the-ahuja-lab/ChemicalDice)
- **Publication**: [https://doi.org/10.1038/s41467-026-77700-z](https://doi.org/10.1038/s41467-026-77700-z)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2026`
- **Ersilia Contributor:** [TiagoJanela](https://github.com/TiagoJanela)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [MIT](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos88ir
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos88ir
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
