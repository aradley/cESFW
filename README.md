# cESFW
cESFW is a feature correlation software based on the principles of Entropy Sorting. 

The theory underpinning cESFW is outlined in the following publications:

[Entropy sorting of single-cell RNA sequencing data reveals the inner cell mass in the human pre-implantation embryo](https://www.cell.com/stem-cell-reports/fulltext/S2213-6711(22)00456-8?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2213671122004568%3Fshowall%3Dtrue)

[Branching topology of the human embryo transcriptome revealed by entropy sort feature weighting](https://journals.biologists.com/dev/article/151/11/dev202832/352632/Branching-topology-of-the-human-embryo)

In [Branching topology of the human embryo transcriptome revealed by entropy sort feature weighting](https://journals.biologists.com/dev/article/151/11/dev202832/352632/Branching-topology-of-the-human-embryo), cESFW is used as part of a feature selection workflow to identify genes that are informative of cellular identity in single cell RNA sequeencing (scRNA-seq) data.

## Usuage
In [Branching topology of the human embryo transcriptome revealed by entropy sort feature weighting](https://journals.biologists.com/dev/article/151/11/dev202832/352632/Branching-topology-of-the-human-embryo) we provide details for usage of cESFW.

The cESFW algorithm takes a 2D matrix as an input where the rows are samples and the columns are features. For scRNA-seq, the rows are cells and the columns are genes. Each feature in the 2D matrix should be scaled so that the maximum value is 1 and the minimum value is 0.

cESFW will then output an Entropy Sort Score (ESS) pairwise correlation matrix, and an Error Potential (EP) pairwise correlation significance matrix.

### Example workflows

To find example workflows/vignettes for using cESFW as a feature selection algorithm, please go to our accompanying [cESFW_Embryo_Topology_Paper](https://github.com/aradley/cESFW_Embryo_Topology_Paper/tree/main) repository.

### Installation
1. Retreive the ripository with: `git clone https://github.com/aradley/cESFW.git`
2. Navigate to the directory where the clone was downloaded to, for example: `cd cESFW/`
3. Run the following on the command line: `python setup.py install`
