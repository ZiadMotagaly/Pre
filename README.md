# Foundational Number Sense Training Gains Are Predicted by Hippocampal-Parietal Circuits

How can humans count ? 

In this research project, we designed a theoretically motivated training program to determine brain circuit mechanisms underlying foundational number sense learning (counting) in female and male elementary school-age children (7-10 years).

Brain activity during counting tasks can provide valuable insights into dyscalculia, but it may not be a definitive diagnostic tool on its own. Recent research has revealed both similarities and differences in brain activity between individuals with dyscalculia and those without. Crucially, the strength of pretraining functional connectivity between the hippocampus and intraparietal sulcus, brain regions implicated in associative learning and quantity discrimination, respectively, predicted individual differences in number sense learning across typically developing children and children with learning difficulties

## Brain Activity Patterns
* Studies using functional magnetic resonance imaging (fMRI) have shown that:
The intraparietal sulcus (IPS) is a key region for numerical processing.
Children with dyscalculia often show less activity in the IPS and reduced connectivity with other brain regions when dealing with numbers.
* The parietal cortex, as well as cortical and subcortical regions, are involved in number processing and calculation.
However, recent findings challenge some previous assumptions:
  * A large-scale fMRI study found no consistent group differences in brain activation between children with and without dyscalculia during arithmetic, magnitude processing, and visuospatial working memory tasks.


## Diagnostic Challenges
* While brain imaging can reveal patterns associated with dyscalculia, several factors complicate its use as a standalone diagnostic tool:
  * Overlap with other conditions: Dyscalculia often co-occurs with other learning disabilities, making it challenging to isolate specific neural markers.
  * Variability in activation patterns: Some studies report decreased activation in certain brain regions for individuals with dyscalculia, while others show increased activation in the same areas.
Need for comprehensive assessment: Diagnosing dyscalculia typically requires a combination of cognitive assessments, tests of number sense, and specific math-related tasks.


This is a probabilistic tractography package primarily based on BEDPOSTX and ProbTrackX2.

There are 2 packages that need to be pre-installed to run the niflow package successfully :

FSL - A comprehensive library of analysis tools for FMRI, MRI and diffusion brain imaging data. https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
Diffusion Pipelines - A custom built package contatining the necessary pre processing registrations, which can be applied to the raw inputs (dwi, bvecs etc..)
The mode of execution used is Multiproc , and its applied on slurm based parallel clustering . Hence, the package includes:

submit.sh: Configures the parameters for cluster computation
mg_test.cfg: Configures the path directories of inputs needed for the main script. Note that the id-list has to have exactly one space charachter between each ID.
dti_raw_bvec: A raw bvec file that is used for the niflow_test script, to validate the bvec_flip function.
The main script is m_niflow_multiproc. It contains all the required libraries that can be used interchangeably to customize the diffusion pipline according to the preference of the user. The principal tractography algorithms are constant: bedpostx and probtrackx2.

m_niflow_multiproc_lite is derived from the main script , and it contains the minimum amount of modules and nodes that are required ot run the pipeline.


