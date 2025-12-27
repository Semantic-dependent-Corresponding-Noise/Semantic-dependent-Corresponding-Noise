# Entity Referential Error Noise Generation Workflow

## Usage Steps

### I. Generate Base Noise Files

**1. Generate Training Set Noise (100% Error)**
   Command: `python Entity_Referential_Error_noise_construct.py`
   - Generate **100% erroneous** noise descriptions for the training set.
   - This step is necessary to create the foundational data for subsequent noise ratio generation.

**2. Generate Test Set Noise**
   Command: `python Entity_Referential_Error_test_noise_construct.py`
   - Directly generate noise files for the test set.

### II. Generate Noise with Specified Ratio

After obtaining the complete base noise files, select the appropriate script based on the error distribution requirements for the captions (1 image corresponds to 5 captions):

- **Partial Error (Mixed)**
  Command: `python Entity_Referential_Error_noise.py`
  - **Description:** Generates a dataset where **not all** 5 descriptions for an image are erroneous (some are correct, some are noisy).

- **All Error (Total)**
  Command: `python Entity_Referential_Error_img5txt_noise.py`
  - **Description:** Generates a dataset where **all 5** descriptions for an image are erroneous.
