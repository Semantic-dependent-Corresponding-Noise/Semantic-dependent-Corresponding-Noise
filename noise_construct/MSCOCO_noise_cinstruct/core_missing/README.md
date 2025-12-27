File Structure
text
.
├── train_file/                    # Output directory for training set noise files
├── test_file/                     # Output directory for test set noise files
├── yibu_batch_core_missing_noise_doubao_construct.py      # Batch noise generation for training set (fully incorrect)
├── yibu_batch_core_missing_test_noise_doubao_construct.py # Batch noise generation for test set (fully incorrect)
├── merge_files.py                 # File merging tool
├── core_missing_noise.py          # Generate noise files with partially incorrect descriptions
└── core_missing_img5txt_noise.py  # Generate noise files with fully incorrect descriptions
Usage Steps
1. Generate Fully Incorrect Noise Files (Batch Processing)
Training Set Noise Generation
bash
python yibu_batch_core_missing_noise_doubao_construct.py
Output Format: The program generates one file for every 1000 images in sequential order, saved as:

text
train_file/train_caps_5_per_image_part{file_number}.txt
Test Set Noise Generation
bash
python yibu_batch_core_missing_test_noise_doubao_construct.py
Output Format: Same as above, but for test set data.

2. Merge Batch Files
bash
python merge_files.py
Merge the segmented batch files into a complete noise file.

3. Generate Noise Files with Specified Ratios
Option A: 5 Description Texts per Image (Partially Incorrect)
bash
python3 core_missing_noise.py
Generate files where only some description texts contain noise for each image.

Option B: 5 Description Texts per Image (Fully Incorrect)
python3 core_missing_img5txt_noise.py
Generate files where all 5 description texts contain noise for each image.

Important Notes
Ensure the corresponding image dataset and original description files are prepared before running

Batch generation processes sequentially and saves intermediate files for every 1000 images

The file merging step must be executed after batch generation is complete

Final noise ratios can be adjusted in the respective scripts according to requirements

Output File Description
Batch generated files: train_caps_5_per_image_part{number}.txt

Merged complete file: 1.0_noise_train_caps 

Final noise files: Generated according to selected script with different noise ratios
