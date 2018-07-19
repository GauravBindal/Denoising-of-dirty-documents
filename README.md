# Denoising-of-dirty-documents

Requirements: -opencv -sklearn -numpy 
It contains 4 sub folder: 

-train 
-train cleaned 
-test 
-test cleaned
-cleaning.py

Put the document you want to clean in test folder and than run the python script (cleaning.py).
The cleaned image will be automatically saved with the same name in the test cleaned folder.

This python script uses a erosion technique to erode the content of the image so to get the background of the image and then it removes that dirty background from the original document with the help of kernels.
