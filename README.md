# Smart Contract Similarity Detection
Fall 2022 - SE6387.001 Advanced Software Engineering Project

## Project Goal 
1. Given two smart contracts, automatically calculate their similarity (using cosine similarity)
2. Generate probability density function graphs to determine the threshold of the similarity score
3. Detect the co-cloned pairs and the non-co-cloned pairs of contracts along with their similarity values using the computed threshold in step 2
 <br> <br>
<p align="center">
  <img width="950" height="200" alt="Flowchart Diagram" src="https://user-images.githubusercontent.com/60459873/205159021-aafd08b5-7ad3-4f05-a46f-27cc847dcf4e.png">
</p>
<p align="center">Figure 1: Flowchart Diagram of the Project Goal</p>

## Description

This diagram represents the different components of our project and its structure. It shows the relationship between the directories in a hierarchical way for a much simpler visualization. <br>

<p align="center">
 <img width="656" alt="Overall Project Structure" src="https://user-images.githubusercontent.com/60459873/205170838-e72f0485-9bc6-4139-8f79-5b2835251bb1.png">
</p>
<p align="center">Figure 2: Diagram of the Overall Project Structure</p>

## Internal Structure of the Program

This class diagram represents the internal classes that are defined and used in the program. It shows all of the main classes along with their attributes and defined methods.

<p align="center">
  <img width="668" alt="Class Diagram" src="https://user-images.githubusercontent.com/60459873/205170625-c9c90c54-845f-41d4-b76f-cf930119efef.png">
</p>
<p align="center">Figure 3: Class Diagram of the Internal Project Structure</p>

## Getting Started

### Setup

1. Create a ```/contracts``` directory under ```/src/```
2. Go to this github repository https://github.com/PrCatch/etherscan-0815-0914 
3. Download project as a ZIP file in another folder NOT in ```/src/```
4. Extract the project from the ZIP file
5. From the etherscan folder, select the contracts you want to compare from the ```contracts``` folder
6. Place those contracts into ```/src/contracts``` directory
    * The contracts should be placed following the format of ```/#/#/project``` <br>
    * Example: <br>
    ![Screenshot 2022-12-01 at 11 20 01 PM](https://user-images.githubusercontent.com/60459873/205220425-d274ec6a-3bad-4be4-91d7-15fda59caa4c.png)

### Environment Prerequisites

* Devices:
    * Mac or Linux
    * Windows
* Python 3.8 or higher
* Python Packages: 
    * os
    * pathlib
    * typing
    * shutil
    * csv
    * subprocess
    * random
    * math
    * json
    * torch
    * tqdm
    * numpy
    * matplotlib
    * r2pipe
    * gensim
    * solcx
Note: Packages that need to be installed will be shown when the repository is open on your preferred external editor
* Solicity Compiler: https://github.com/ethereum/solidity <br>
* Radare2 Compiler: https://github.com/radareorg/radare2 <br>
* ASM_2_VEC model: https://github.com/oalieno/asm2vec-pytorch.git <br>

## Authors

| Name | Email |
| --- | --- |
| Junaid Hashmi | jsh171030@utdallas.edu |
| Miao Miao | mxm190091@utdallas.edu | 
| Jeongwon Seo | jxs170012@utdallas.edu | 

## Version History

* FinalCodeSubmission
    * Final Version of the Code

## Acknowledgments

* Dr. Wei Yang - Professor of SE6387.001
* Simin Chen - Staff Advisor
