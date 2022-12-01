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

### Dependencies

* Devices:
    * Mac or Linux
    * Windows
* Python Libraries: 
    * gensim
    * numpy
    * matplotlib
    * r2pipe
    * torch
    * py-solc-x
    * etc,. <br>
Note: Libraries that need to be installed will be shown when the repository is open on your preferred external editor
* External ASM_2_VEC model:
    * https://github.com/oalieno/asm2vec-pytorch.git

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
