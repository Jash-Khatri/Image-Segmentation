#######################################################################################
#                                                                      		      #			
#             Software implements Graph Cuts using OpenMP, C++        		      #	
#								      	 	      #	
#										      #	
#   Copyright (c) 2021 Indian Institute of Technology Madras.			      #	
#   All rights reserved.							      #	
#									      	      #	
#   THE SOFTWARE IS PROVIDED WITHOUT WARRANTY OF ANY KIND. 			      #	
#   									      	      #	
#   Please report any issue to Jash Khatri ( cs19s018@smail.iitm.ac.in)		      #	
#	                                                                              #
#   "Faster Image segmentation on the multi-core platforms using the graph cuts"      #
#   Jash Khatri	(CS19S018)	                                                      #
#										      #
#	                                                                              #
#######################################################################################


=======================================================================================================================                                                                            
1. Dependencies

-> The machine should have OpenCV installed in it. The preferred OpenCV version for the smooth functioning of the submitted codes is 4.5.0 or higher.
-> The machine should have g++ installed in it. The preferred g++ version for the smooth functioning of the submitted codes is 7.5.0 or higher.
-> The machine should have support for OpenMP. The preferred OpenMP version for the smooth functioning of the submitted codes is 4.5 or higher.

=======================================================================================================================

2. Primary Codes and Image Dataset

-> Code implementing Serial Image segmentation application present at: codes/Image-Seg-Serial.cpp 
-> Code implementing CPU Parallel Image segmentation application present at: codes/Image-Seg-Omppar.cpp
-> The Dataset folder contains all the sample square grayscale images on which the code is tested. 

=======================================================================================================================

3. Here is the brief description of how to compile and run the submitted serial and parallel Image segmentation applications in the Linux Environment

-----------------------------------------------------------------------------------------------------------------------

Serial Image segmentation applications can be compiled and run on a Linux terminal as follows:

g++ Image-Seg-Serial.cpp `pkg-config --cflags --libs opencv`
./a.out image_name bseed1 bseed2 bseed3 bseed4 bseed5 fseed1

Note: 
image_name: It is the name of the file containing the image that is to be segmented.
bseed1 bseed2 bseed3 bseed4 bseed5: These are the background seeds that have to be provided by the user.
fseed1: This is the foreground seed that has to be specified by the user.

Note:
The output file containing the resulting segmented image will be available in the current directory in which the above commands are executed. 

Examples showing how to run the sample images of the Dataset folder: 

./a.out ../Dataset/test1.png 130 390 790 1210 1590 700

./a.out ../Dataset/test2.png 125 395 795 1205 1595 700

./a.out ../Dataset/test3.png 130 390 790 1210 1590 700

./a.out ../Dataset/test4.png 123 398 797 1202 1598 700

./a.out ../Dataset/test5.png 123 398 797 1202 1598 700

./a.out ../Dataset/test6.png 123 398 797 1202 1598 700

./a.out ../Dataset/test7.png 100 300 500 700 840 1300

./a.out ../Dataset/test8.png 93 243 422 633 873 435

./a.out ../Dataset/test9.png 63 183 198 263 398 190

./a.out ../Dataset/test10.png 100 250 400 500 610 890

./a.out ../Dataset/test11.png 50 100 155 200 250 390

./a.out ../Dataset/test12.png 93 243 422 633 873 435

./a.out ../Dataset/test13.png 63 183 198 263 398 190

-------------------------------------------------------------------------------------------------------------------------

Parallel Image segmentation applications can be compiled and run on a Linux terminal as follows:

g++ -fopenmp Image-Seg-Omppar.cpp `pkg-config --cflags --libs opencv`
./a.out image_name bseed1 bseed2 bseed3 bseed4 bseed5 fseed1 num_threads

Note: 
image_name: It is the name of the file containing the image that is to be segmented.
bseed1 bseed2 bseed3 bseed4 bseed5: These are the background seeds that have to be provided by the user.
fseed1: This is the foreground seed that has to be specified by the user.
num_threads: The number of threads with which the application is to run.

Note:
The output file containing the resulting segmented image will be available in the current directory in which the above commands are executed. 

Examples showing how to run the sample images of the Dataset folder: 

./a.out ../Dataset/test1.png 130 390 790 1210 1590 700 2

./a.out ../Dataset/test2.png 125 395 795 1205 1595 700 2

./a.out ../Dataset/test3.png 130 390 790 1210 1590 700 2

./a.out ../Dataset/test4.png 123 398 797 1202 1598 700 2

./a.out ../Dataset/test5.png 123 398 797 1202 1598 700 2

./a.out ../Dataset/test6.png 123 398 797 1202 1598 700 2

./a.out ../Dataset/test7.png 100 300 500 700 840 1300 2

./a.out ../Dataset/test8.png 93 243 422 633 873 435 2

./a.out ../Dataset/test9.png 63 183 198 263 398 190 2

./a.out ../Dataset/test10.png 100 250 400 500 610 890 2

./a.out ../Dataset/test11.png 50 100 155 200 250 390 2

./a.out ../Dataset/test12.png 93 243 422 633 873 435 2

./a.out ../Dataset/test13.png 63 183 198 263 398 190 2


=======================================================================================================================

4. Testing

The codes are tested on a machine with an Intel Core i5-6200U @ 2.30GHz CPU having 8 GB RAM and four processing cores. The machine runs Ubuntu Debian 18.04.5 LTS (64-bit) and has OpenCV, OpenMP, gcc, and g++ installed in it. Both serial and parallel codes segmented the grayscale images present in the Dataset folder successfully without any issues on the above-specified machine.

=======================================================================================================================

5. Limitations

Input images provided to both serial and parallel image segmentation should be square and grayscale images. Images should be in (.png) format.
Note that the square grayscale images are selected only for the sake of convenience. The same implementation of the serial and parallel graph cut algorithm can work on the non-square colored images with some modification in the main routine. 


