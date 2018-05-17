#!/bin/bash

#################################################
#	Authors: 				#	
#		- Marco Cianfriglia (@IAC-CNR)	#
#		- Flavio Vella (@Dividiti)	#
#						#
#	Contacts:				#
#		- m.cianfriglia@iac.cnr.it	#
#		- flavio@dividiti.com		#
#						#
#	Requirements:				# 
#		- CK should be available	#
#		- Internet Connection		#
#		- Python 2.7			#
#						#
#################################################


#     _____                _         __  __           _      _ 
#    / ____|              | |       |  \/  |         | |    | |
#   | |     _ __ ___  __ _| |_ ___  | \  / | ___   __| | ___| |
#   | |    | '__/ _ \/ _` | __/ _ \ | |\/| |/ _ \ / _` |/ _ \ |
#   | |____| | |  __/ (_| | ||  __/ | |  | | (_) | (_| |  __/ |
#    \_____|_|  \___|\__,_|\__\___| |_|  |_|\___/ \__,_|\___|_|
#                                                              
#                                                              




echo "   _____                _         __  __           _      _ ";
echo "  / ____|              | |       |  \/  |         | |    | |";
echo " | |     _ __ ___  __ _| |_ ___  | \  / | ___   __| | ___| |";
echo " | |    | '__/ _ \/ _\` | __/ _ \ | |\/| |/ _ \ / _\` |/ _ \ |";
echo " | |____| | |  __/ (_| | ||  __/ | |  | | (_) | (_| |  __/ |";
echo "  \_____|_|  \___|\__,_|\__\___| |_|  |_|\___/ \__,_|\___|_|";
echo "                                                            ";
echo "                                                            ";


#Fill the following variables and remove the '#' to uncomment them

#TREE_DEPTH=0
#MIN_SAMPLES_PER_LEAF=1
#ROOT_OUTPUT_DIRECTORY

#Use the following export for a fine device control
#export CUDA_VISIBLE_DEVICES=0
#export GPU_DEVICES_ORDINAL=0


#The selected model will be generated for each of the following datasets
#Please uncomment[comment] the lines you are[are not] interested to  

#Toy

#Power-Of-Two

#Grid-Of-Two

#AntonNet 


#Generating datasets
echo "[INFO] - Generating Datasets"

echo "[INFO] - By default only a Toy dataset will be generated"

echo "[INFO] - Remove the comment on the lines representing the datasets you want to generate"

printf "[INFO] - Press any key to continue\n"
read -s
#Toy Dataset (3 matrices)
echo "[INFO] - Generating Toy"
ck run program:clblast-generate-dataset --cmd_key=Toy --deps.compiler={$gcc_uid}

#Power-Of-Two (216 matrices)
#echo "[INFO] - Generating Power-Of-Two"
#ck run program:clblast-generate-dataset --cmd_key=Power-Of-Two

#Grid-Of-Two (3375 matrices)
#echo "[INFO] - Generating Grid-Of-Two"
#ck run program:clblast-generate-dataset --cmd_key=Grid-Of-Two

#AntonNet (458 matrices)
#echo "[INFO] - Generating AntonNet"
#ck run program:clblast-generate-dataset --cmd_key=AntonNet


echo "[INFO] - All the datasets have been generated"

