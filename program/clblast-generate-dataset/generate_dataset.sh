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
#		- OpenCL installed		#
#						#
#################################################

#     _____                           _         _____        _                 _       
#    / ____|                         | |       |  __ \      | |               | |      
#   | |  __  ___ _ __   ___ _ __ __ _| |_ ___  | |  | | __ _| |_ __ _ ___  ___| |_ ___ 
#   | | |_ |/ _ \ '_ \ / _ \ '__/ _` | __/ _ \ | |  | |/ _` | __/ _` / __|/ _ \ __/ __|
#   | |__| |  __/ | | |  __/ | | (_| | ||  __/ | |__| | (_| | || (_| \__ \  __/ |_\__ \
#    \_____|\___|_| |_|\___|_|  \__,_|\__\___| |_____/ \__,_|\__\__,_|___/\___|\__|___/
#                                                                                      
#                                                                                      



echo "   _____                           _         _____        _                 _       ";
echo "  / ____|                         | |       |  __ \      | |               | |      ";
echo " | |  __  ___ _ __   ___ _ __ __ _| |_ ___  | |  | | __ _| |_ __ _ ___  ___| |_ ___ ";
echo " | | |_ |/ _ \ '_ \ / _ \ '__/ _\` | __/ _ \ | |  | |/ _\` | __/ _\` / __|/ _ \ __/ __|";
echo " | |__| |  __/ | | |  __/ | | (_| | ||  __/ | |__| | (_| | || (_| \__ \  __/ |_\__ \\";
echo "  \_____|\___|_| |_|\___|_|  \__,_|\__\___| |_____/ \__,_|\__\__,_|___/\___|\__|___/";
echo "                                                                                    ";
echo "                                                                                    ";


#Testing CK installation
printf "[INFO] - Testing CK installation..."
ck &> /dev/null

if [ $? -ne 0 ]
then
	echo "[ERROR] - This script requires a valid CK installation"
	exit 1
fi
printf "done!\n"

#Testing Internet Connection
printf "[INFO] - Testing Internet connectivity..."
wget https://www.wikipedia.org -O /dev/null &> /dev/null

if [ $? -ne 0 ]
then
	echo "[ERROR] - This script requires Internet connection"
	exit 1
fi
printf "done!\n"

#Select the accelerator ID
#export CUDA_VISIBLE_DEVICES=0
#export GPU_DEVICES_ORDINAL=0

#Install Clblast Multiconf Default
printf "[INFO] - Installing CLBlast multiconf default\n"
printf "[INFO] - Please follow the instructions on the screen\n"
printf "[INFO] - Press any key to continue\n"
read -s
ck install package:lib-clblast-master-universal-tune-multiconf --env.PACKAGE_GIT=YES

#Run program to assign a default valure fro the compler 
ck run program:tool-print-opencl-devices
#Collecting GCC env UID
gcc_uid=$(ck show env --tags=compiler,gcc | tail -n 1 | cut -d' ' -f1)
llvm_uid=$(ck show env --tags=compiler,llvm | tail -n 1 | cut -d' ' -f1)
ck rm env:$llvm_uid 
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

