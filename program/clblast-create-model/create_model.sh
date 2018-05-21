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

#Set this to your installation path
CKTOOLS="${HOME}/CK-TOOLS"
OUTDIR="/tmp"
DATASET_DIR="/tmp"
#TREE_DEPTH=0
#MIN_SAMPLES_PER_LEAF=1
#ROOT_OUTPUT_DIRECTORY

#Use the following export for a fine device control
#export CUDA_VISIBLE_DEVICES=0
#export GPU_DEVICES_ORDINAL=0


#The selected model will be generated for each of the following datasets
#Please uncomment[comment] the lines you are[are not] interested to  

height=$TREE_DEPTH
leafs=$MIN_SAMPLES_PER_LEAF
#Toy
datasetName="Toy"
tag="${datasetName}-h${height}-L${leafs}"
ck install package:lib-clblast-master-universal-tune-multiconf --extra_version=-$tag --extra_tags=-$tag --env.PACKAGE_GIT="YES" 

libroot=${CKTOOLS}/$(ls -1 ${CKTOOLS} | grep $tag)/src
ck run program:clblast-create-model --env.CK_DATASET_DIR=${DATASET_DIR}/Toy --env.CK_CLBLAST_ROOT=${libroot} --env.CK_OUTPUT_DIR=${OUTDIR}/${tag} --cmd_key="Toy"

#Power-Of-Two
datasetName="Power-Of-Two"
tag="${datasetName}-h${height}-L${leafs}"
ck install package:lib-clblast-master-universal-tune-multiconf --extra_version=-$tag --extra_tags=-$tag --env.PACKAGE_GIT="YES" 

libroot=${CKTOOLS}/$(ls -1 ${CKTOOLS} | grep $tag)/src
ck run program:clblast-create-model --env.CK_DATASET_DIR=${DATASET_DIR}/Toy --env.CK_CLBLAST_ROOT=${libroot} --env.CK_OUTPUT_DIR=${OUTDIR}/${tag} --cmd_key="Power-Of-Two"

#Grid-Of-Two
datasetName="Grid-Of-Two"
tag="${datasetName}-h${height}-L${leafs}"
ck install package:lib-clblast-master-universal-tune-multiconf --extra_version=-$tag --extra_tags=-$tag --env.PACKAGE_GIT="YES" 

libroot=${CKTOOLS}/$(ls -1 ${CKTOOLS} | grep $tag)/src
ck run program:clblast-create-model --env.CK_DATASET_DIR=${DATASET_DIR}/Toy --env.CK_CLBLAST_ROOT=${libroot} --env.CK_OUTPUT_DIR=${OUTDIR}/${tag} --cmd_key="Grid-Of-Two"

#AntonNet 
datasetName="AntonNet"
tag="${datasetName}-h${height}-L${leafs}"
ck install package:lib-clblast-master-universal-tune-multiconf --extra_version=-$tag --extra_tags=-$tag --env.PACKAGE_GIT="YES" 

libroot=${CKTOOLS}/$(ls -1 ${CKTOOLS} | grep $tag)/src
ck run program:clblast-create-model --env.CK_DATASET_DIR=${DATASET_DIR}/Toy --env.CK_CLBLAST_ROOT=${libroot} --env.CK_OUTPUT_DIR=${OUTDIR}/${tag} --cmd_key="anton-net"



echo "[INFO] - All the models have been generated"

