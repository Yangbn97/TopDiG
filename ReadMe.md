This is the Inference code of CVPR2023 paper "TopDiG: Class-agnostic Topological Directional Graph Extraction from Remote Sensing Images"[link](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_TopDiG_Class-Agnostic_Topological_Directional_Graph_Extraction_From_Remote_Sensing_Images_CVPR_2023_paper.html)

1、Switch datasets：specify config file in the main.py

2、Start command：python main.py

3、Parameters in config files：
Experiment：
	object_type: when set as 'line'，extract centerline, otherwise extracting contours
	save_shp：whether to save results as shapefile. The default is True
	save_seg：whether to save results as binary mask
	evaluate：whether to calculate accuracy metrics. If True, the script will firstly saves results as binary mask to SaveRoot and then calculate metrics.
	dataset_name：dataset name
	detection_resume：whether to load TCND checkpoints
	match_resume：whether to load DiG generator checkpoints

Paths:
	TestRoot：image path.对于有raster label的数据，目录下应包含/image和/binary_map两个子目录，对于coco格式（如CrowdAI）， 根目录下应包含/image和annotation.json
	TestLabelRoot：label path
	SaveRoot：path to save results
	records_filename：save intermedia outputs
	pretrained_detection_weight_name: basename of TCND ckecpoint file
	pretrained_match_weight_name: basename of DiG generator checkpoint file

Model：
	NUM_POINTS：the total number of detected nodes per image
	dilate_pixels：for evaluate boundary IoU
	phi：distance among detected nodes
	delta：tolarence when match detected and GT nodes
	num_attention_layers，num_heads， hidden_dim：ViT layer number、head number and hidden dimensions
	Sinkhorn：whether to conduct Sinkhorn. True for polygon shape objects; Flase for line shape objects

4、Datasets dictionary：
examples for datasets with label:
"TestRoot": "/data02/ybn/Datasets/Building/Inria/raw/train/image",
"TestLabelRoot": "/data02/ybn/Datasets/Building/Inria/raw/train/binary_map",
If no accuracy evaluate step, this script can run without labels. In this case, set evaluate=0

5、Pretrained checkpoints:
Google drive:https://drive.google.com/drive/folders/1E3jNSO8CGl_72V1rq38a-aagGEFL7S6c?usp=drive_link
Baidu drive:https://pan.baidu.com/s/1EddQLzkyWCoqZVFIxIuuAQ (password:yqxa) 
The downloaded contents are /records/ which contains checkpoints for Inria, GID and Massachusetts. It should be downloaded to root path (./)




