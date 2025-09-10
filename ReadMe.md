This is the Inference code of CVPR2023 paper ["TopDiG: Class-agnostic Topological Directional Graph Extraction from Remote Sensing Images"](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_TopDiG_Class-Agnostic_Topological_Directional_Graph_Extraction_From_Remote_Sensing_Images_CVPR_2023_paper.html)

1、Start
=
* Setting the environment by command:
```python
pip install requirements.txt
```
* Set the config file in /configs
*  specify config file in the main.py
*  run
  ```python 
python main.py
```


2、Parameters in config files
=
**Experiment**:<br>
	>>object_type: when set as 'line'，extract centerline, otherwise extracting contours<br>
	>>**save_shp**：whether to save results as shapefile. The default is True<br>
	>>**save_seg**：whether to save results as binary mask<br>
	>>**evaluate**：whether to calculate accuracy metrics. If True, the script will firstly saves results as binary mask to SaveRoot and then calculate metrics.<br>
	>>dataset_name：dataset name<br>
	>>**detection_resume**：whether to load TCND checkpoints<br>
	>>**match_resume**：whether to load DiG generator checkpoints<br>
<br>
**Paths**:<br>
	>>**TestRoot**：image path<br>
	>>**TestLabelRoot**：label path<br>
	>>**SaveRoot**：path to save results<br>
	>>records_filename：save intermedia outputs<br>
	>>pretrained_detection_weight_name: basename of TCND ckecpoint file<br>
	>>pretrained_match_weight_name: basename of DiG generator checkpoint file<br>

**Model**:<br>
	>>NUM_POINTS：the total number of detected nodes per image<br>
	>>dilate_pixels：for evaluate boundary IoU<br>
	>>phi：distance among detected nodes<br>
	>>delta：tolarence when match detected and GT nodes<br>
	>>num_attention_layers，num_heads， hidden_dim：ViT layer number、head number and hidden dimensions<br>
	>>Sinkhorn：whether to conduct Sinkhorn. True for polygon shape objects; Flase for line shape objects<br>

3、Datasets dictionary
=
**Examples for datasets with label**:<br>

"TestRoot": "/data02/ybn/Datasets/Building/Inria/raw/train/image"<br>
"TestLabelRoot": "/data02/ybn/Datasets/Building/Inria/raw/train/binary_map"

If no accuracy evaluation step, this script can run without labels. In this case, set evaluate=0

4、Pretrained checkpoints
=
**Google drive**:https://drive.google.com/drive/folders/1E3jNSO8CGl_72V1rq38a-aagGEFL7S6c?usp=drive_link<br>
**Baidu drive**:https://pan.baidu.com/s/1EddQLzkyWCoqZVFIxIuuAQ (password:yqxa) <br>
The downloaded contents are /records/ which contains checkpoints for Inria, GID and Massachusetts. It should be downloaded to root path (./)




