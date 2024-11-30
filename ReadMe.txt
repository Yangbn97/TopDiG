训练TopDiG的流程为：先训练关键点检测网络，再整体训练TopDiG，详细参数设置见第5点

1、启动：python main.py

2、指定数据集：main.py中指定config文件，第42行

3、调整超参：在/configs目录下各个数据集的json文件中调整实验参数

4、config文件中关键参数解释：
Experiment：
	object_type: 设定为'line'时，提取线状目标中心线；设置其他参数均执行面状地物轮廓提取
	pretrain：预训练关键点检测网络
	infer：纯推理
	infer_eval：推理+算指标，执行过程中默认保存栅格推理结果至['Paths']['SaveRoot']
	save_shp：是否保存shapefile结果
	save_seg：是否保存raster结果，infer_eval=True时默认开启
	evaluate：是否单独执行train_eval
	dataset_name：数据集名，/data/DataLoader.py中用于加载各个数据集
	detection_resume：是否加载关键点检测网络权重
	match_resume：是否加载邻接矩阵预测网络
	ViT_pretrain： 是否加载ViT预训练权重，默认为True

Paths:
	TrainRoot, ValRoot：训练集和测试集，对于有raster label的数据，目录下应包含/image和/binary_map两个子目录，对于coco格式（如CrowdAI）， 根目录下应包含/image和annotation.json
	TestLabelRoot：测试集标签路径
	SaveRoot：推理结果保存路径，必须要有
	records_filename：实验中间结果记录在./records/records_filename目录下

Hyper：实验超参，已设置好

Model：
	detection_model：选择关键点网络，默认DFF(即TCND)，另提供FPN用于消融实验
	NUM_POINTS：关键点数量
	input_img_size：大幅面推理时的patch_size，默认定为训练数据的尺寸
	overlap_stride：大幅面推理时patch间重叠的步长，已定为默认值
	dilate_pixels：boundary IoU膨胀系数
	phi：检出点间隔，已默认
	delta：预测点和真值点配队的容忍距离
	num_attention_layers，num_heads， hidden_dim：ViT层数、头数和维度
	Sinkhorn：是否执行Sinkhorn算法，面状地物默认执行，线状地物不执行
	threshold：线状地物对预测的邻接矩阵进行二值化的阈值
	deep_supervision：是否在训练TCND时启用deep supervision策略
	show_outputs：是否可视化TCND每层sideout，预训练时生效

5、参数设置：
预训练关键点检测网络：pretrain=1，infer=0，infer_eval=0，save_shp=0，save_seg=0，evaluate=0, detection_resume=0, match_resume=0
TopDiG整体训练：pretrain=0， infer=0，infer_eval=0，save_shp=0，save_seg=0，evaluate=0, detection_resume=1, match_resume=0，ViT_pretrain=1 （加载预训练的TCND权重）
只执行train_eval: pretrain=0，infer=0，infer_eval=0，save_shp=0，save_seg=0，evaluate=1, detection_resume=1, match_resume=1
纯推理不保存结果：pretrain=0，infer=1，infer_eval=0，save_shp=0，save_seg=0，evaluate=0, detection_resume=1, match_resume=1
推理+算指标：pretrain=0，infer=1，infer_eval=1，save_shp=0，save_seg=0，evaluate=0, detection_resume=1, match_resume=1


