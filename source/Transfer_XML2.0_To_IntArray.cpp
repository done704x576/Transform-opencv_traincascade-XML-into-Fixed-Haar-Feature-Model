// Transfer_XML2.0_To_IntArray.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <time.h>   
#include <iostream>  
#include <string>
#include <queue>  
#include <fstream>       
#include <vector> 

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/video/video.hpp>  
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/core/internal.hpp"

#include "cascadeclassifier.h"
#include "adi_haar_structure.h"

//#include "cv.h"
//#include "highgui.h"

struct CV_EXPORTS DTree
{
	int nodeCount;
};

struct CV_EXPORTS DTreeNode
{
	int featureIdx;
	float threshold; // for ordered features only
	int left;
	int right;
};

struct CV_EXPORTS Stage
{
	int first;
	int ntrees;
	float threshold;
};

#define THRESHOLD_EPS 0.0000099999997
#define RECT_NUM 3

using namespace std;
using namespace cv;

int _tmain(int argc, _TCHAR* argv[])
{	
	FileStorage fs;
	fs.open("facecascade.xml",FileStorage::READ);
	if (!fs.isOpened())
	{
		cout <<"Read xml failed!"<<endl;
		return -1;
	}

	FileNode root = fs.getFirstTopLevelNode();	//获取根节点


	string stageTypeStr = (string)root[CC_STAGE_TYPE];

	string stageType;
	if( stageTypeStr == CC_BOOST )
	{
		stageType = "BOOST";
	}
	else
	{
		stageType = "Unkown";
	}

	string featureTypeStr = (string)root[CC_FEATURE_TYPE];
	int featureType = 0;
	if( featureTypeStr == CC_HAAR )
		featureType = FeatureEvaluator::HAAR;
	else if( featureTypeStr == CC_LBP )
		featureType = FeatureEvaluator::LBP;
	else if( featureTypeStr == CC_HOG )
		featureType = FeatureEvaluator::HOG;

	int origWin_width = (int)root[CC_WIDTH];	//原始窗口的宽度
	int origWin_height = (int)root[CC_HEIGHT];	//原始窗口的高度

	// load feature params
	FileNode fn = root[CC_FEATURE_PARAMS];
	if( fn.empty() )
		return false;

	int ncategories = fn[CC_MAX_CAT_COUNT];
	int featSize = fn[CC_FEATURE_SIZE];
	int subsetSize = (ncategories + 31)/32,
		nodeStep = 3 + ( ncategories>0 ? subsetSize : 1 );

	// load stages
	fn = root[CC_STAGES];
	if( fn.empty() )
		return false;

	vector<Stage> stages;		//保存每个强分类器的阈值、每个强分类器中若分类器的个数
	vector<DTree> classifiers;	//保存每个弱分类器中haar_like特征的个数
	vector<DTreeNode> nodes;	//保存每个弱分类器的阈值、对应的特征编号featureIdx
	vector<float> leaves;		//每个弱分类器的左值和右值
	vector<int> subsets;

	stages.reserve(fn.size());
	classifiers.clear();
	nodes.clear();

	FileNodeIterator it = fn.begin(), it_end = fn.end();

	for( int si = 0; it != it_end; si++, ++it )
	{
		FileNode fns = *it;
		Stage stage;
		stage.threshold = (float)fns[CC_STAGE_THRESHOLD] - THRESHOLD_EPS;
		fns = fns[CC_WEAK_CLASSIFIERS];
		if(fns.empty())
			return false;
		stage.ntrees = (int)fns.size();
		stage.first = (int)classifiers.size();
		stages.push_back(stage);
		classifiers.reserve(stages[si].first + stages[si].ntrees);

		FileNodeIterator it1 = fns.begin(), it1_end = fns.end();
		for( ; it1 != it1_end; ++it1 ) // weak trees
		{
			FileNode fnw = *it1;
			FileNode internalNodes = fnw[CC_INTERNAL_NODES];
			FileNode leafValues = fnw[CC_LEAF_VALUES];
			if( internalNodes.empty() || leafValues.empty() )
				return false;

			DTree tree;
			tree.nodeCount = (int)internalNodes.size()/nodeStep;
			classifiers.push_back(tree);

			nodes.reserve(nodes.size() + tree.nodeCount);
			leaves.reserve(leaves.size() + leafValues.size());
			if( subsetSize > 0 )
				subsets.reserve(subsets.size() + tree.nodeCount*subsetSize);

			FileNodeIterator internalNodesIter = internalNodes.begin(), internalNodesEnd = internalNodes.end();

			for( ; internalNodesIter != internalNodesEnd; ) // nodes
			{
				DTreeNode node;
				node.left = (int)*internalNodesIter; ++internalNodesIter;
				node.right = (int)*internalNodesIter; ++internalNodesIter;
				node.featureIdx = (int)*internalNodesIter; ++internalNodesIter;
				if( subsetSize > 0 )
				{
					for( int j = 0; j < subsetSize; j++, ++internalNodesIter )
						subsets.push_back((int)*internalNodesIter);
					node.threshold = 0.f;
				}
				else
				{
					node.threshold = (float)*internalNodesIter; ++internalNodesIter;
				}
				nodes.push_back(node);
			}

			internalNodesIter = leafValues.begin(), internalNodesEnd = leafValues.end();

			for( ; internalNodesIter != internalNodesEnd; ++internalNodesIter ) // leaves
				leaves.push_back((float)*internalNodesIter);
		}
	}

	fn = root[CC_FEATURES];
	if( fn.empty() )
		return false;
	vector<CvHaarFeature>  haar_features;
	FileNodeIterator iter = fn.begin();
	for (; iter != fn.end(); ++iter)
	{
		FileNode fn_features = *iter;
		FileNode rnode = fn_features[CC_RECTS];
		FileNodeIterator it = rnode.begin(), it_end = rnode.end();

		CvHaarFeature haar_feature;
		int ri;
		for( ri = 0; ri < RECT_NUM; ri++ )
		{
			haar_feature.rect[ri].r = Rect();
			haar_feature.rect[ri].weight = 0.f;
		}

		for( ri = 0; it != it_end; ++it, ri++)
		{
			FileNodeIterator it2 = (*it).begin();
			it2 >> haar_feature.rect[ri].r.x >> haar_feature.rect[ri].r.y >>
				haar_feature.rect[ri].r.width >> haar_feature.rect[ri].r.height >> haar_feature.rect[ri].weight;
		}

		haar_feature.tilted = (int)fn_features[CC_TILTED] != 0;

		haar_features.push_back(haar_feature);
	}

	std::ofstream out;
	out.open ("haar_features_params.txt", std::ofstream::out | std::ofstream::app);	//创建一个txt文件，用于写入数据的，每次写入数据追加到文件尾
	//注意:C++中的几个stream流的浮点数默认精度设置都是6
	out.precision(16);	//设置写入的精度，默认是小数点后6位，这里设置为16位，为了移植到嵌入式平台用的。
	cout.precision(16);	//设置cout的精度，默认是小数点后6位，这里设置为16位。

	int m = 1;
	int n = -1;
	if (!out.is_open())
	{
		cout << "create result file failed!" << endl;
		return -1;
	}
	else
	{
		out << "#include \"stdafx.h\"" << endl;
		out << "#include \"adi_tool_chain.h\"" << endl;
		out << endl << endl;

		int stage_count = stages.size();
		out << "int32_t haar_features_params[] = {" << stage_count << "," << endl;

		int index = 0;
		for (int i = 0; i < stage_count; i++)
		{
			int classifier_count = stages[i].ntrees;
			out << classifier_count << "," << endl;

			for (int j = 0; j < classifier_count; j++)
			{
				int node_count = classifiers[index].nodeCount;
				out << node_count << "," << endl;

				int feature_index = nodes[index].featureIdx;
				int tilted = haar_features[feature_index].tilted;
				out << tilted << "," << endl;

				for (int k = 0; k < RECT_NUM; k++)
				{
					int x = haar_features[feature_index].rect[k].r.x;
					out << x << "," << endl;

					int y = haar_features[feature_index].rect[k].r.y;
					out << y << "," << endl;

					int width = haar_features[feature_index].rect[k].r.width;
					out << width << "," << endl;

					int height = haar_features[feature_index].rect[k].r.height;
					out << height << "," << endl;

					int weight = (int)(haar_features[feature_index].rect[k].weight * 65536);
					out << weight << "," << endl;
				}
				
				int node_threshold = (int)(nodes[index].threshold * 65536);
				out << node_threshold << "," << endl;

				int left = nodes[index].left;
				out << left << "," << endl;

				int right = nodes[index].right;
				out << right << "," << endl;

				int alpha_0 = (int)(leaves[2 * index] * 65536);
				out << alpha_0 << "," << endl;

				int alpha_1 = (int)(leaves[2 * index + 1] * 65536);
				out << alpha_1 << "," << endl;

				index++;
			}

			int stage_threshold = (int)(stages[i].threshold * 65536);
			out << stage_threshold <<  "," << endl;
		}
		out << "};" << endl;

		//print some info for ADI
		int classifier_count = index;
		int node_count = index;
		cout << "Total stage num = " << stage_count
			<< ", Total classifier num = " << classifier_count
			<< ", Total node num = " << node_count
			<< endl;

		int total_data_bytes = 4 * (1 + stage_count * 2 + classifier_count * 22);
		cout << "ADI_TRAINED_FILE_SIZE = " << total_data_bytes << endl;

		int total_structure_memory = sizeof(ADI_HAARCLASSIFIERCASCADE)
			+ sizeof(ADI_HAARSTAGECLASSIFIER) * stage_count
			+ sizeof(ADI_HAARCLASSIFIER) * classifier_count
			+ sizeof(ADI_HAARFEATURE) * node_count
			+ sizeof(int32_t *) * (node_count * 5)
			+ sizeof(ADI_PVT_CLASSIFIERCASCADE)
			+ sizeof(ADI_PVT_HAARSTAGECLASSIFIER) * stage_count
			+ sizeof(ADI_PVT_HAARCLASSIFIER) * classifier_count
			+ sizeof(ADI_PVT_HAARTREENODE) * node_count
			+ sizeof(void *) * (node_count + classifier_count);

		cout << "ADI_MEMORYFOR_TRAINEDDATA = " << total_structure_memory << endl;
	}

	out << endl << endl;
	out.close();

	return 0;
}



