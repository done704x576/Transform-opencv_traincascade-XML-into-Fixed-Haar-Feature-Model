#ifndef __ADI_HAAR_STRUCTURES__
#define __ADI_HAAR_STRUCTURES__

typedef signed int          int32_t;
typedef float               float32_t;
typedef unsigned short      uint16_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t; 

/* Object type for internal rectangle feature */
typedef struct adi_PvtHaarRectangle
{
	uint32_t    *p0, *p1, *p2, *p3;
	int32_t     nWeight;
} ADI_PVT_HAARRECTANGLE;

/* Object type for internal haar feature */
typedef struct adi_Pvt_HaarFeature
{
	ADI_PVT_HAARRECTANGLE   oPvtHaarRect[3];
} ADI_PVT_HAARFEATURE;


/* Object type for internal haar node */
typedef struct adi_PvtHaarTreeNode
{
	ADI_PVT_HAARFEATURE oFeature;
	int32_t             nThreshold;
	int32_t             nLeft;
	int32_t             nRight;
} ADI_PVT_HAARTREENODE;

/* Object type for internal haar classifier */
typedef struct adi_PvtHaarClassifier
{
	int32_t                 nCount;
	ADI_PVT_HAARTREENODE    *pNode;
	int32_t                 *pAlpha;
} ADI_PVT_HAARCLASSIFIER;

/* Object type for image dimension */
typedef struct adi_ImageSize
{
	uint32_t    nWidth;
	uint32_t    nHeight;
} ADI_IMAGE_SIZE;

/* Object type for internal haar stage classifier */
typedef struct adi_PvtHaarStageClassifier
{
	int32_t                 nCount;
	int32_t                 nThreshold;
	ADI_PVT_HAARCLASSIFIER  *pClassifier;
	int32_t                 nTwoRects;
} ADI_PVT_HAARSTAGECLASSIFIER;

/* Object type for internal cascade classifier */
typedef struct adi_PvtClassifierCascade
{
	int32_t                     nCount;
	int32_t                     nInverseWindowArea;
	uint32_t                    *pSum;
	uint64_t                    *pSqSum;
	ADI_PVT_HAARSTAGECLASSIFIER *pStageClassifier;
	uint64_t                    *pq0, *pq1, *pq2, *pq3;
	uint32_t                    *p0, *p1, *p2, *p3;
} ADI_PVT_CLASSIFIERCASCADE;


//rect
typedef struct adi_Rectangle
{
	uint16_t    nX;
	uint16_t    nY;
	uint16_t    nWidth;
	uint16_t    nHeight;
} ADI_RECTANGLE;

//rect and weight
typedef struct adi_HaarRectangle
{
	ADI_RECTANGLE   nRect;	//矩形框
	int32_t         nWeight;//权重
} ADI_HAARRECTANGLE;

//haar feature
typedef struct ADI_HaarFeature
{
	int32_t             nTilted;		  //是否倾斜
	ADI_HAARRECTANGLE   nHaarRectangle[3];//矩形框
} ADI_HAARFEATURE;

//node tree
typedef struct ADI_HaarClassifier
{
	int32_t         nCount;		  //haar 特征的个数
	ADI_HAARFEATURE *pHaarFeature;//haar特征
	int32_t         *pThreshold;
	int32_t         *pLeft;
	int32_t         *pRight;
	int32_t         *pAlpha;
} ADI_HAARCLASSIFIER;

//stage classifier
typedef struct ADI_HaarStageClassifier
{
	int32_t             nCount;		 //弱分类器中的节点个数
	int32_t             nThreshold;	 //弱分类器的阈值
	ADI_HAARCLASSIFIER  *pClassifier;//弱分类器的节点
} ADI_HAARSTAGECLASSIFIER;

//cascade classifier
typedef struct adi_HaarClassifierCascade
{
	int32_t                     nCount;				//强分类器的个数
	ADI_IMAGE_SIZE              oOriginalWindowSize;//原始检测框的尺寸
	ADI_IMAGE_SIZE              oRealWindowSize;	//实际检测框的尺寸
	float32_t                   nScale;				//检测框的扩大系数
	ADI_HAARSTAGECLASSIFIER     *pStageClassifier;	//强分类器
	ADI_PVT_CLASSIFIERCASCADE   *pPvtCascade;
} ADI_HAARCLASSIFIERCASCADE;



#endif // !__ADI_HAAR_STRUCTURES__
