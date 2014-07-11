/**
 * Illumination normalization.
 *
 * @blackball
 */

#include "inorm.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>

#define IPLIMAGE_TO_IMAGE(ipl, im)                                      \
        do {                                                            \
                (im)->data = (unsigned char*)(ipl)->imageData;          \
                (im)->w = (ipl)->width;                                 \
                (im)->ws = (ipl)->widthStep;                            \
                (im)->h = (ipl)->height;                                \
        } while (0)

static void 
test(const char *iname, const char *dstname) {
        
	IplImage *dst = NULL, *src= cvLoadImage(iname, 0); 
	
	int i = 0, j = 0, k = 0;

	struct image_t isrc, idst;

	if (!src) {
		return ;
	}

	dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
        
	IPLIMAGE_TO_IMAGE(src, &isrc);
	IPLIMAGE_TO_IMAGE(dst, &idst);

	inorm(&isrc, &idst);

        cvSaveImage(dstname, dst, 0);
        
	cvNamedWindow("src", 1);
	cvNamedWindow("dst", 1);
	cvShowImage("src", src);
	cvShowImage("dst", dst);
	cvWaitKey(0);
	cvDestroyWindow("src");
	cvDestroyWindow("dst");
        
	cvReleaseImage(&src);
	cvReleaseImage(&dst);
}

int 
main(int argc, char *argv[]) {
        if (argc != 3) {
                fprintf(stderr, "\n>>>inorm-test <src>.jpg <dst>.jpg\n\n");
                return 0;
        }
   	test(argv[1], argv[2]);
	return 0;
}
